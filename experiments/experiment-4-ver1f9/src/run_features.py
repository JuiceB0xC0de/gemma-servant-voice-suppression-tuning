"""GPU pipeline: SAE-feature rescoring, feature clamping vs the raw direction, exemplars, SFT shift.

Stage 1  token-level rescoring of the layer {4,10,22} JumpReLU SAEs on the 600 matched pairs
         (Bella reply vs Gemma's own reply to the same prompt): per-feature firing rate, mean
         activation and Cohen's d per side, joined with the atlas align/shift; real-token L0 and EV;
         feature-set selection (A_k Assistant-ward by align, D_k by token-level d, matched random R_k).
Stage 4  Bella fine-tune minus base mean residual shift per layer on the same prompts with the Gemma
         reply teacher-forced; cos with the Bella-minus-Gemma direction and the fraction of the gap closed.
Stage 2  generation grid: unsteered, raw direction at -0.35 / -0.5 (regenerated), feature clamping
         cells at layer 10 (A_k, D_k, R_k) and layer 4 (A_k, R_k); 290 chat prompts x 128 tokens plus
         100 neutral stems x 64 tokens per cell, greedy; dose response on the 100 test pairs.
Stage 3  exemplars + logit lens for the top-50 layer-10 features (by align) plus A_20 and 78793.

Layer index convention (matches the axis experiment and the SAE trainer): "layer l" = output of
decoder block l (0-indexed) = hidden_states[l+1].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:  # platform telemetry (best effort)
    from silico.slurm_telemetry import report_progress
except Exception:  # pragma: no cover
    def report_progress(**kw):
        return None

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "results" / "data"
PRIOR = HERE / "results" / "prior"
BASE_HF, BASE_REV = "google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2"
FT_HF, FT_REV = "juiceb0xc0de/bella-bartender-gemma-e4b", "ac5ff55239c029c461a41bd2788d7423d0f2ac7c"
SAE_REPO, SAE_REV = "juiceb0xc0de/gemma-4-e4b-SAE", "57bacb61a1bcc32be212a815b985a9bb42ff9a16"
ATLAS_BUCKET = "https://huggingface.co/buckets/juiceb0xc0de/atlas-runs/resolve/atlas-gemma-4-e4b/rlhf/"
SEED = 42
SAE_LAYERS = [4, 10, 22]
FIRE_MIN = 0.01  # feature must fire on >= 1% of Gemma reply tokens (train split) to be selectable
K_GRID = [5, 20, 50, 200]
NAMED_FEATURE = 78793

DEFAULT_CELLS = (
    ["base", "dir:10:-0.35", "dir:10:-0.5"]
    + [f"A:10:{k}" for k in K_GRID] + [f"D:10:{k}" for k in K_GRID]
    + ["R:10:50:0", "R:10:50:1", "R:10:200:0", "R:10:200:1"]
    + [f"A:4:{k}" for k in K_GRID] + ["R:4:50:0", "R:4:200:0"]
)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, p)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- model


class Wrapped:
    def __init__(self, hf_id, revision, tok_from=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        tid, trev = tok_from or (hf_id, revision)
        self.tok = AutoTokenizer.from_pretrained(tid, revision=trev)
        if self.tok.chat_template is None:
            from transformers import AutoProcessor
            self.tok.chat_template = AutoProcessor.from_pretrained(tid, revision=trev).chat_template
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        cfg = AutoConfig.from_pretrained(hf_id, revision=revision)
        self.text_cfg = getattr(cfg, "text_config", cfg)
        self.n_layers = self.text_cfg.num_hidden_layers
        self.d = self.text_cfg.hidden_size
        self.layer_types = list(getattr(self.text_cfg, "layer_types", []))
        self.global_layers = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision, dtype=torch.bfloat16,
                                                          device_map="cuda", attn_implementation="sdpa")
        self.model.eval()
        assert self.model.dtype == torch.bfloat16, self.model.dtype
        cands = [(n, m) for n, m in self.model.named_modules()
                 if isinstance(m, torch.nn.ModuleList) and len(m) == self.n_layers
                 and all(hasattr(x, "self_attn") for x in m)]
        assert cands, "decoder layer container not found"
        self.layers_name, self.layers = cands[0]
        parent = self.model.get_submodule(self.layers_name.rsplit(".", 1)[0])
        self.final_norm = getattr(parent, "norm")
        self.lm_head = self.model.get_output_embeddings()
        log(f"model {type(self.model).__name__} ({hf_id}@{revision[:8]}) layers at {self.layers_name} n={self.n_layers} "
            f"d={self.d} global={self.global_layers} dtype={self.model.dtype} head={tuple(self.lm_head.weight.shape)}")
        self.gen_cfg = dict(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                            pad_token_id=self.tok.pad_token_id)
        self._handles = []
        self.eot_ids = set(self.tok.convert_tokens_to_ids(t) for t in ["<end_of_turn>", "<turn|>", "<|turn>"]
                           if t in self.tok.get_vocab())
        self.eot_ids.add(self.tok.eos_token_id)

    def chat_prefix(self, prompt: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True)

    def clear_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def add_hook(self, layer, fn):
        self._handles.append(self.layers[layer].register_forward_hook(fn))

    @torch.inference_mode()
    def generate(self, texts, max_new_tokens, batch_size=48, phase="generate", chat=True):
        prefixes = [self.chat_prefix(t) if chat else (self.tok.bos_token or "") + t for t in texts]
        order = sorted(range(len(prefixes)), key=lambda i: -len(prefixes[i]))
        out = [None] * len(texts)
        for bi in range(0, len(order), batch_size):
            idx = order[bi:bi + batch_size]
            self.tok.padding_side = "left"
            enc = self.tok([prefixes[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.cuda() for k, v in enc.items()}
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, **self.gen_cfg)
            new = gen[:, enc["input_ids"].shape[1]:]
            for j, i in enumerate(idx):
                ids = new[j].tolist()
                cut = len(ids)
                for k, t in enumerate(ids):
                    if t in self.eot_ids or t == self.tok.pad_token_id:
                        cut = k
                        break
                ids = ids[:cut]
                out[i] = {"text": self.tok.decode(ids, skip_special_tokens=True).strip(), "n_tokens": len(ids),
                          "truncated": cut == max_new_tokens}
            report_progress(step=min(bi + batch_size, len(order)), total_steps=len(order), phase=phase)
        return out

    @torch.inference_mode()
    def harvest(self, prefixes, replies, batch_size=16, phase="harvest", token_layers=(), on_tokens=None,
                keep_ids=False):
        """Teacher-forced forward over prefix+reply. Returns mean block outputs over reply tokens
        [N, L, d] fp32, per-reply token counts, per-layer median token norm. For each layer in
        token_layers calls on_tokens(layer, h_tokens[n_reply_tokens, d] fp32, reply_index_per_token, batch_idx).
        keep_ids -> also returns the reply token ids per reply."""
        N, L, d = len(prefixes), self.n_layers, self.d
        mean_all = np.zeros((N, L, d), np.float32)
        n_tok = np.zeros(N, np.int32)
        norm_samples = [[] for _ in range(L)]
        reply_ids = [None] * N
        store = {}

        def mk(l):
            def hook(mod, inp, out):
                store[l] = out[0] if isinstance(out, tuple) else out
            return hook

        self.clear_hooks()
        for l in range(L):
            self.add_hook(l, mk(l))
        try:
            full = [p + r for p, r in zip(prefixes, replies)]
            order = sorted(range(N), key=lambda i: -len(full[i]))
            for bi in range(0, N, batch_size):
                idx = order[bi:bi + batch_size]
                self.tok.padding_side = "right"
                enc = self.tok([full[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
                plen = torch.tensor([len(self.tok(prefixes[i], add_special_tokens=False)["input_ids"]) for i in idx])
                am = enc["attention_mask"]
                T = am.shape[1]
                pos = torch.arange(T)[None, :]
                reply_mask = (pos >= plen[:, None]) & (am == 1)
                self.model(**{k: v.cuda() for k, v in enc.items()}, use_cache=False)
                rm_cuda = reply_mask.cuda()
                rmf = rm_cuda.unsqueeze(-1).float()
                for l in range(L):
                    h = store[l].float()
                    ma = (h * rmf).sum(1) / rmf.sum(1).clamp(min=1)
                    mean_all[idx, l] = ma.cpu().numpy()
                    if len(norm_samples[l]) < 20000:
                        norm_samples[l].extend(h.norm(dim=-1)[rm_cuda][:2000].cpu().tolist())
                    if l in token_layers and on_tokens is not None:
                        rep_idx = torch.tensor(idx).cuda()[:, None].expand(-1, T)[rm_cuda]
                        on_tokens(l, h[rm_cuda], rep_idx, idx)
                for j, i in enumerate(idx):
                    n_tok[i] = int(reply_mask[j].sum())
                    if keep_ids:
                        reply_ids[i] = enc["input_ids"][j][reply_mask[j]].tolist()
                store.clear()
                report_progress(step=min(bi + batch_size, N), total_steps=N, phase=phase)
        finally:
            self.clear_hooks()
        res = {"mean_all": mean_all, "n_tokens": n_tok,
               "median_norm": np.array([float(np.median(s)) if s else float("nan") for s in norm_samples])}
        if keep_ids:
            res["reply_ids"] = reply_ids
        return res

    @torch.inference_mode()
    def mean_logprob(self, prefixes, continuations, batch_size=16):
        N = len(prefixes)
        out = np.zeros(N, np.float64)
        ntok = np.zeros(N, np.int32)
        full = [p + c for p, c in zip(prefixes, continuations)]
        order = sorted(range(N), key=lambda i: -len(full[i]))
        for bi in range(0, N, batch_size):
            idx = order[bi:bi + batch_size]
            self.tok.padding_side = "right"
            enc = self.tok([full[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
            plen = [len(self.tok(prefixes[i], add_special_tokens=False)["input_ids"]) for i in idx]
            ids = enc["input_ids"].cuda()
            am = enc["attention_mask"].cuda()
            logits = self.model(input_ids=ids, attention_mask=am, use_cache=False).logits.float()
            lp = F.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
            T = ids.shape[1]
            pos = torch.arange(1, T, device=ids.device)[None, :]
            m = (pos >= torch.tensor(plen, device=ids.device)[:, None]) & (am[:, 1:] == 1)
            s = (lp * m).sum(1)
            n = m.sum(1)
            for j, i in enumerate(idx):
                out[i] = float(s[j] / n[j].clamp(min=1))
                ntok[i] = int(n[j])
        return out, ntok


# --------------------------------------------------------------------------- SAE


def load_sae(layer: int):
    """event-aware-SAE-trainer convention (examples/use_trained_sae.py): W_enc.weight [F, d] (F.linear),
    W_enc.bias [F], W_dec.weight [d, F], b_dec [d], log_threshold [F]; pre = W_enc(x - b_dec) + b_enc,
    z = pre * (pre > exp(log_threshold)); x_hat = W_dec z + b_dec."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(SAE_REPO, f"layer_{layer:02d}_s0/sae.pt", repo_type="dataset", revision=SAE_REV)
    m = hf_hub_download(SAE_REPO, f"layer_{layer:02d}_s0/meta.json", repo_type="dataset", revision=SAE_REV)
    meta = json.load(open(m))
    sd = torch.load(p, map_location="cpu", weights_only=True)
    d_in, n_feat = meta["d_in"], meta["n_features"]
    W_enc, W_dec = sd["W_enc.weight"].float(), sd["W_dec.weight"].float()
    assert tuple(W_enc.shape) == (n_feat, d_in), W_enc.shape
    assert tuple(W_dec.shape) == (d_in, n_feat), W_dec.shape
    b_enc = sd.get("W_enc.bias", torch.zeros(n_feat)).float()
    b_dec = sd.get("b_dec", torch.zeros(d_in)).float()
    assert "log_threshold" in sd, list(sd.keys())
    thr = sd["log_threshold"].float().exp()
    sae = {"W_enc": W_enc.cuda(), "b_enc": b_enc.cuda(), "W_dec": W_dec.cuda(), "b_dec": b_dec.cuda(), "thr": thr.cuda(),
           "meta": meta, "keys": sorted(sd.keys()), "n_feat": n_feat, "d_in": d_in, "layer": layer, "file_sha256": sha256(p)}
    log(f"SAE L{layer}: keys={sae['keys']} thr mean={float(thr.mean()):.3f} dec col norm mean={float(W_dec.norm(dim=0).mean()):.3f} "
        f"meta L0={meta.get('final_metrics', {}).get('mean_l0')} EV={meta.get('final_metrics', {}).get('ev')}")
    return sae


def sae_encode(sae, x):
    pre = F.linear(x - sae["b_dec"], sae["W_enc"], sae["b_enc"])
    return pre * (pre > sae["thr"])


def sae_decode(sae, z):
    return F.linear(z, sae["W_dec"], sae["b_dec"])


class FeatureEdit:
    """Forward hook that edits a set of SAE features in the block output:
    scale mode: h <- h - (1 - scale) * sum_f z_f(h) W_dec[:, f]   (scale=0 clamps to zero)
    amp mode:   h <- h + (factor - 1) * sum_f z_f(h) W_dec[:, f]
    Applied at every position of every forward (prefill and decode), like the direction hook."""

    def __init__(self, sae, feats, scale=0.0, amp=None):
        f = torch.tensor(sorted(int(x) for x in feats)).cuda()
        self.feats = f
        self.We = sae["W_enc"][f]          # [k, d]
        self.be = sae["b_enc"][f]
        self.thr = sae["thr"][f]
        self.Wd = sae["W_dec"][:, f]       # [d, k]
        self.b_dec = sae["b_dec"]
        self.mult = (amp - 1.0) if amp is not None else -(1.0 - scale)
        self.reset()

    def reset(self):
        self.dec_pos = 0
        self.dec_active = 0.0
        self.dec_zsum = 0.0
        self.dec_per_feat = torch.zeros(len(self.feats), device="cuda")
        self.pre_pos = 0
        self.pre_active = 0.0

    def __call__(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        x = h.float()
        pre = F.linear(x - self.b_dec, self.We, self.be)
        z = pre * (pre > self.thr)
        delta = F.linear(z, self.Wd) * self.mult  # [B, T, d]
        active = (z > 0)
        if x.shape[1] == 1:  # decode step: every position is a real generated token
            self.dec_pos += x.shape[0]
            self.dec_active += float(active.sum())
            self.dec_zsum += float(z.sum())
            self.dec_per_feat += active.sum((0, 1)).float()
        else:
            self.pre_pos += x.shape[0] * x.shape[1]
            self.pre_active += float(active.sum())
        new = (x + delta).to(h.dtype)
        if isinstance(out, tuple):
            return (new,) + tuple(out[1:])
        return new

    def stats(self):
        return {"n_features": int(len(self.feats)), "decode_positions": self.dec_pos,
                "mean_active_per_generated_token": (self.dec_active / self.dec_pos) if self.dec_pos else None,
                "mean_z_sum_per_generated_token": (self.dec_zsum / self.dec_pos) if self.dec_pos else None,
                "prefill_positions_incl_padding": self.pre_pos,
                "mean_active_per_prefill_position": (self.pre_active / self.pre_pos) if self.pre_pos else None,
                "per_feature_fire_rate_generated": {int(f): float(c / max(self.dec_pos, 1)) for f, c in
                                                    zip(self.feats.tolist(), self.dec_per_feat.tolist())}}


class DirectionAdd:
    def __init__(self, vec):
        self.v = vec.to(torch.bfloat16).cuda()

    def __call__(self, mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] + self.v,) + tuple(out[1:])
        return out + self.v


# --------------------------------------------------------------------------- inputs


def fetch_inputs(out: Path):
    """Direction + atlas scores (small) into out/inputs. Returns (dirs [42, d], atlas dict per layer)."""
    import urllib.request
    from huggingface_hub import hf_hub_download
    inp = out / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(SAE_REPO, "directions/e4b_directions.npz", repo_type="dataset", revision=SAE_REV)
    shutil.copy(p, inp / "e4b_directions.npz")
    dz = np.load(inp / "e4b_directions.npz")
    dirs = dz["all"].astype(np.float32)
    assert dirs.shape[0] == 42 and abs(np.linalg.norm(dirs[10]) - 1) < 1e-3
    tok = os.environ["HF_TOKEN"]
    atlas = {}
    for l in SAE_LAYERS:
        name = f"l{l}_rlhf_scores.npz"
        dst = inp / name
        if not dst.exists():
            req = urllib.request.Request(ATLAS_BUCKET + name, headers={"Authorization": "Bearer " + tok})
            with urllib.request.urlopen(req, timeout=300) as r, dst.open("wb") as f:
                shutil.copyfileobj(r, f)
        z = np.load(dst)
        assert {"align", "shift", "base_fire", "bella_fire", "base_mean", "bella_mean"} <= set(z.files), z.files
        atlas[l] = {k: z[k] for k in z.files}
        assert atlas[l]["align"].shape == (81920,)
    req = urllib.request.Request(ATLAS_BUCKET + "rlhf_summary.json", headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=120) as r, (inp / "rlhf_summary.json").open("wb") as f:
        shutil.copyfileobj(r, f)
    return dirs, atlas


def load_pairs(args):
    pairs = read_jsonl(DATA / "pairs.jsonl")
    gem = {g["pid"]: g for g in read_jsonl(PRIOR / "gemma_replies.jsonl")}
    assert len(pairs) == 600 and all(p["pid"] in gem for p in pairs)
    if args.smoke:
        pairs = pairs[: args.smoke_n]
    return pairs, gem


# --------------------------------------------------------------------------- stats


def cohens_d(mean_a, var_a, mean_b, var_b):
    sp = np.sqrt((var_a + var_b) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(sp > 0, (mean_a - mean_b) / sp, 0.0)
    return d


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return float(r.correlation), float(r.pvalue)


# --------------------------------------------------------------------------- stage 1


class FeatStats:
    """Streaming per-feature statistics for one (layer, side): token sums per split + per-reply means."""

    def __init__(self, n_feat, n_replies, splits):
        self.splits = splits  # np array of split labels per reply index
        self.names = sorted(set(splits.tolist()))
        self.sum = {s: torch.zeros(n_feat, device="cuda", dtype=torch.float64) for s in self.names}
        self.sum2 = {s: torch.zeros(n_feat, device="cuda", dtype=torch.float64) for s in self.names}
        self.act = {s: torch.zeros(n_feat, device="cuda", dtype=torch.float64) for s in self.names}
        self.n = {s: 0 for s in self.names}
        self.reply_mean = torch.zeros(n_replies, n_feat, device="cuda", dtype=torch.float32)
        self.reply_n = torch.zeros(n_replies, device="cuda")
        self.l0_sum = 0.0
        self.ntok = 0
        self.sq_err = 0.0
        self.sq_tot = 0.0
        self.sum_x = torch.zeros(1, device="cuda", dtype=torch.float64)

    def update(self, sae, h, rep_idx):
        """h [n_tok, d] fp32; rep_idx [n_tok] reply indices."""
        z = sae_encode(sae, h)
        xhat = sae_decode(sae, z)
        self.sq_err += float(((h - xhat) ** 2).sum())
        self.sq_tot += float(((h - h.mean()) ** 2).sum())
        self.l0_sum += float((z > 0).sum())
        self.ntok += h.shape[0]
        # per-reply means (scatter-add)
        self.reply_mean.index_add_(0, rep_idx, z)
        self.reply_n.index_add_(0, rep_idx, torch.ones_like(rep_idx, dtype=torch.float32))
        sp = self.splits[rep_idx.cpu().numpy()]
        for s in self.names:
            m = torch.tensor(sp == s).cuda()
            if m.any():
                zs = z[m]
                self.sum[s] += zs.sum(0).double()
                self.sum2[s] += (zs ** 2).sum(0).double()
                self.act[s] += (zs > 0).sum(0).double()
                self.n[s] += int(m.sum())

    def finalize(self):
        self.reply_mean /= self.reply_n.clamp(min=1)[:, None]
        out = {}
        for s in self.names:
            n = max(self.n[s], 1)
            mean = self.sum[s] / n
            var = (self.sum2[s] / n - mean ** 2).clamp(min=0)
            out[s] = {"mean": mean.float().cpu().numpy(), "var": var.float().cpu().numpy(),
                      "fire": (self.act[s] / n).float().cpu().numpy(), "n_tokens": self.n[s]}
        allm = sum(self.sum[s] for s in self.names) / max(self.ntok, 1)
        allv = (sum(self.sum2[s] for s in self.names) / max(self.ntok, 1) - allm ** 2).clamp(min=0)
        out["all"] = {"mean": allm.float().cpu().numpy(), "var": allv.float().cpu().numpy(),
                      "fire": (sum(self.act[s] for s in self.names) / max(self.ntok, 1)).float().cpu().numpy(), "n_tokens": self.ntok}
        out["l0"] = self.l0_sum / max(self.ntok, 1)
        out["ev"] = 1 - self.sq_err / max(self.sq_tot, 1e-9)
        return out


def select_sets(align, d_tok, fire_gemma_train, rng_seed):
    """A_k: most negative align among features with fire >= FIRE_MIN; D_k: most negative d (Gemma > Bella);
    R_k: random draws matched to A_k's firing-rate distribution (log10 bins, width 0.25), excluding A_200 u D_200."""
    pool = np.where(fire_gemma_train >= FIRE_MIN)[0]
    order_a = pool[np.argsort(align[pool])]
    order_d = pool[np.argsort(d_tok[pool])]
    sets = {"pool_size": int(len(pool)), "A": {}, "D": {}, "R": {}, "R_bin_width_hist": {}}
    kmax = max(K_GRID)
    for k in K_GRID:
        sets["A"][k] = [int(x) for x in order_a[:k]]
        sets["D"][k] = [int(x) for x in order_d[:k]]
    excl = set(order_a[:kmax].tolist()) | set(order_d[:kmax].tolist())
    lf = np.log10(np.clip(fire_gemma_train, 1e-6, 1))
    bins = np.floor(lf / 0.25)
    for k in (50, 200):
        for draw in (0, 1):
            rng = np.random.default_rng(rng_seed + 1000 * k + draw)
            chosen = []
            used = set(excl)
            widen_hist = {}
            for f in sets["A"][k]:
                width = 0
                while True:
                    cands = [c for c in np.where((np.abs(bins - bins[f]) <= width) & (fire_gemma_train >= FIRE_MIN))[0] if c not in used]
                    if cands or width > 40:
                        break
                    width = 1 if width == 0 else width * 2  # widen the firing-rate window progressively
                if not cands:  # last resort: any eligible, unused feature
                    cands = [c for c in pool if c not in used]
                    width = -1
                if not cands:
                    raise RuntimeError(f"random control: eligible pool ({len(pool)}) exhausted at k={k}")
                widen_hist[str(width)] = widen_hist.get(str(width), 0) + 1
                c = int(rng.choice(cands))
                chosen.append(c)
                used.add(c)
            sets["R"][f"{k}:{draw}"] = chosen
            sets["R_bin_width_hist"][f"{k}:{draw}"] = widen_hist
    return sets


def stage1(W: Wrapped, out: Path, args, dirs, atlas):
    pairs, gem = load_pairs(args)
    N = len(pairs)
    splits = np.array([p["split"] for p in pairs])
    prompts = [p["prompt"] for p in pairs]
    prefixes = [W.chat_prefix(p) for p in prompts]
    t0 = time.time()
    saes = {l: load_sae(l) for l in SAE_LAYERS}
    stats = {l: {"bella": FeatStats(81920, N, splits), "gemma": FeatStats(81920, N, splits)} for l in SAE_LAYERS}
    keep10 = {"bella": {}, "gemma": {}}  # reply idx -> bf16 residual tokens at layer 10 (for exemplars)
    harv = {}
    for side, replies in (("bella", [p["bella"] for p in pairs]), ("gemma", [gem[p["pid"]]["text"] or " " for p in pairs])):
        def on_tokens(l, h, rep_idx, idx, side=side):
            stats[l][side].update(saes[l], h, rep_idx)
            if l == 10:
                ri = rep_idx.cpu().numpy()
                hc = h.to(torch.bfloat16).cpu()
                for i in idx:
                    keep10[side][i] = hc[torch.tensor(ri == i)]
        log(f"stage1: harvesting {side} side ({N} replies)")
        harv[side] = W.harvest(prefixes, replies, batch_size=args.harvest_batch, phase=f"s1_harvest_{side}",
                               token_layers=SAE_LAYERS, on_tokens=on_tokens, keep_ids=True)
    res = {"n_pairs": N, "splits": {s: int((splits == s).sum()) for s in ("train", "val", "test")},
           "n_tok_bella_mean": float(harv["bella"]["n_tokens"].mean()), "n_tok_gemma_mean": float(harv["gemma"]["n_tokens"].mean()),
           "median_norm_gemma": harv["gemma"]["median_norm"].tolist(), "median_norm_bella": harv["bella"]["median_norm"].tolist(),
           "per_layer": {}, "fire_min": FIRE_MIN, "k_grid": K_GRID}
    # gap_l for stage 4: Bella-minus-Gemma mean projection on the direction, per split
    B, G = harv["bella"]["mean_all"], harv["gemma"]["mean_all"]
    proj_b = np.einsum("nld,ld->nl", B, dirs)
    proj_g = np.einsum("nld,ld->nl", G, dirs)
    res["gap"] = {"all": (proj_b - proj_g).mean(0).tolist(),
                  "test": (proj_b - proj_g)[splits == "test"].mean(0).tolist() if (splits == "test").any() else None,
                  "train": (proj_b - proj_g)[splits == "train"].mean(0).tolist() if (splits == "train").any() else None,
                  "gemma_mean_proj_all": proj_g.mean(0).tolist(), "bella_mean_proj_all": proj_b.mean(0).tolist()}
    np.savez_compressed(out / "acts_means.npz", bella_mean=B.mean(0), gemma_mean=G.mean(0),
                        proj_bella=proj_b, proj_gemma=proj_g, splits=splits,
                        bella_mean_test=B[splits == "test"].mean(0) if (splits == "test").any() else B.mean(0),
                        gemma_mean_test=G[splits == "test"].mean(0) if (splits == "test").any() else G.mean(0))
    # the axis experiment's cross-check numbers (its stage1: mean projection / median norm on the test split)
    try:
        ps1 = json.load(open(PRIOR / "stage1.json"))
        A = ps1["pooling"]["all"]
        res["gap_prior_test"] = [(A["mean_proj_bella_test"][l] - A["mean_proj_gemma_test"][l]) * ps1["median_norm_gemma"][l]
                                 for l in range(42)]
    except Exception as e:  # pragma: no cover
        res["gap_prior_test"] = None
        log("prior stage1 not available:", repr(e))

    sets_by_layer = {}
    for l in SAE_LAYERS:
        sb, sg = stats[l]["bella"].finalize(), stats[l]["gemma"].finalize()
        tr = "train" if "train" in sb else "all"
        te = "test" if "test" in sb else "all"
        d_tok_train = cohens_d(sb[tr]["mean"], sb[tr]["var"], sg[tr]["mean"], sg[tr]["var"])
        d_tok_test = cohens_d(sb[te]["mean"], sb[te]["var"], sg[te]["mean"], sg[te]["var"])
        # reply-level d (per-reply mean activation), on the test split
        rb, rg = stats[l]["bella"].reply_mean, stats[l]["gemma"].reply_mean
        tm = torch.tensor(splits == te).cuda() if te != "all" else torch.ones(N, dtype=torch.bool).cuda()
        d_reply_test = cohens_d(rb[tm].mean(0).cpu().numpy(), rb[tm].var(0, unbiased=False).cpu().numpy(),
                                rg[tm].mean(0).cpu().numpy(), rg[tm].var(0, unbiased=False).cpu().numpy())
        frac_replies_active_b = (rb > 0).float().mean(0).cpu().numpy()
        frac_replies_active_g = (rg > 0).float().mean(0).cpu().numpy()
        al = atlas[l]["align"].astype(np.float32)
        # align recomputed here from the decoder columns and the direction (checks the convention)
        Wd = saes[l]["W_dec"]
        v = torch.tensor(dirs[l]).cuda()
        my_align = ((Wd / Wd.norm(dim=0, keepdim=True).clamp(min=1e-8)).T @ v).cpu().numpy()
        align_check = {"pearson_with_atlas": float(np.corrcoef(my_align, al)[0, 1]),
                       "max_abs_diff": float(np.abs(my_align - al).max()),
                       "top1_atlas": int(np.argmax(np.abs(al))), "top1_here": int(np.argmax(np.abs(my_align)))}
        np.savez_compressed(out / f"features_L{l}.npz", align=al, my_align=my_align.astype(np.float32),
                            shift=atlas[l]["shift"], atlas_base_fire=atlas[l]["base_fire"], atlas_bella_fire=atlas[l]["bella_fire"],
                            fire_bella_train=sb[tr]["fire"], fire_gemma_train=sg[tr]["fire"],
                            fire_bella_test=sb[te]["fire"], fire_gemma_test=sg[te]["fire"],
                            mean_bella_train=sb[tr]["mean"], mean_gemma_train=sg[tr]["mean"],
                            mean_bella_test=sb[te]["mean"], mean_gemma_test=sg[te]["mean"],
                            d_tok_train=d_tok_train, d_tok_test=d_tok_test, d_reply_test=d_reply_test,
                            frac_replies_active_bella=frac_replies_active_b, frac_replies_active_gemma=frac_replies_active_g,
                            dec_norm=Wd.norm(dim=0).cpu().numpy())
        sets = select_sets(al, d_tok_train, sg[tr]["fire"], SEED)
        sets_by_layer[l] = sets
        top_d = np.argsort(-np.abs(d_tok_train))[:200]
        top_a = np.argsort(-np.abs(al))[:200]
        union = np.array(sorted(set(top_d.tolist()) | set(top_a.tolist())))
        rho, pval = spearman(d_tok_train[union], al[union])
        rho_all, _ = spearman(d_tok_train, al)
        rho_test, _ = spearman(d_tok_test[union], al[union])
        firing = sg[tr]["fire"]
        res["per_layer"][str(l)] = {
            "l0_bella": sb["l0"], "l0_gemma": sg["l0"], "l0_all": (sb["l0"] * sb["all"]["n_tokens"] + sg["l0"] * sg["all"]["n_tokens"]) / (sb["all"]["n_tokens"] + sg["all"]["n_tokens"]),
            "ev_bella": sb["ev"], "ev_gemma": sg["ev"], "meta_l0": saes[l]["meta"].get("final_metrics", {}).get("mean_l0"),
            "meta_ev": saes[l]["meta"].get("final_metrics", {}).get("ev"), "atlas_l0_base": None,
            "n_tokens": {"bella": sb["all"]["n_tokens"], "gemma": sg["all"]["n_tokens"]},
            "align_check": align_check, "sae_keys": saes[l]["keys"], "sae_sha256": saes[l]["file_sha256"],
            "spearman_d_vs_align_union_top200": {"rho": rho, "p": pval, "n": int(len(union))},
            "spearman_d_vs_align_all": rho_all, "spearman_dtest_vs_align_union_top200": rho_test,
            "pool_size_fire_ge_1pct": sets["pool_size"],
            "n_features_abs_dtok_gt_0p2_train": int((np.abs(d_tok_train) > 0.2).sum()),
            "n_features_abs_dreply_gt_0p5_test": int((np.abs(d_reply_test) > 0.5).sum()),
            "sets": {"A": {str(k): v for k, v in sets["A"].items()}, "D": {str(k): v for k, v in sets["D"].items()}, "R": sets["R"]},
            "A_align": {str(k): [float(al[f]) for f in sets["A"][k]] for k in K_GRID},
            "A_fire_gemma_train": {str(k): [float(firing[f]) for f in sets["A"][k]] for k in K_GRID},
            "A_d_tok_test": {str(k): [float(d_tok_test[f]) for f in sets["A"][k]] for k in K_GRID},
            "D_d_tok_train": {str(k): [float(d_tok_train[f]) for f in sets["D"][k]] for k in K_GRID},
            "D_d_tok_test": {str(k): [float(d_tok_test[f]) for f in sets["D"][k]] for k in K_GRID},
            "D_align": {str(k): [float(al[f]) for f in sets["D"][k]] for k in K_GRID},
            "R_fire_gemma_train": {k: [float(firing[f]) for f in v] for k, v in sets["R"].items()},
            "R_bin_width_hist": sets["R_bin_width_hist"],
            "overlap_A_D": {str(k): len(set(sets["A"][k]) & set(sets["D"][k])) for k in K_GRID},
            "top_align_neg": [{"f": int(f), "align": float(al[f]), "fire_gemma": float(firing[f]), "d_tok_test": float(d_tok_test[f]), "shift": float(atlas[l]["shift"][f])} for f in np.argsort(al)[:25]],
            "top_align_pos": [{"f": int(f), "align": float(al[f]), "fire_gemma": float(firing[f]), "d_tok_test": float(d_tok_test[f]), "shift": float(atlas[l]["shift"][f])} for f in np.argsort(-al)[:25]],
            "top_d_neg": [{"f": int(f), "align": float(al[f]), "fire_gemma": float(firing[f]), "d_tok_train": float(d_tok_train[f]), "d_tok_test": float(d_tok_test[f])} for f in np.argsort(d_tok_train)[:25]],
            "top_d_pos": [{"f": int(f), "align": float(al[f]), "fire_gemma": float(firing[f]), "d_tok_train": float(d_tok_train[f]), "d_tok_test": float(d_tok_test[f])} for f in np.argsort(-d_tok_train)[:25]],
        }
        log(f"stage1 L{l}: L0 bella={sb['l0']:.1f} gemma={sg['l0']:.1f} (meta {res['per_layer'][str(l)]['meta_l0']}) EV={sg['ev']:.3f}; "
            f"align check r={align_check['pearson_with_atlas']:.4f}; spearman(d, align) union={rho:.3f} all={rho_all:.3f}; pool={sets['pool_size']}; "
            f"A_5={sets['A'][5]} D_5={sets['D'][5]}")
    res["timing_s"] = time.time() - t0
    dump(out / "stage1.json", res)
    # free the big accumulators, keep the SAEs for stages 2/3
    stats.clear()
    torch.cuda.empty_cache()
    l0_10 = res["per_layer"]["10"]["l0_gemma"]
    res["l0_check_layer10"] = {"l0_gemma": l0_10, "window": [35, 65], "pass": 35 <= l0_10 <= 65}
    dump(out / "stage1.json", res)
    log(f"stage1 done in {time.time()-t0:.0f}s; layer-10 real-token L0 (Gemma side) = {l0_10:.1f} -> {'PASS' if res['l0_check_layer10']['pass'] else 'FAIL'}")
    ctx = {"pairs": pairs, "gem": gem, "prefixes": prefixes, "harv": harv, "keep10": keep10, "saes": saes, "sets": sets_by_layer, "res": res}
    return ctx


# --------------------------------------------------------------------------- stage 4


def stage4(out: Path, args, dirs, ctx):
    """Bella fine-tune minus base mean residual shift over the Gemma reply tokens, per layer."""
    t0 = time.time()
    pairs, gem, prefixes = ctx["pairs"], ctx["gem"], ctx["prefixes"]
    replies = [gem[p["pid"]]["text"] or " " for p in pairs]
    log("stage4: loading the Bella fine-tune")
    Wft = Wrapped(FT_HF, FT_REV, tok_from=(BASE_HF, BASE_REV))
    hft = Wft.harvest(prefixes, replies, batch_size=args.harvest_batch, phase="s4_harvest_ft")
    ft_mean = hft["mean_all"]                       # [N, L, d]
    base_mean = ctx["harv"]["gemma"]["mean_all"]    # same prompts, same reply tokens, base model
    bella_mean = ctx["harv"]["bella"]["mean_all"]
    splits = np.array([p["split"] for p in pairs])
    L = ft_mean.shape[1]
    shift = (ft_mean - base_mean).mean(0)           # [L, d]
    res = {"ft": {"hf": FT_HF, "revision": FT_REV, "arch": type(Wft.model).__name__}, "n_prompts": len(pairs),
           "n_tok_mean": float(hft["n_tokens"].mean()), "per_layer": []}
    rng = np.random.default_rng(SEED)
    for l in range(L):
        v = dirs[l]
        s = shift[l]
        proj = float(s @ v)
        gap_all = float(((bella_mean[:, l] - base_mean[:, l]) @ v).mean())
        gap_test = float(((bella_mean[splits == "test", l] - base_mean[splits == "test", l]) @ v).mean()) if (splits == "test").any() else gap_all
        # per-prompt shift projections for a CI
        pp = (ft_mean[:, l] - base_mean[:, l]) @ v
        se = float(pp.std(ddof=1) / math.sqrt(len(pp))) if len(pp) > 1 else float("nan")
        rand = np.abs([(s / (np.linalg.norm(s) + 1e-8)) @ (r / np.linalg.norm(r)) for r in rng.standard_normal((20, len(s)))])
        res["per_layer"].append({
            "layer": l, "cos": float(proj / (np.linalg.norm(s) + 1e-8)), "shift_norm": float(np.linalg.norm(s)),
            "shift_norm_over_median_norm": float(np.linalg.norm(s) / ctx["res"]["median_norm_gemma"][l]),
            "proj_on_dir": proj, "proj_se": se, "gap_all": gap_all, "gap_test": gap_test,
            "frac_gap_closed_all": proj / gap_all if gap_all else None, "frac_gap_closed_test": proj / gap_test if gap_test else None,
            "cos_ft_minus_base_vs_bella_minus_base": float(s @ (bella_mean[:, l] - base_mean[:, l]).mean(0)
                                                            / (np.linalg.norm(s) * np.linalg.norm((bella_mean[:, l] - base_mean[:, l]).mean(0)) + 1e-8)),
            "random_dir_abs_cos_p95": float(np.percentile(rand, 95)),
            "is_global_attention": l in Wft.global_layers,
        })
    res["timing_s"] = time.time() - t0
    dump(out / "stage4.json", res)
    np.savez_compressed(out / "sft_shift.npz", shift=shift, ft_mean=ft_mean.mean(0), base_mean=base_mean.mean(0))
    log("stage4 cos per layer:", " ".join(f"{r['cos']:.2f}" for r in res["per_layer"]))
    log(f"stage4 done in {time.time()-t0:.0f}s; L4 cos={res['per_layer'][4]['cos']:.3f} L10 cos={res['per_layer'][10]['cos']:.3f} "
        f"frac closed L10 (test gap)={res['per_layer'][10]['frac_gap_closed_test']}")
    del Wft
    torch.cuda.empty_cache()
    return res


# --------------------------------------------------------------------------- stage 2


def parse_cell(spec, ctx, dirs, med_norm, clamp_scale):
    """Returns (row_meta, hook_layer, hook_obj)."""
    parts = spec.split(":")
    kind = parts[0]
    if kind == "base":
        return {"cell": spec, "kind": "base", "layer": -1, "coef": 0.0, "k": 0}, None, None
    layer = int(parts[1])
    if kind == "dir":
        c = float(parts[2])
        a_hat = -torch.tensor(dirs[layer])  # Assistant direction = Gemma - Bella; h += c * median_norm * a_hat
        return {"cell": spec, "kind": "dir", "layer": layer, "coef": c, "k": 0}, layer, DirectionAdd(a_hat * float(c * med_norm[layer]))
    sets = ctx["sets"][layer]
    sae = ctx["saes"][layer]
    if kind in ("A", "D"):
        k = int(parts[2])
        feats = sets[kind][k]
        return {"cell": spec, "kind": kind, "layer": layer, "coef": 0.0, "k": k, "clamp_scale": clamp_scale}, layer, FeatureEdit(sae, feats, scale=clamp_scale)
    if kind == "R":
        k, draw = int(parts[2]), int(parts[3])
        feats = sets["R"][f"{k}:{draw}"]
        return {"cell": spec, "kind": "R", "layer": layer, "coef": 0.0, "k": k, "draw": draw, "clamp_scale": clamp_scale}, layer, FeatureEdit(sae, feats, scale=clamp_scale)
    if kind == "amp":  # amplify the k most Bella-ward (positive align, firing) features by a factor
        k, factor = int(parts[2]), float(parts[3])
        al = np.load(Path(ctx["out"]) / f"features_L{layer}.npz")
        pool = np.where(al["fire_bella_train"] >= FIRE_MIN)[0]
        feats = pool[np.argsort(-al["align"][pool])][:k].tolist()
        return {"cell": spec, "kind": "amp", "layer": layer, "coef": factor, "k": k}, layer, FeatureEdit(sae, feats, amp=factor)
    raise ValueError(spec)


def stage2(W: Wrapped, out: Path, args, dirs, ctx, cells):
    pairs, gem = ctx["pairs"], ctx["gem"]
    test_pairs = [p for p in pairs if p["split"] == "test"] or pairs[-6:]
    evalp = read_jsonl(DATA / "eval_prompts.jsonl")
    crisis = read_jsonl(DATA / "crisis_eval.jsonl")
    red = read_jsonl(DATA / "red_team.jsonl")
    neutral = read_jsonl(DATA / "neutral.jsonl")
    if args.smoke:
        evalp, crisis, red, neutral, test_pairs = evalp[:6], crisis[:3], red[:6], neutral[:6], test_pairs[:6]
    chat_items = ([("eval", e["pid"], e["prompt"]) for e in evalp] + [("crisis", c["pid"], c["prompt"]) for c in crisis]
                  + [("redteam", r["pid"], r["prompt"]) for r in red])
    med_norm = np.array(ctx["res"]["median_norm_gemma"])
    prior_med = json.load(open(PRIOR / "stage1.json"))["median_norm_gemma"] if (PRIOR / "stage1.json").exists() else None
    if prior_med is not None:  # use the axis experiment's scale so the direction cells match its recipe exactly
        med_norm = np.array(prior_med)
    log(f"stage2: {len(cells)} cells, chat prompts={len(chat_items)} neutral={len(neutral)} gen_tokens={args.gen_tokens} "
        f"median_norm L10={med_norm[10]:.2f} L4={med_norm[4]:.2f} clamp_scale={args.clamp_scale}")
    gens_path = out / "generations.jsonl"
    rows = read_jsonl(gens_path) if gens_path.exists() and not args.smoke else []
    done = {r["cell"] for r in rows}
    dose_path = out / "dose_response.json"
    dose = json.load(open(dose_path)) if dose_path.exists() and not args.smoke else []
    dose = [d for d in dose if d["cell"] in done]
    cell_meta = json.load(open(out / "cells.json")) if (out / "cells.json").exists() and not args.smoke else {}
    test_prefix = [W.chat_prefix(p["prompt"]) for p in test_pairs]
    test_bella = [p["bella"] for p in test_pairs]
    test_gemma = [gem[p["pid"]]["text"] or " " for p in test_pairs]
    t0 = time.time()
    for ci, spec in enumerate(cells, 1):
        if spec in done:
            continue
        meta, layer, hook = parse_cell(spec, ctx, dirs, med_norm, args.clamp_scale)
        W.clear_hooks()
        if hook is not None:
            W.add_hook(layer, hook)
        tc = time.time()
        g = W.generate([x[2] for x in chat_items], max_new_tokens=args.gen_tokens, batch_size=args.gen_batch, phase=f"s2_cell{ci}/{len(cells)}")
        chat_stats = hook.stats() if isinstance(hook, FeatureEdit) else None
        if isinstance(hook, FeatureEdit):
            hook.reset()
        for (s, pid, pr), gg in zip(chat_items, g):
            rows.append({**meta, "set": s, "pid": pid, "prompt": pr, **gg})
        gn = W.generate([x["text"] for x in neutral], max_new_tokens=64, batch_size=args.gen_batch, chat=False, phase=f"s2_cell{ci}_neutral")
        neutral_stats = hook.stats() if isinstance(hook, FeatureEdit) else None
        if isinstance(hook, FeatureEdit):
            hook.reset()
        for nn, gg in zip(neutral, gn):
            rows.append({**meta, "set": "neutral", "pid": nn["pid"], "prompt": nn["text"], **gg})
        lpb, _ = W.mean_logprob(test_prefix, test_bella, batch_size=args.harvest_batch)
        lpg, _ = W.mean_logprob(test_prefix, test_gemma, batch_size=args.harvest_batch)
        dose.append({**meta, "mean_lp_bella": float(lpb.mean()), "mean_lp_gemma": float(lpg.mean()),
                     "contrast": float((lpb - lpg).mean()), "contrast_se": float((lpb - lpg).std(ddof=1) / math.sqrt(len(lpb))),
                     "lp_bella": lpb.tolist(), "lp_gemma": lpg.tolist()})
        cell_meta[spec] = {**meta, "features": (hook.feats.tolist() if isinstance(hook, FeatureEdit) else None),
                           "clamp_stats_chat": chat_stats, "clamp_stats_neutral": neutral_stats, "time_s": time.time() - tc,
                           "median_norm_used": float(med_norm[layer]) if layer is not None else None}
        write_jsonl(gens_path, rows)
        dump(dose_path, dose)
        dump(out / "cells.json", cell_meta)
        done.add(spec)
        extra = f" clamped/tok={chat_stats['mean_active_per_generated_token']:.2f}" if chat_stats else ""
        log(f"cell {ci}/{len(cells)} {spec} done in {time.time()-tc:.0f}s; contrast={dose[-1]['contrast']:.3f}{extra}; sample: {g[0]['text'][:140]!r}")
    W.clear_hooks()
    dump(out / "stage2_meta.json", {"cells": cells, "n_chat_prompts": len(chat_items), "n_neutral": len(neutral), "gen_tokens": args.gen_tokens,
                                    "neutral_tokens": 64, "decoding": "greedy", "clamp_scale": args.clamp_scale,
                                    "median_norm": {"4": float(med_norm[4]), "10": float(med_norm[10])}, "timing_s": time.time() - t0})
    log(f"stage2 done in {time.time()-t0:.0f}s")


# --------------------------------------------------------------------------- stage 3


def stage3(W: Wrapped, out: Path, args, dirs, ctx):
    """Exemplars (top-20 activating token windows over the 600 pairs, both sides, plus this run's unsteered
    eval replies) and logit lens for the top-25 Assistant-ward / top-25 Bella-ward layer-10 features by align,
    plus A_20 members and feature 78793."""
    t0 = time.time()
    l = 10
    sae = ctx["saes"][l]
    f10 = np.load(out / f"features_L{l}.npz")
    al = f10["align"]
    want = list(np.argsort(al)[:25]) + list(np.argsort(-al)[:25]) + list(ctx["sets"][l]["A"][20]) + [NAMED_FEATURE]
    want = list(dict.fromkeys(int(f) for f in want))
    if args.smoke:
        want = want[:8]
    # corpus: pairs (both sides) + unsteered eval replies from this run
    pairs, gem = ctx["pairs"], ctx["gem"]
    corpus = []  # (source, prompt, reply, ids, acts bf16 [T, d])
    ids_b, ids_g = ctx["harv"]["bella"]["reply_ids"], ctx["harv"]["gemma"]["reply_ids"]
    for i, p in enumerate(pairs):
        corpus.append(("bella", p["prompt"], p["bella"], ids_b[i], ctx["keep10"]["bella"][i]))
        corpus.append(("gemma", p["prompt"], gem[p["pid"]]["text"], ids_g[i], ctx["keep10"]["gemma"][i]))
    gens_path = out / "generations.jsonl"
    if gens_path.exists():
        base_eval = [r for r in read_jsonl(gens_path) if r["cell"] == "base" and r["set"] == "eval" and r["text"]]
        if base_eval:
            keep = {}
            def on_tokens(ll, h, rep_idx, idx):
                ri = rep_idx.cpu().numpy()
                hc = h.to(torch.bfloat16).cpu()
                for i in idx:
                    keep[i] = hc[torch.tensor(ri == i)]
            hv = W.harvest([W.chat_prefix(r["prompt"]) for r in base_eval], [r["text"] for r in base_eval], batch_size=args.harvest_batch,
                           phase="s3_harvest_eval", token_layers=(l,), on_tokens=on_tokens, keep_ids=True)
            for i, r in enumerate(base_eval):
                corpus.append(("gemma_eval", r["prompt"], r["text"], hv["reply_ids"][i], keep[i]))
    fidx = torch.tensor(want).cuda()
    We, be, thr = sae["W_enc"][fidx], sae["b_enc"][fidx], sae["thr"][fidx]
    tops = {f: [] for f in want}  # (act, corpus idx, tok idx)
    fire = {f: {"bella": [0, 0], "gemma": [0, 0]} for f in want}
    with torch.inference_mode():
        for ci, (src, pr, rep, ids, acts) in enumerate(corpus):
            if acts is None or acts.shape[0] == 0:
                continue
            x = acts.cuda().float()
            pre = F.linear(x - sae["b_dec"], We, be)
            z = (pre * (pre > thr)).cpu().numpy()  # [T, nf]
            side = "bella" if src == "bella" else "gemma"
            for j, f in enumerate(want):
                col = z[:, j]
                fire[f][side][0] += int((col > 0).sum())
                fire[f][side][1] += len(col)
                nz = np.where(col > 0)[0]
                if len(nz):
                    for ti in nz[np.argsort(-col[nz])][:3]:
                        tops[f].append((float(col[ti]), ci, int(ti)))
    res = {"layer": l, "features": {}, "n_corpus": len(corpus), "sources": {s: sum(1 for c in corpus if c[0] == s) for s in ("bella", "gemma", "gemma_eval")}}
    a_sets = {k: set(v) for k, v in ctx["sets"][l]["A"].items()}
    with torch.inference_mode():
        for f in want:
            ex = sorted(tops[f], reverse=True)[:20]
            exemplars = []
            for act, ci, ti in ex:
                src, pr, rep, ids, _ = corpus[ci]
                lo, hi = max(0, ti - 10), min(len(ids), ti + 6)
                window = W.tok.decode(ids[lo:ti]) + "[[" + W.tok.decode(ids[ti:ti + 1]) + "]]" + W.tok.decode(ids[ti + 1:hi])
                exemplars.append({"act": round(act, 3), "source": src, "token": W.tok.decode(ids[ti:ti + 1]), "window": window,
                                  "prompt": pr[:120]})
            v = sae["W_dec"][:, f]
            v = v / v.norm() * float(ctx["res"]["median_norm_gemma"][l])
            h = W.final_norm(v[None, None, :].to(torch.bfloat16))
            logits = W.lm_head(h)[0, 0].float()
            res["features"][str(f)] = {
                "align": float(al[f]), "shift": float(f10["shift"][f]), "dec_norm": float(f10["dec_norm"][f]),
                "d_tok_test": float(f10["d_tok_test"][f]), "d_reply_test": float(f10["d_reply_test"][f]),
                "fire_bella_pairs": fire[f]["bella"][0] / max(fire[f]["bella"][1], 1), "fire_gemma_pairs": fire[f]["gemma"][0] / max(fire[f]["gemma"][1], 1),
                "fire_gemma_train_stage1": float(f10["fire_gemma_train"][f]), "fire_bella_train_stage1": float(f10["fire_bella_train"][f]),
                "in_A": [k for k, s in a_sets.items() if f in s], "group": ("assistant_ward" if al[f] < 0 else "bella_ward"),
                "rank_neg_align": int(np.where(np.argsort(al) == f)[0][0]) + 1, "rank_pos_align": int(np.where(np.argsort(-al) == f)[0][0]) + 1,
                "logit_lens_promoted": [W.tok.decode([t]) for t in torch.topk(logits, 15).indices.tolist()],
                "logit_lens_suppressed": [W.tok.decode([t]) for t in torch.topk(-logits, 15).indices.tolist()],
                "n_firing_positions_found": len(tops[f]), "exemplars": exemplars,
            }
    res["timing_s"] = time.time() - t0
    dump(out / "stage3_exemplars.json", res)
    log(f"stage3 done in {time.time()-t0:.0f}s; {len(want)} features, corpus {res['sources']}")


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="1,4,2,3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_n", type=int, default=24)
    ap.add_argument("--gen_batch", type=int, default=48)
    ap.add_argument("--harvest_batch", type=int, default=16)
    ap.add_argument("--gen_tokens", type=int, default=128)
    ap.add_argument("--cells", default=None, help="comma list of cell specs overriding the default grid")
    ap.add_argument("--clamp_scale", type=float, default=0.0, help="0 = clamp selected features to zero; 0.5 = halve them")
    ap.add_argument("--force", action="store_true", help="continue past a failed layer-10 L0 check")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    root = Path(args.out or os.environ.get("SILICO_EXPERIMENT_ARTIFACTS_DIR", HERE / "results" / "artifacts"))
    out = root / ("smoke" if args.smoke else "run")
    out.mkdir(parents=True, exist_ok=True)
    log(f"config: {vars(args)} out={out}")
    assert torch.cuda.device_count() == 1, torch.cuda.device_count()
    stages = [int(s) for s in args.stages.split(",")]
    cells = args.cells.split(",") if args.cells else DEFAULT_CELLS
    if args.smoke:
        cells = ["base", "dir:10:-0.35", "A:10:20", "R:10:50:0", "A:4:20"]
    report_progress(step=0, total_steps=1, phase="loading")
    dirs, atlas = fetch_inputs(out)
    W = Wrapped(BASE_HF, BASE_REV)
    log("chat prefix sample:", repr(W.chat_prefix("hi there")))
    ctx = stage1(W, out, args, dirs, atlas)
    ctx["out"] = out
    if not ctx["res"]["l0_check_layer10"]["pass"] and not args.force and not args.smoke:
        log("STOP: layer-10 real-token L0 outside [35, 65]; not generating (rerun with --force to override)")
        return
    for s in stages:
        if s == 1:
            continue
        if s == 4:
            stage4(out, args, dirs, ctx)
        elif s == 2:
            stage2(W, out, args, dirs, ctx, cells)
        elif s == 3:
            stage3(W, out, args, dirs, ctx)
    log("ALL DONE ->", out)


if __name__ == "__main__":
    main()
