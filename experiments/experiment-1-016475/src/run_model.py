"""Per-model GPU pipeline: Bella-minus-Gemma direction, layer profile, steering grid, SAE decomposition.

Stage 1  generate Gemma replies for the 600 matched prompts, harvest residual-stream activations
         (output of every decoder block, mean-pooled over assistant-reply tokens), build the
         difference-of-means direction per layer on the train split, score val/test, nulls,
         adjacent-layer cosine, authentic/corporate cross-check, logit lens, mean projections.
Stage 2  steering grid: 4 layers x coefficients on 390 prompts (+ neutral stems), 128 new tokens,
         greedy; contrastive dose response on the test pairs; perplexity of neutral continuations
         under the unsteered model.
Stage 3  (E2B) project the direction onto the JumpReLU SAE atlas for the 4 stage-2 layers.

Layer index convention: "layer l" = output of decoder block l (0-indexed), i.e. the residual stream
entering block l+1. Global-attention layers use the same 0-indexed convention as config.layer_types.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
MODELS = {
    "E2B": {"hf": "google/gemma-4-E2B-it", "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7", "holdout_layer": 9,
            "predicted_crests": [4, 13, 14]},
    "E4B": {"hf": "google/gemma-4-E4B-it", "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2", "holdout_layer": 11,
            "predicted_crests": []},
}
SAE_REPO = "juiceb0xc0de/gemma-4-e2b-it-SAE"
COEFS = [-0.25, -0.5, -1.0, -2.0, -4.0, 1.0]
SEED = 42
FIXED_USER_PROMPT_FOR_SNIPPETS = "hey, what's up?"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with Path(p).open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1))


# --------------------------------------------------------------------------- model


class Wrapped:
    def __init__(self, hf_id, revision):
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        self.tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        if self.tok.chat_template is None:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(hf_id, revision=revision)
            self.tok.chat_template = proc.chat_template
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
        # resolve the decoder-layer container semantically
        cands = [(n, m) for n, m in self.model.named_modules()
                 if isinstance(m, torch.nn.ModuleList) and len(m) == self.n_layers
                 and all(hasattr(x, "self_attn") for x in m)]
        assert cands, "decoder layer container not found"
        self.layers_name, self.layers = cands[0]
        parent_name = self.layers_name.rsplit(".", 1)[0]
        parent = self.model.get_submodule(parent_name)
        self.final_norm = getattr(parent, "norm")
        self.lm_head = self.model.get_output_embeddings()
        log(f"model {type(self.model).__name__} layers at {self.layers_name} n={self.n_layers} d={self.d} "
            f"global={self.global_layers} dtype={self.model.dtype} head={tuple(self.lm_head.weight.shape)}")
        self.gen_cfg = dict(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                            pad_token_id=self.tok.pad_token_id)
        # steering state
        self._steer = None  # (layer, vector[d] tensor on cuda)
        self._steer_handles = []
        # end-of-turn token for stripping
        self.eot_ids = set(self.tok.convert_tokens_to_ids(t) for t in ["<end_of_turn>", "<turn|>", "<|turn>"]
                           if t in self.tok.get_vocab())
        self.eot_ids.add(self.tok.eos_token_id)

    # ---- prompts
    def chat_prefix(self, prompt: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True)

    def encode_batch(self, texts, side="left"):
        self.tok.padding_side = side
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        return {k: v.cuda() for k, v in enc.items()}

    # ---- generation
    @torch.inference_mode()
    def generate(self, texts, max_new_tokens, batch_size=32, phase="generate", chat=True):
        """texts: raw user prompts (chat=True) or raw stems (chat=False). Returns list of dicts."""
        prefixes = [self.chat_prefix(t) if chat else (self.tok.bos_token or "") + t for t in texts]
        order = sorted(range(len(prefixes)), key=lambda i: -len(prefixes[i]))
        out = [None] * len(texts)
        for bi in range(0, len(order), batch_size):
            idx = order[bi:bi + batch_size]
            enc = self.encode_batch([prefixes[i] for i in idx], side="left")
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

    # ---- harvesting
    @torch.inference_mode()
    def harvest(self, prefixes, replies, batch_size=16, first_k=32, phase="harvest", token_level_layers=None):
        """Mean-pool block outputs over reply tokens. Returns dict of arrays:
        mean_all [N, L, d] fp32, mean_first [N, L, d], norms: per-layer list of per-token norms (sample),
        n_tokens [N]. If token_level_layers is set, also returns token-level acts for those layers."""
        N, L, d = len(prefixes), self.n_layers, self.d
        mean_all = np.zeros((N, L, d), np.float32)
        mean_first = np.zeros((N, L, d), np.float32)
        n_tok = np.zeros(N, np.int32)
        norm_samples = [[] for _ in range(L)]
        tok_level = {l: [] for l in (token_level_layers or [])}
        store = {}

        def mk_hook(l):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                store[l] = h
            return hook

        handles = [self.layers[l].register_forward_hook(mk_hook(l)) for l in range(L)]
        try:
            full = [p + r for p, r in zip(prefixes, replies)]
            order = sorted(range(N), key=lambda i: -len(full[i]))
            for bi in range(0, N, batch_size):
                idx = order[bi:bi + batch_size]
                self.tok.padding_side = "right"
                enc = self.tok([full[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
                plen = [len(self.tok(prefixes[i], add_special_tokens=False)["input_ids"]) for i in idx]
                am = enc["attention_mask"]
                T = am.shape[1]
                pos = torch.arange(T)[None, :]
                reply_mask = (pos >= torch.tensor(plen)[:, None]) & (am == 1)
                first_mask = reply_mask & (pos < (torch.tensor(plen)[:, None] + first_k))
                self.model(**{k: v.cuda() for k, v in enc.items()})
                rm = reply_mask.cuda().unsqueeze(-1).float()
                fm = first_mask.cuda().unsqueeze(-1).float()
                for l in range(L):
                    h = store[l].float()
                    ma = (h * rm).sum(1) / rm.sum(1).clamp(min=1)
                    mf = (h * fm).sum(1) / fm.sum(1).clamp(min=1)
                    for j, i in enumerate(idx):
                        mean_all[i, l] = ma[j].cpu().numpy()
                        mean_first[i, l] = mf[j].cpu().numpy()
                    if len(norm_samples[l]) < 20000:
                        nr = h.norm(dim=-1)[reply_mask.cuda()]
                        norm_samples[l].extend(nr[:2000].cpu().tolist())
                    if l in tok_level:
                        for j, i in enumerate(idx):
                            m = reply_mask[j]
                            tok_level[l].append((i, h[j][m.cuda()].cpu().to(torch.bfloat16)))
                for j, i in enumerate(idx):
                    n_tok[i] = int(reply_mask[j].sum())
                store.clear()
                report_progress(step=min(bi + batch_size, N), total_steps=N, phase=phase)
        finally:
            for h in handles:
                h.remove()
        res = {"mean_all": mean_all, "mean_first": mean_first, "n_tokens": n_tok,
               "median_norm": np.array([float(np.median(s)) if s else float("nan") for s in norm_samples])}
        if token_level_layers:
            res["token_level"] = tok_level
        return res

    # ---- steering
    def set_steering(self, layer: int | None, vec: torch.Tensor | None):
        for h in self._steer_handles:
            h.remove()
        self._steer_handles = []
        self._steer = None
        if layer is None:
            return
        v = vec.to(torch.bfloat16).cuda()

        def hook(mod, inp, out):
            if isinstance(out, tuple):
                return (out[0] + v,) + tuple(out[1:])
            return out + v

        self._steer_handles.append(self.layers[layer].register_forward_hook(hook))
        self._steer = (layer, v)

    # ---- scoring
    @torch.inference_mode()
    def mean_logprob(self, prefixes, continuations, batch_size=16):
        """Mean per-token log-prob of continuation given prefix (under whatever steering is set)."""
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
            logits = self.model(input_ids=ids, attention_mask=am).logits.float()
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


# --------------------------------------------------------------------------- stats


def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def cohens_d(a, b):
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp = math.sqrt((sa ** 2 + sb ** 2) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def dom_profile(bella, gemma, train, evalidx):
    """bella/gemma: [N, L, d]. Returns unit directions [L, d] (Bella-minus-Gemma) and per-layer eval metrics."""
    L = bella.shape[1]
    dirs = np.zeros((L, bella.shape[2]), np.float32)
    au, d = np.zeros(L), np.zeros(L)
    for l in range(L):
        v = bella[train, l].mean(0) - gemma[train, l].mean(0)
        v = v / (np.linalg.norm(v) + 1e-8)
        dirs[l] = v
        pb, pg = bella[evalidx, l] @ v, gemma[evalidx, l] @ v
        au[l], d[l] = auroc(pb, pg), cohens_d(pb, pg)
    return dirs, au, d


def local_maxima(x, above=None):
    out = []
    for i in range(len(x)):
        lo = x[i - 1] if i > 0 else -np.inf
        hi = x[i + 1] if i < len(x) - 1 else -np.inf
        if x[i] >= lo and x[i] > hi and (above is None or x[i] > above[i]):
            out.append(i)
    return out


# --------------------------------------------------------------------------- stages


def stage1(W: Wrapped, out: Path, args):
    pairs = read_jsonl(DATA / "pairs.jsonl")
    if args.smoke:
        pairs = pairs[: args.smoke_n]
    prompts = [p["prompt"] for p in pairs]
    t0 = time.time()
    log(f"stage1: generating {len(prompts)} Gemma replies (greedy, 96 new tokens)")
    gen_path = out / "gemma_replies.jsonl"
    if gen_path.exists() and not args.smoke:
        gemma = read_jsonl(gen_path)
        assert len(gemma) == len(pairs)
    else:
        g = W.generate(prompts, max_new_tokens=96, batch_size=args.gen_batch, phase="s1_generate")
        gemma = [{"pid": p["pid"], "split": p["split"], **gg} for p, gg in zip(pairs, g)]
        write_jsonl(gen_path, gemma)
    log(f"generation done in {time.time()-t0:.0f}s; sample: {gemma[0]['text'][:200]!r}")
    empty = sum(1 for g in gemma if not g["text"])
    log(f"empty Gemma replies: {empty}; truncated: {sum(g['truncated'] for g in gemma)}")

    prefixes = [W.chat_prefix(p) for p in prompts]
    log("stage1: harvesting Bella side")
    hb = W.harvest(prefixes, [p["bella"] for p in pairs], batch_size=args.harvest_batch, phase="s1_harvest_bella")
    log("stage1: harvesting Gemma side")
    hg = W.harvest(prefixes, [g["text"] if g["text"] else " " for g in gemma], batch_size=args.harvest_batch,
                   phase="s1_harvest_gemma")
    np.savez_compressed(out / "acts_pairs.npz", bella_all=hb["mean_all"], bella_first=hb["mean_first"],
                        gemma_all=hg["mean_all"], gemma_first=hg["mean_first"],
                        n_tok_bella=hb["n_tokens"], n_tok_gemma=hg["n_tokens"],
                        median_norm_gemma=hg["median_norm"], median_norm_bella=hb["median_norm"])
    splits = np.array([p["split"] for p in pairs])
    train, val, test = (np.where(splits == s)[0] for s in ("train", "val", "test"))
    if args.smoke:
        n = len(pairs)
        train, val, test = np.arange(0, n * 2 // 3), np.arange(n * 2 // 3, n * 5 // 6), np.arange(n * 5 // 6, n)
    L = W.n_layers
    rng = np.random.default_rng(SEED)
    res = {"model": args.model, "hf": MODELS[args.model]["hf"], "n_layers": L, "d": W.d,
           "global_layers": W.global_layers, "layer_types": W.layer_types, "splits": {"train": len(train), "val": len(val), "test": len(test)},
           "median_norm_gemma": hg["median_norm"].tolist(), "median_norm_bella": hb["median_norm"].tolist(),
           "n_tok_bella_mean": float(hb["n_tokens"].mean()), "n_tok_gemma_mean": float(hg["n_tokens"].mean()),
           "pooling": {}}
    dirs_by_pool = {}
    for pool in ("all", "first"):
        B, G = hb[f"mean_{pool}"], hg[f"mean_{pool}"]
        dirs, au_t, d_t = dom_profile(B, G, train, test)
        _, au_v, d_v = dom_profile(B, G, train, val)
        dirs_by_pool[pool] = dirs
        # shuffled-label null: flip Bella/Gemma within random pairs (train), evaluate on test
        n_sh = 5 if args.smoke else 20
        null_d, null_au = np.zeros((n_sh, L)), np.zeros((n_sh, L))
        for s in range(n_sh):
            flip = rng.random(len(train)) < 0.5
            Bt, Gt = B[train].copy(), G[train].copy()
            Bt[flip], Gt[flip] = G[train][flip], B[train][flip]
            for l in range(L):
                v = Bt[:, l].mean(0) - Gt[:, l].mean(0)
                v /= (np.linalg.norm(v) + 1e-8)
                pb, pg = B[test, l] @ v, G[test, l] @ v
                null_d[s, l], null_au[s, l] = abs(cohens_d(pb, pg)), max(auroc(pb, pg), 1 - auroc(pb, pg))
        # random unit direction null
        rand_d = np.zeros((n_sh, L))
        for s in range(n_sh):
            for l in range(L):
                v = rng.standard_normal(W.d).astype(np.float32)
                v /= np.linalg.norm(v)
                rand_d[s, l] = abs(cohens_d(B[test, l] @ v, G[test, l] @ v))
        # adjacent-layer cosine
        adj = [float(dirs[l] @ dirs[l + 1]) for l in range(L - 1)]
        # mean projection of each side onto the direction, in units of the layer's median residual norm
        proj_g = [float((G[test, l] @ dirs[l]).mean() / hg["median_norm"][l]) for l in range(L)]
        proj_b = [float((B[test, l] @ dirs[l]).mean() / hb["median_norm"][l]) for l in range(L)]
        # raw DOM norm relative to residual norm (how big is the shift)
        raw = [float(np.linalg.norm(B[train, l].mean(0) - G[train, l].mean(0)) / hg["median_norm"][l]) for l in range(L)]
        res["pooling"][pool] = {
            "test_auroc": au_t.tolist(), "test_d": d_t.tolist(), "val_auroc": au_v.tolist(), "val_d": d_v.tolist(),
            "null_shuffled_d_p95": np.percentile(null_d, 95, axis=0).tolist(),
            "null_shuffled_d_mean": null_d.mean(0).tolist(),
            "null_shuffled_auroc_p95": np.percentile(null_au, 95, axis=0).tolist(),
            "null_random_d_p95": np.percentile(rand_d, 95, axis=0).tolist(),
            "adjacent_cosine": adj, "mean_proj_gemma_test": proj_g, "mean_proj_bella_test": proj_b,
            "dom_norm_over_median_norm": raw,
        }
    np.savez_compressed(out / "directions.npz", all=dirs_by_pool["all"], first=dirs_by_pool["first"])
    # pooling choice on validation split (mean d over layers), reported both
    val_all = float(np.mean(res["pooling"]["all"]["val_d"]))
    val_first = float(np.mean(res["pooling"]["first"]["val_d"]))
    pool = "all" if val_all >= val_first else "first"
    res["pooling_choice"] = {"chosen": pool, "val_mean_d_all": val_all, "val_mean_d_first": val_first}
    dirs = dirs_by_pool[pool]
    log(f"pooling chosen: {pool} (val mean d all={val_all:.3f} first={val_first:.3f})")

    # authentic vs corporate cross-check (snippets wrapped as assistant replies to a fixed user turn)
    au_rows = read_jsonl(DATA / "authentic.jsonl")
    co_rows = read_jsonl(DATA / "corporate.jsonl")
    if args.smoke:
        au_rows, co_rows = au_rows[:24], co_rows[:24]
    pre = W.chat_prefix(FIXED_USER_PROMPT_FOR_SNIPPETS)
    ha = W.harvest([pre] * len(au_rows), [r["text"] for r in au_rows], batch_size=args.harvest_batch, phase="s1_authentic")
    hc = W.harvest([pre] * len(co_rows), [r["text"] for r in co_rows], batch_size=args.harvest_batch, phase="s1_corporate")
    ac_dir = ha["mean_all"].mean(0) - hc["mean_all"].mean(0)
    ac_dir /= np.linalg.norm(ac_dir, axis=-1, keepdims=True) + 1e-8
    res["authentic_corporate"] = {
        "cosine_with_pair_direction": [float(ac_dir[l] @ dirs[l]) for l in range(L)],
        "cosine_with_pair_direction_all_pool": [float(ac_dir[l] @ dirs_by_pool["all"][l]) for l in range(L)],
        "n_authentic": len(au_rows), "n_corporate": len(co_rows), "fixed_user_prompt": FIXED_USER_PROMPT_FOR_SNIPPETS,
        "pair_direction_auroc_on_snippets": [auroc(ha["mean_all"][:, l] @ dirs[l], hc["mean_all"][:, l] @ dirs[l]) for l in range(L)],
    }

    # logit lens: direction through final norm and unembedding
    lens = {}
    with torch.inference_mode():
        for l in range(L):
            v = torch.tensor(dirs[l]).cuda().to(torch.bfloat16) * float(hg["median_norm"][l])
            h = W.final_norm(v[None, None, :])
            logits = W.lm_head(h)[0, 0].float()
            top = torch.topk(logits, 20).indices.tolist()
            bot = torch.topk(-logits, 20).indices.tolist()
            lens[str(l)] = {"promoted": [W.tok.decode([t]) for t in top], "suppressed": [W.tok.decode([t]) for t in bot]}
    res["logit_lens"] = lens

    # layer selection for stage 2, on the validation profile of the chosen pooling
    P = res["pooling"][pool]
    val_d, null95 = np.array(P["val_d"]), np.array(P["null_shuffled_d_p95"])
    maxima = local_maxima(val_d, above=null95)
    maxima_sorted = sorted(maxima, key=lambda l: -val_d[l])
    chosen = []
    for l in maxima_sorted[:2]:
        chosen.append(int(l))
    gl = [l for l in W.global_layers if l < L]
    strongest_global = int(max(gl, key=lambda l: val_d[l]))
    if strongest_global not in chosen:
        chosen.append(strongest_global)
    hold = MODELS[args.model]["holdout_layer"]
    if hold not in chosen:
        chosen.append(hold)
    for l in maxima_sorted[2:]:
        if len(chosen) >= 4:
            break
        if l not in chosen:
            chosen.append(int(l))
    for l in np.argsort(-val_d):
        if len(chosen) >= 4:
            break
        if int(l) not in chosen:
            chosen.append(int(l))
    res["stage2_layers"] = {"layers": chosen[:4], "val_local_maxima_above_null": [int(x) for x in maxima],
                            "strongest_global": strongest_global, "holdout": hold,
                            "test_local_maxima_above_null": [int(x) for x in local_maxima(np.array(P["test_d"]), above=null95)]}
    # pre-registered readout on the TEST profile (chosen pooling)
    test_d = np.array(P["test_d"])
    res["test_profile_summary"] = {
        "argmax_layer": int(np.argmax(test_d)), "max_d": float(test_d.max()),
        "min_d": float(test_d.min()), "layers_above_null": int((test_d > null95).sum()),
        "local_maxima_above_null": res["stage2_layers"]["test_local_maxima_above_null"],
    }
    res["timing_s"] = {"stage1_total": time.time() - t0}
    dump(out / "stage1.json", res)
    log("stage1 done:", json.dumps({k: res[k] for k in ("pooling_choice", "stage2_layers", "test_profile_summary")}))
    return res, dirs, hg["median_norm"]


def stage2(W: Wrapped, out: Path, args, s1, dirs, med_norm):
    pairs = read_jsonl(DATA / "pairs.jsonl")
    test_pairs = [p for p in pairs if p["split"] == "test"]
    gemma_replies = {g["pid"]: g for g in read_jsonl(out / "gemma_replies.jsonl")}
    evalp = read_jsonl(DATA / "eval_prompts.jsonl")
    crisis = read_jsonl(DATA / "crisis_eval.jsonl")
    red = read_jsonl(DATA / "red_team.jsonl")
    neutral = read_jsonl(DATA / "neutral.jsonl")
    layers = s1["stage2_layers"]["layers"]
    coefs = [float(c) for c in args.coefs.split(",")] if args.coefs else COEFS
    if args.smoke:
        evalp, crisis, red, neutral, test_pairs = evalp[:6], crisis[:3], red[:6], neutral[:6], test_pairs[:6]
        layers, coefs = layers[:2], [-1.0, 1.0]
    chat_items = ([("eval", e["pid"], e["prompt"]) for e in evalp] + [("crisis", c["pid"], c["prompt"]) for c in crisis]
                  + [("redteam", r["pid"], r["prompt"]) for r in red])
    n_cells = 1 + len(layers) * len(coefs)
    log(f"stage2: layers={layers} coefs={coefs} chat prompts={len(chat_items)} neutral={len(neutral)} cells={n_cells}")
    gens_path = out / "steering_generations.jsonl"
    done = set()
    rows = []
    if gens_path.exists() and not args.smoke:
        rows = read_jsonl(gens_path)
        done = {(r["layer"], r["coef"]) for r in rows}
    dose = []
    if (out / "dose_response.json").exists() and not args.smoke:
        dose = [x for x in json.load(open(out / "dose_response.json")) if (x["layer"], x["coef"]) in done]
    t0 = time.time()
    cell_i = 0
    test_prefix = [W.chat_prefix(p["prompt"]) for p in test_pairs]
    test_bella = [p["bella"] for p in test_pairs]
    test_gemma = [gemma_replies[p["pid"]]["text"] or " " for p in test_pairs] if not args.smoke else [" x"] * len(test_pairs)
    cells = [(None, 0.0)] + [(l, c) for l in layers for c in coefs]
    for (l, c) in cells:
        cell_i += 1
        key = (l if l is not None else -1, c)
        if key in done:
            continue
        if l is None:
            W.set_steering(None, None)
        else:
            # Assistant direction a = Gemma - Bella = -v; h += c * median_norm_l * a_hat
            a_hat = -torch.tensor(dirs[l])
            W.set_steering(l, a_hat * float(c * med_norm[l]))
        tc = time.time()
        g = W.generate([x[2] for x in chat_items], max_new_tokens=args.gen_tokens, batch_size=args.gen_batch,
                       phase=f"s2_cell{cell_i}/{n_cells}")
        for (s, pid, pr), gg in zip(chat_items, g):
            rows.append({"layer": key[0], "coef": c, "set": s, "pid": pid, "prompt": pr, **gg})
        gn = W.generate([x["text"] for x in neutral], max_new_tokens=64, batch_size=args.gen_batch, chat=False,
                        phase=f"s2_cell{cell_i}_neutral")
        for nn, gg in zip(neutral, gn):
            rows.append({"layer": key[0], "coef": c, "set": "neutral", "pid": nn["pid"], "prompt": nn["text"], **gg})
        # dose response on test pairs: mean logprob of Bella reply vs Gemma reply under the same steering
        lpb, nb = W.mean_logprob(test_prefix, test_bella, batch_size=args.harvest_batch)
        lpg, ng = W.mean_logprob(test_prefix, test_gemma, batch_size=args.harvest_batch)
        dose.append({"layer": key[0], "coef": c, "mean_lp_bella": float(lpb.mean()), "mean_lp_gemma": float(lpg.mean()),
                     "contrast": float((lpb - lpg).mean()), "contrast_se": float((lpb - lpg).std(ddof=1) / math.sqrt(len(lpb))),
                     "lp_bella": lpb.tolist(), "lp_gemma": lpg.tolist()})
        write_jsonl(gens_path, rows)
        dump(out / "dose_response.json", dose)
        log(f"cell {cell_i}/{n_cells} layer={key[0]} coef={c} done in {time.time()-tc:.0f}s; "
            f"contrast={dose[-1]['contrast']:.3f}; sample: {g[0]['text'][:120]!r}")
    W.set_steering(None, None)
    # perplexity of neutral continuations under the unsteered model
    log("stage2: scoring neutral continuations under the unsteered model")
    ppl = []
    by_cell = {}
    for r in rows:
        if r["set"] == "neutral":
            by_cell.setdefault((r["layer"], r["coef"]), []).append(r)
    bos = W.tok.bos_token or ""
    for (l, c), rr in by_cell.items():
        conts = [x["text"] if x["text"] else " " for x in rr]
        lp, nt = W.mean_logprob([bos + x["prompt"] for x in rr], conts, batch_size=args.harvest_batch)
        ppl.append({"layer": l, "coef": c, "mean_nll": float(-lp.mean()), "ppl": float(math.exp(-lp.mean())),
                    "median_ppl": float(np.exp(np.median(-lp))), "mean_len": float(nt.mean()), "nll": (-lp).tolist()})
    dump(out / "neutral_perplexity.json", ppl)
    all_coefs = sorted({r["coef"] for r in rows if r["layer"] != -1})
    dump(out / "stage2_meta.json", {"layers": layers, "coefs": all_coefs, "coefs_this_run": coefs, "iteration": args.iteration, "n_chat_prompts": len(chat_items), "n_neutral": len(neutral),
                                    "gen_tokens": args.gen_tokens, "median_norm": [float(med_norm[l]) for l in layers],
                                    "timing_s": time.time() - t0, "cells": n_cells})
    log(f"stage2 done in {time.time()-t0:.0f}s")


def load_sae(layer: int):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(SAE_REPO, f"layer_{layer:02d}_s0/sae.pt", repo_type="dataset")
    m = hf_hub_download(SAE_REPO, f"layer_{layer:02d}_s0/meta.json", repo_type="dataset")
    sd = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    meta = json.load(open(m))
    d_in, n_feat = meta["d_in"], meta["n_features"]
    W_enc = W_dec = b_enc = b_dec = thr = None
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        kl = k.lower()
        if v.ndim == 2 and set(v.shape) == {d_in, n_feat}:
            if "enc" in kl:
                W_enc = v.float()
            elif "dec" in kl:
                W_dec = v.float()
        elif v.ndim == 1 and v.shape[0] == n_feat:
            if "thresh" in kl:
                thr = v.float()
            elif "enc" in kl or "b_e" in kl:
                b_enc = v.float()
        elif v.ndim == 1 and v.shape[0] == d_in:
            b_dec = v.float()
    assert W_enc is not None and W_dec is not None, list(sd.keys())
    if W_enc.shape[0] != d_in:
        W_enc = W_enc.T  # -> [d_in, n_feat]
    if W_dec.shape[0] != n_feat:
        W_dec = W_dec.T  # -> [n_feat, d_in]
    if b_enc is None:
        b_enc = torch.zeros(n_feat)
    if b_dec is None:
        b_dec = torch.zeros(d_in)
    thr_kind = None
    for k in sd:
        if "thresh" in k.lower():
            thr_kind = k
    return {"W_enc": W_enc.cuda(), "W_dec": W_dec.cuda(), "b_enc": b_enc.cuda(), "b_dec": b_dec.cuda(),
            "thr": (thr.cuda() if thr is not None else None), "thr_key": thr_kind, "keys": list(sd.keys()), "meta": meta}


def sae_encode(sae, x):
    pre = (x - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]
    if sae["thr"] is None:
        return torch.relu(pre)
    t = sae["thr"]
    if sae["thr_key"] and "log" in sae["thr_key"].lower():
        t = torch.exp(t)
    return pre * (pre > t)


def stage3(W: Wrapped, out: Path, args, s1, dirs):
    layers = s1["stage2_layers"]["layers"]
    if args.smoke:
        layers = layers[:1]
    pairs = read_jsonl(DATA / "pairs.jsonl")
    test_pairs = [p for p in pairs if p["split"] == "test"]
    gemma_replies = {g["pid"]: g for g in read_jsonl(out / "gemma_replies.jsonl")}
    if args.smoke:
        test_pairs = [p for p in pairs[: args.smoke_n]][-6:]
        for p in test_pairs:
            gemma_replies.setdefault(p["pid"], {"text": "I can help with that."})
    prefixes = [W.chat_prefix(p["prompt"]) for p in test_pairs]
    log(f"stage3: token-level harvest of {len(test_pairs)} test pairs at layers {layers}")
    hb = W.harvest(prefixes, [p["bella"] for p in test_pairs], batch_size=args.harvest_batch, token_level_layers=layers, phase="s3_bella")
    hg = W.harvest(prefixes, [gemma_replies[p["pid"]]["text"] or " " for p in test_pairs], batch_size=args.harvest_batch,
                   token_level_layers=layers, phase="s3_gemma")
    res = {"layers": layers, "per_layer": {}}
    for l in layers:
        t0 = time.time()
        sae = load_sae(l)
        v = torch.tensor(dirs[l]).cuda()
        Wd = sae["W_dec"]
        Wd_unit = Wd / Wd.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos = Wd_unit @ v
        order = torch.argsort(-cos.abs())
        # least-squares reconstruction of v from the top-k decoder directions
        ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        frac = {}
        for k in ks:
            A = Wd_unit[order[:k]].T  # [d, k]
            sol = torch.linalg.lstsq(A, v[:, None]).solution
            frac[k] = float(1 - ((A @ sol)[:, 0] - v).norm() ** 2 / v.norm() ** 2)
        need = {}
        for target in (0.5, 0.8, 0.95):
            need[str(target)] = next((k for k in ks if frac[k] >= target), None)
        # encoder projection of the direction
        enc_proj = (v @ sae["W_enc"])
        top_enc = torch.topk(enc_proj, 20).indices.tolist()
        # feature activations on test pairs, token-level, averaged per reply then across replies
        def feat_means(tok_level):
            per = []
            for (i, h) in tok_level:
                z = sae_encode(sae, h.cuda().float())
                per.append((i, z.mean(0), z.max(0).values))
            per.sort(key=lambda x: x[0])
            M = torch.stack([p[1] for p in per])
            X = torch.stack([p[2] for p in per])
            return M, X
        Mb, Xb = feat_means(hb["token_level"][l])
        Mg, Xg = feat_means(hg["token_level"][l])
        diff = Mb.mean(0) - Mg.mean(0)
        sp = torch.sqrt((Mb.var(0) + Mg.var(0)) / 2).clamp(min=1e-6)
        dz = diff / sp
        top_b = torch.topk(diff, 20).indices.tolist()
        top_g = torch.topk(-diff, 20).indices.tolist()
        # calibration: does our hook site / scale match the SAE's training site? (meta: L0 ~ k, EV ~ best_ev)
        Xtok = torch.cat([h.cuda().float() for _, h in hg["token_level"][l]] + [h.cuda().float() for _, h in hb["token_level"][l]])
        Z = sae_encode(sae, Xtok)
        Xhat = Z @ sae["W_dec"] + sae["b_dec"]
        ev = float(1 - ((Xtok - Xhat) ** 2).mean() / Xtok.var())
        calib = {"mean_l0": float((Z > 0).float().sum(-1).mean()), "ev": ev, "n_tokens": int(Xtok.shape[0]),
                 "rms_per_element": float(Xtok.pow(2).mean().sqrt()), "meta_mean_l0": sae["meta"].get("final_metrics", {}).get("mean_l0"),
                 "meta_ev": sae["meta"].get("final_metrics", {}).get("ev"),
                 "meta_activation_norm_probe": sae["meta"].get("final_metrics", {}).get("activation_norm_probe")}
        del Xtok, Z, Xhat
        # exemplars for top features (by |cos| with the direction and by activation diff)
        exemplars = {}
        want = list(dict.fromkeys(order[:10].tolist() + top_b[:10] + top_g[:10]))
        for f in want:
            ex = []
            for side, tl, pp in (("bella", hb["token_level"][l], test_pairs), ("gemma", hg["token_level"][l], test_pairs)):
                for (i, h) in tl:
                    z = sae_encode(sae, h.cuda().float())[:, f]
                    if z.max() > 0:
                        ex.append((float(z.max()), side, i, int(z.argmax())))
            ex.sort(reverse=True)
            exemplars[str(f)] = []
            for val, side, i, ti in ex[:5]:
                reply = test_pairs[i]["bella"] if side == "bella" else gemma_replies[test_pairs[i]["pid"]]["text"]
                ids = W.tok(reply, add_special_tokens=False)["input_ids"]
                tokstr = W.tok.decode(ids[ti:ti + 1]) if ti < len(ids) else "?"
                exemplars[str(f)].append({"act": val, "side": side, "prompt": test_pairs[i]["prompt"][:160], "reply": reply[:300], "token": tokstr, "token_idx": ti})
        res["per_layer"][str(l)] = {
            "calibration": calib, "sae_keys": sae["keys"], "sae_meta": {k: sae["meta"].get(k) for k in ("d_in", "n_features", "k", "final_metrics")},
            "recon_fraction_by_k": frac, "features_needed": need,
            "top_cos_features": [{"f": int(f), "cos": float(cos[f]), "dec_norm": float(Wd[f].norm())} for f in order[:20].tolist()],
            "top_encoder_projection": [{"f": int(f), "proj": float(enc_proj[f])} for f in top_enc],
            "top_features_bella_gt_gemma": [{"f": int(f), "diff": float(diff[f]), "d": float(dz[f]), "mean_bella": float(Mb.mean(0)[f]), "mean_gemma": float(Mg.mean(0)[f]),
                                             "frac_replies_active_bella": float((Xb[:, f] > 0).float().mean()), "frac_replies_active_gemma": float((Xg[:, f] > 0).float().mean())} for f in top_b],
            "top_features_gemma_gt_bella": [{"f": int(f), "diff": float(diff[f]), "d": float(dz[f]), "mean_bella": float(Mb.mean(0)[f]), "mean_gemma": float(Mg.mean(0)[f]),
                                             "frac_replies_active_bella": float((Xb[:, f] > 0).float().mean()), "frac_replies_active_gemma": float((Xg[:, f] > 0).float().mean())} for f in top_g],
            "n_features_with_abs_d_gt_0p5": int((dz.abs() > 0.5).sum()),
            "n_features_with_abs_d_gt_1": int((dz.abs() > 1).sum()),
            "cos_abs_top1": float(cos.abs().max()), "cos_abs_p99": float(torch.quantile(cos.abs(), 0.99)),
            "exemplars": exemplars, "timing_s": time.time() - t0,
        }
        dump(out / "stage3.json", res)
        log(f"stage3 layer {l}: calib L0={calib['mean_l0']:.1f} (meta {calib['meta_mean_l0']}) EV={calib['ev']:.3f} (meta {calib['meta_ev']}); need {need}, top cos {float(cos.abs().max()):.3f}, "
            f"features |d|>1: {int((dz.abs() > 1).sum())}, {time.time()-t0:.0f}s")
        del sae
        torch.cuda.empty_cache()
    log("stage3 done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--stages", default="1,2,3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_n", type=int, default=36)
    ap.add_argument("--gen_batch", type=int, default=48)
    ap.add_argument("--harvest_batch", type=int, default=16)
    ap.add_argument("--gen_tokens", type=int, default=128)
    ap.add_argument("--coefs", default=None, help="comma list overriding COEFS (second steering iteration)")
    ap.add_argument("--iteration", type=int, default=1)
    ap.add_argument("--seed_from", default=None, help="directory holding a previous run's outputs to resume from (copied into out)")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    root = Path(args.out or os.environ.get("SILICO_EXPERIMENT_ARTIFACTS_DIR", HERE / "results" / "artifacts"))
    out = root / args.model / ("smoke" if args.smoke else "run")
    out.mkdir(parents=True, exist_ok=True)
    if args.seed_from:
        import shutil
        for f in ("stage1.json", "directions.npz", "gemma_replies.jsonl", "steering_generations.jsonl", "dose_response.json",
                  "neutral_perplexity.json", "stage2_meta.json", "stage3.json"):
            src = Path(args.seed_from) / f
            if src.exists():
                shutil.copy(src, out / f)
        log(f"seeded out dir from {args.seed_from}: {sorted(p.name for p in out.iterdir())}")
    log(f"config: {vars(args)} out={out}")
    stages = [int(s) for s in args.stages.split(",")]
    report_progress(step=0, total_steps=1, phase="loading")
    W = Wrapped(MODELS[args.model]["hf"], MODELS[args.model]["revision"])
    log("chat prefix sample:", repr(W.chat_prefix("hi there")))
    if 1 in stages:
        s1, dirs, med = stage1(W, out, args)
    else:
        s1 = json.load(open(out / "stage1.json"))
        dz = np.load(out / "directions.npz")
        dirs = dz[s1["pooling_choice"]["chosen"]]
        med = np.array(s1["median_norm_gemma"])
    if 2 in stages:
        stage2(W, out, args, s1, dirs, med)
    if 3 in stages and args.model == "E2B":
        stage3(W, out, args, s1, dirs)
    log("ALL DONE ->", out)


if __name__ == "__main__":
    main()
