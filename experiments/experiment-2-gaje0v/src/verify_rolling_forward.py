#!/usr/bin/env python3
"""Check that the trainer's single-block rolling walk reproduces the model's own
forward pass for Gemma 4 E4B, layer by layer, across the shared-KV boundary.

The trainer's Gemma path (`rolling` / `rolling-float`) runs one decoder block at a
time on the previous block's pooled output and rebuilds shared KV for the
post-boundary layers from the source layer's block run over an anchor pool. The
trainer only self-verifies the *generic* rolling path, so this script performs the
equivalent proof for the Gemma path: walk blocks 0..max_layer exactly the way
`_produce_pool` does and compare each block output with
`model(..., output_hidden_states=True).hidden_states[L + 1]`.

Usage (inside the trainer image, trainer importable):
    python3 src/verify_rolling_forward.py --model-id google/gemma-4-E4B-it \
        --max-layer 30 --out results/forward_check_e4b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/gemma-4-E4B-it")
    ap.add_argument("--max-layer", type=int, default=30,
                    help="last layer to walk (inclusive); must be < n_layers-1 so the "
                         "HF hidden_states entry is a raw block output")
    ap.add_argument("--n-seqs", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--tol", type=float, default=2e-2,
                    help="max allowed relative error (max|diff| / rms(ref)) per layer")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    import torch
    sys.path.insert(0, os.environ.get("SAE_TRAINER_DIR", "trainer"))
    import sae_trainer_rolling as t
    from gemma_attention import prepare_gemma4_attention, select_gemma4_attention

    device = torch.device("cuda")
    enabled = prepare_gemma4_attention(args.model_id)
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    hf_token = t._resolve_hf_token()
    tok = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, token=hf_token, dtype=torch.bfloat16, device_map="cpu")
    select_gemma4_attention(model, enabled)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    print(f"[load] {args.model_id} on GPU in {time.time() - t0:.0f}s", flush=True)

    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    n_layers = int(tcfg.num_hidden_layers)
    text_model, _, decoder_layers = t._find_text_model(model, n_layers)
    kv_share_start = n_layers - int(getattr(tcfg, "num_kv_shared_layers", 0))
    print(f"[arch] n_layers={n_layers} hidden={tcfg.hidden_size} "
          f"num_kv_shared_layers={getattr(tcfg, 'num_kv_shared_layers', None)} "
          f"(boundary L{kv_share_start}) attn={tcfg._attn_implementation}", flush=True)
    kv_sources = {L: t._gemma_kv_source_layer(decoder_layers, L) for L in range(n_layers)}
    print(f"[arch] kv source map (shared layers only): "
          f"{ {L: s for L, s in kv_sources.items() if s is not None} }", flush=True)
    layer_types = list(tcfg.layer_types)
    print(f"[arch] full_attention layers: {[i for i, lt in enumerate(layer_types) if lt == 'full_attention']}",
          flush=True)

    assert args.max_layer < n_layers - 1, "max_layer must leave the final block out"

    # Deterministic natural-language probe (BOS-prepended, like the token pool).
    texts = [
        "The mitochondrion is the powerhouse of the cell. It produces ATP through "
        "oxidative phosphorylation, a process that couples electron transport to "
        "proton pumping across the inner membrane. ",
        "In 1687 Isaac Newton published the Principia, laying out three laws of motion "
        "and a law of universal gravitation that explained both falling apples and the "
        "orbits of the planets around the Sun. ",
    ][: args.n_seqs]
    ids_list = []
    for s in texts:
        enc = tok(s * 60, add_special_tokens=False)["input_ids"][: args.seq_len - 1]
        ids_list.append([tok.bos_token_id] + enc)
    L_min = min(len(x) for x in ids_list)
    ids = torch.tensor([x[:L_min] for x in ids_list], device=device)
    print(f"[probe] ids {tuple(ids.shape)}", flush=True)

    # Reference: the model's own forward.
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    print(f"[ref] {len(hs)} hidden states", flush=True)

    # Rolling walk, mirroring _produce_pool / _produce_shared_kv_pool.
    inv = t._make_invariants(text_model, tcfg, ids, static={})
    pools = {}
    hidden = inv["inputs_embeds0"]
    emb_diff = (hidden.float() - hs[0].float()).abs().max().item()
    print(f"[walk] embeddings max|diff| vs hidden_states[0] = {emb_diff:.3e}", flush=True)
    rows = []
    ok_all = True
    for L in range(0, args.max_layer + 1):
        src = kv_sources[L]
        shared = {}
        if src is not None:
            anchor = pools[src - 1]
            t._run_block(decoder_layers, tcfg, src, anchor, inv, shared_kv_states=shared)
        out_L = t._run_block(decoder_layers, tcfg, L, hidden, inv, shared_kv_states=shared)
        pools[L] = out_L
        ref = hs[L + 1].float()
        diff = (out_L.float() - ref)
        rms_ref = ref.pow(2).mean().sqrt().item()
        max_abs = diff.abs().max().item()
        rel = max_abs / max(rms_ref, 1e-8)
        rms_rel = diff.pow(2).mean().sqrt().item() / max(rms_ref, 1e-8)
        nan = bool(torch.isnan(out_L).any().item())
        ok = (rel <= args.tol) and not nan
        ok_all &= ok
        rows.append({"layer": L, "layer_type": layer_types[L], "kv_source": src,
                     "rms_ref": rms_ref, "max_abs_diff": max_abs, "max_rel": rel,
                     "rms_rel": rms_rel, "nan": nan, "ok": ok})
        tag = "OK " if ok else "BAD"
        print(f"[walk] {tag} L{L:02d} {layer_types[L]:>17s} kv_src={src!s:>4} "
              f"rms_ref={rms_ref:9.4f} max|diff|={max_abs:.3e} max_rel={rel:.3e} rms_rel={rms_rel:.3e}",
              flush=True)
        hidden = out_L
        # keep only anchors (src-1 for any future shared layer) and the running hidden
        needed = {s - 1 for s in kv_sources.values() if s is not None}
        for k in list(pools):
            if k not in needed and k != L:
                del pools[k]

    summary = {
        "model_id": args.model_id, "n_layers": n_layers, "hidden": int(tcfg.hidden_size),
        "kv_share_start": kv_share_start, "attn_implementation": tcfg._attn_implementation,
        "kv_sources": {str(k): v for k, v in kv_sources.items() if v is not None},
        "full_attention_layers": [i for i, lt in enumerate(layer_types) if lt == "full_attention"],
        "probe_shape": list(ids.shape), "tol": args.tol, "all_ok": ok_all, "layers": rows,
    }
    print(f"[result] all_ok={ok_all} over layers 0..{args.max_layer}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=1))
        print(f"[result] wrote {args.out}", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
