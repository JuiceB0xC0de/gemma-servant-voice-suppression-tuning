#!/usr/bin/env python3
"""Check that the trainer's single-block rolling walk reproduces Gemma 2's own forward.

The trainer's `rolling-hf` / `rolling-hf-float` path runs one decoder block at a time
(`_make_llama_invariants` + `_run_hf_block`) on the previous block's pooled output.
The trainer only self-verifies its *generic* rolling path, so this script performs the
equivalent proof for the HF path on Gemma 2: walk blocks 0..n_layers-1 exactly the way
`_produce_pool_hf_rolling` does and compare each block output with
`model(..., output_hidden_states=True).hidden_states[L + 1]` on real token shards.

It also reports which capture path the trainer would select for this model
(`_is_hf_rolling_supported`, `_is_generic_rolling_supported`).

Usage (inside the trainer image, trainer importable through SAE_TRAINER_DIR):
    python3 src/verify_rolling_forward.py --model-id google/gemma-2-2b \
        --tok-dir <token pool dir> --n-batches 4 --out results/rolling_check.json
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
    ap.add_argument("--model-id", default="google/gemma-2-2b")
    ap.add_argument("--tok-dir", type=str, default=None,
                    help="trainer token pool dir (shard_*.pt of [n_seqs, 2048] ids); "
                         "if absent, a deterministic text probe is tokenized instead")
    ap.add_argument("--n-batches", type=int, default=4,
                    help="token shards to check (each [16, 2048])")
    ap.add_argument("--seqs-per-batch", type=int, default=4,
                    help="sequences taken from each shard (keeps 26 x hidden_states in VRAM)")
    ap.add_argument("--tol", type=float, default=2e-2,
                    help="max allowed relative error (max|diff| / rms(ref)) per layer")
    ap.add_argument("--attn", type=str, default=None,
                    help="force attn_implementation (default: model default)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    import torch
    sys.path.insert(0, os.environ.get("SAE_TRAINER_DIR", "trainer"))
    import sae_trainer_rolling as t

    device = torch.device("cuda")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_token = t._resolve_hf_token()
    tok = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    t0 = time.time()
    kw = {"token": hf_token, "dtype": torch.bfloat16, "device_map": "cpu"}
    if args.attn:
        kw["attn_implementation"] = args.attn
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kw)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    print(f"[load] {args.model_id} on GPU in {time.time() - t0:.0f}s", flush=True)

    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    n_layers = int(tcfg.num_hidden_layers)
    text_model, _, decoder_layers = t._find_text_model(model, n_layers)
    layer_types = list(getattr(tcfg, "layer_types", ["?"] * n_layers))
    import inspect
    sig = str(inspect.signature(decoder_layers[0].forward))
    hf_ok = t._is_hf_rolling_supported(text_model, decoder_layers)
    gen_ok = t._is_generic_rolling_supported(model, text_model, decoder_layers)
    capture_path = ("rolling-hf" if hf_ok else
                    "rolling-generic" if gen_ok else "hooked (auto)")
    print(f"[arch] model={type(model).__name__} text_model={type(text_model).__name__} "
          f"n_layers={n_layers} hidden={tcfg.hidden_size} attn={tcfg._attn_implementation} "
          f"sliding_window={getattr(tcfg, 'sliding_window', None)} "
          f"softcap={getattr(tcfg, 'attn_logit_softcapping', None)}", flush=True)
    print(f"[arch] block signature: {sig}", flush=True)
    print(f"[arch] trainer support: hf_rolling={hf_ok} generic_rolling={gen_ok} "
          f"-> trainer capture path: {capture_path}", flush=True)
    print(f"[arch] layer types: {layer_types}", flush=True)

    # Probe tokens: real token-pool shards when available (BOS-prepended, like training).
    batches = []
    if args.tok_dir and Path(args.tok_dir).is_dir():
        paths = t._shard_paths(Path(args.tok_dir))[: args.n_batches]
        for p in paths:
            ids = torch.load(p, map_location="cpu", weights_only=True)[: args.seqs_per_batch]
            batches.append(ids.to(device))
        print(f"[probe] {len(batches)} token shards from {args.tok_dir}, "
              f"each {tuple(batches[0].shape)}", flush=True)
    else:
        texts = [
            "The mitochondrion is the powerhouse of the cell. It produces ATP through "
            "oxidative phosphorylation, a process that couples electron transport to "
            "proton pumping across the inner membrane. ",
            "In 1687 Isaac Newton published the Principia, laying out three laws of motion "
            "and a law of universal gravitation that explained both falling apples and the "
            "orbits of the planets around the Sun. ",
        ]
        ids_list = []
        for s in texts:
            enc = tok(s * 60, add_special_tokens=False)["input_ids"][: 2047]
            ids_list.append([tok.bos_token_id] + enc)
        L_min = min(len(x) for x in ids_list)
        batches.append(torch.tensor([x[:L_min] for x in ids_list], device=device))
        print(f"[probe] synthetic ids {tuple(batches[0].shape)}", flush=True)

    per_layer = {L: {"max_abs": 0.0, "max_rel": 0.0, "rms_rel": 0.0, "rms_ref": 0.0, "nan": False}
                 for L in range(n_layers)}
    emb_max = 0.0
    for b_i, ids in enumerate(batches):
        # Reference: raw block outputs from the model's own forward, via hooks on every
        # decoder layer (hidden_states[-1] is post-final-norm in some HF versions).
        refs = {}
        handles = [
            decoder_layers[L].register_forward_hook(
                lambda m, i, o, L=L: refs.__setitem__(L, (o[0] if isinstance(o, tuple) else o)))
            for L in range(n_layers)
        ]
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        for h in handles:
            h.remove()
        hs = out.hidden_states
        # hidden_states[L+1] should equal the hooked block output except possibly the last.
        hs_vs_hook = max((hs[L + 1].float() - refs[L].float()).abs().max().item()
                         for L in range(n_layers - 1))
        inv = t._make_llama_invariants(text_model, ids)
        hidden = inv["inputs_embeds"]
        emb_max = max(emb_max, (hidden.float() - hs[0].float()).abs().max().item())
        for L in range(n_layers):
            out_L = t._run_hf_block(decoder_layers[L], hidden, inv)
            ref = refs[L].float()
            diff = out_L.float() - ref
            rms_ref = ref.pow(2).mean().sqrt().item()
            max_abs = diff.abs().max().item()
            d = per_layer[L]
            d["max_abs"] = max(d["max_abs"], max_abs)
            d["max_rel"] = max(d["max_rel"], max_abs / max(rms_ref, 1e-8))
            d["rms_rel"] = max(d["rms_rel"], diff.pow(2).mean().sqrt().item() / max(rms_ref, 1e-8))
            d["rms_ref"] = max(d["rms_ref"], rms_ref)
            d["nan"] |= bool(torch.isnan(out_L).any().item())
            hidden = out_L
        del hs, out, refs
        torch.cuda.empty_cache()
        print(f"[walk] batch {b_i + 1}/{len(batches)} done "
              f"(hidden_states[L+1] vs hooked block output max|diff|={hs_vs_hook:.3e})", flush=True)

    rows = []
    ok_all = True
    print(f"[walk] embeddings max|diff| vs hidden_states[0] = {emb_max:.3e}", flush=True)
    for L in range(n_layers):
        d = per_layer[L]
        ok = (d["max_rel"] <= args.tol) and not d["nan"]
        ok_all &= ok
        rows.append({"layer": L, "layer_type": layer_types[L], **d, "ok": ok})
        print(f"[walk] {'OK ' if ok else 'BAD'} L{L:02d} {layer_types[L]:>17s} "
              f"rms_ref={d['rms_ref']:9.4f} max|diff|={d['max_abs']:.3e} "
              f"max_rel={d['max_rel']:.3e} rms_rel={d['rms_rel']:.3e}", flush=True)

    summary = {
        "model_id": args.model_id, "n_layers": n_layers, "hidden": int(tcfg.hidden_size),
        "attn_implementation": tcfg._attn_implementation,
        "sliding_window": getattr(tcfg, "sliding_window", None),
        "attn_logit_softcapping": getattr(tcfg, "attn_logit_softcapping", None),
        "block_signature": sig,
        "trainer_hf_rolling_supported": hf_ok,
        "trainer_generic_rolling_supported": gen_ok,
        "trainer_capture_path": capture_path,
        "n_batches": len(batches), "probe_shape": list(batches[0].shape),
        "embeddings_max_abs_diff": emb_max,
        "tol": args.tol, "all_ok": ok_all, "layers": rows,
    }
    print(f"[result] all_ok={ok_all} over {n_layers} layers, capture path {capture_path}",
          flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=1))
        print(f"[result] wrote {args.out}", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
