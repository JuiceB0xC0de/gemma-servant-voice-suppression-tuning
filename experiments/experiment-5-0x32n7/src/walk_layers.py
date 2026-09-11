#!/usr/bin/env python3
"""Drive the trainer's own pool-production primitives to walk the residual chain
layer by layer without training, with per-phase timing.

Why this exists: at commit c796c2ee the trainer's `rolling-hf` orchestration
cannot walk the chain without training (a mid-chain start bootstraps with a hooked
full forward; the float variant that keeps the walk is broken on the HF branch).
This harness calls the exact functions `_produce_pool_hf_rolling` calls, in the
same order, on the same token pool, so the pools are byte-identical to the
trainer's own walk (checked with --check-against-trainer), and it additionally
times the read / forward / write phases the trainer does not print on this path.

Modes:
  --tokens-only        capture the trainer token pool (`_capture_token_pool`) and,
                       optionally, N extra held-out shards from the SAME stream
                       (batches 500.. are never seen by training)
  --layers A,B         walk blocks A..B (inclusive) from pool[A-1] (or --src-dir)
  --consume            delete pool[L-1] once pool[L] is complete (never deletes
                       --src-dir when it is the resume pool)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def log(msg):
    print(f"[walk {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/gemma-2-2b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-batches", type=int, default=500)
    ap.add_argument("--tokens-only", action="store_true")
    ap.add_argument("--heldout-shards", type=int, default=0,
                    help="extra token shards from the same stream after the pool")
    ap.add_argument("--heldout-dir", type=str, default=None)
    ap.add_argument("--layers", type=str, default=None, help="inclusive 'a,b'")
    ap.add_argument("--src-dir", type=str, default=None,
                    help="source pool for layer a (default: pool a-1 in SAE_SCRATCH_DIR)")
    ap.add_argument("--consume", action="store_true")
    ap.add_argument("--check-against-trainer", type=int, default=0,
                    help="also run the trainer's _produce_pool_hf_rolling on this many "
                         "shards for layer a and compare bytes")
    ap.add_argument("--out", type=str, required=True, help="timing JSON path")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    import torch
    sys.path.insert(0, os.environ.get("SAE_TRAINER_DIR", "trainer"))
    import sae_trainer_rolling as t

    t.MODEL_ID = args.model_id
    scratch = Path(os.environ["SAE_SCRATCH_DIR"])
    scratch.mkdir(parents=True, exist_ok=True)
    tok_dir = t._pool_dir(f"tokens_{t._slug(args.model_id)}_s{args.seed}")
    device = torch.device("cuda")
    record = {"tag": args.tag, "model_id": args.model_id, "scratch": str(scratch),
              "seed": args.seed, "pool_batches": args.pool_batches, "phases": {},
              "layers": []}

    hf_token = t._resolve_hf_token()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    cfg = AutoConfig.from_pretrained(args.model_id, token=hf_token)
    tcfg = getattr(cfg, "text_config", cfg)
    vocab_size = getattr(tcfg, "vocab_size", None) or len(tokenizer)
    bos_token_id = tokenizer.bos_token_id or 2
    record["phases"]["tokenizer_load_s"] = time.perf_counter() - t0

    if args.tokens_only:
        # One pass over the trainer's deterministic token stream (same dataset builder,
        # same seed, same BOS-prepend slicing as `_capture_token_pool`): batches
        # 0..pool_batches-1 become the training token pool, the next N batches the
        # held-out set (never trained on). Afterwards `_capture_token_pool` is called
        # and must report that it reuses the cached shards (vocab-range validated).
        hd = Path(args.heldout_dir) if args.heldout_shards > 0 else None
        if hd is not None:
            hd.mkdir(parents=True, exist_ok=True)
        tok_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        ds = t._build_token_dataset(hf_token=hf_token, batch_tokens=t.BATCH_TOKENS,
                                    seed=args.seed, use_pretok=False, model_id=args.model_id)
        it = iter(ds)
        n_seqs = t.BATCH_TOKENS // t.SEQ_LEN
        total = args.pool_batches + args.heldout_shards
        for i in range(total):
            batch = next(it)
            real = batch[: n_seqs * (t.SEQ_LEN - 1)].view(n_seqs, t.SEQ_LEN - 1)
            bos = torch.full((n_seqs, 1), bos_token_id, dtype=real.dtype)
            ids = torch.cat([bos, real], dim=1)
            if ids.min() < 0 or ids.max() >= vocab_size:
                raise SystemExit(f"token id out of range in batch {i}")
            if i < args.pool_batches:
                torch.save(ids.cpu(), tok_dir / f"shard_{i:05d}.pt")
            else:
                torch.save(ids.cpu(), hd / f"shard_{i:05d}.pt")
            if (i + 1) % 50 == 0:
                log(f"  token batch {i + 1}/{total} ({time.perf_counter() - t0:.0f}s)")
        record["phases"]["token_pool_s"] = time.perf_counter() - t0
        # Trainer must accept the cached pool as its own.
        t._capture_token_pool(hf_token, args.seed, args.pool_batches, use_pretok=False,
                              tok_dir=tok_dir, bos_token_id=bos_token_id,
                              model_id=args.model_id, vocab_size=vocab_size)
        record["token_pool_shards"] = len(t._shard_paths(tok_dir))
        assert record["token_pool_shards"] == args.pool_batches
        log(f"token pool: {record['token_pool_shards']} shards in "
            f"{record['phases']['token_pool_s']:.0f}s -> {tok_dir}")
        if hd is not None:
            record["heldout"] = {"dir": str(hd), "shard_ids": list(range(
                args.pool_batches, total)), "n_tokens": args.heldout_shards * t.BATCH_TOKENS}
            log(f"held-out: shards {args.pool_batches}..{total - 1} -> {hd}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=1))
        return

    a, b = map(int, args.layers.split(","))
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, token=hf_token, dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    record["phases"]["model_load_cpu_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.to(device)
    torch.cuda.synchronize()
    record["phases"]["model_to_gpu_s"] = time.perf_counter() - t0
    n_layers = int(tcfg.num_hidden_layers)
    text_model, _, decoder_layers = t._find_text_model(model, n_layers)
    assert t._is_hf_rolling_supported(text_model, decoder_layers)
    log(f"model on GPU (load {record['phases']['model_load_cpu_s']:.0f}s, "
        f"to-gpu {record['phases']['model_to_gpu_s']:.1f}s)")

    tok_paths = t._shard_paths(tok_dir)
    n = len(tok_paths)
    assert n >= args.pool_batches, f"token pool has {n} shards"
    n = args.pool_batches

    def pool_dir_for(L):
        return t._pool_dir(f"pool_{t._slug(args.model_id)}_L{L:02d}_s{args.seed}")

    src_dir = Path(args.src_dir) if args.src_dir else (pool_dir_for(a - 1) if a >= 1 else None)
    resume_dir = t._resume_pool_dir(args.seed)
    if a >= 1:
        assert src_dir is not None and len(t._shard_paths(src_dir)) >= n, \
            f"source pool {src_dir} incomplete"

    if args.check_against_trainer > 0:
        # Byte-equality of the harness loop vs the trainer's own producer on k shards.
        k = args.check_against_trainer
        chk = Path(str(pool_dir_for(a)) + "_trainercheck")
        t._rm_pool(chk)
        chk.mkdir(parents=True)
        # Run the trainer function on a k-shard token subset (temporary tok dir).
        tok_sub = Path(str(tok_dir) + "_sub")
        t._rm_pool(tok_sub)
        tok_sub.mkdir(parents=True)
        for i in range(k):
            shutil.copyfile(tok_paths[i], tok_sub / tok_paths[i].name)
        t._produce_pool_hf_rolling(model, text_model, decoder_layers, a, tok_sub, src_dir,
                                   chk, device)
        record["trainer_check"] = {"layer": a, "shards": k, "dir": str(chk)}

    for L in range(a, b + 1):
        dst = pool_dir_for(L)
        dst.mkdir(parents=True, exist_ok=True)
        src = src_dir if L == a else pool_dir_for(L - 1)
        t_read = t_fwd = t_write = 0.0
        t_first = None
        t0 = time.perf_counter()
        log(f"L{L}: producing {n} shards from {'tokens' if L == 0 else src} -> {dst}")
        for i in range(n):
            _t = time.perf_counter()
            ids = t._read_shard(tok_dir, i).to(device)
            inv = t._make_llama_invariants(text_model, ids)
            if L == 0:
                hidden = inv["inputs_embeds"]
            else:
                hidden = t._read_shard(src, i).to(device, dtype=inv["inputs_embeds"].dtype)
            torch.cuda.synchronize()
            t_read += time.perf_counter() - _t
            _t = time.perf_counter()
            out = t._run_hf_block(decoder_layers[L], hidden, inv)
            torch.cuda.synchronize()
            t_fwd += time.perf_counter() - _t
            _t = time.perf_counter()
            t._write_shard(dst, i, out)
            t_write += time.perf_counter() - _t
            if t_first is None:
                t_first = time.perf_counter() - t0
            if (i + 1) % max(1, n // 5) == 0:
                el = time.perf_counter() - t0
                log(f"  L{L} {i + 1}/{n}  {(i + 1) * t.BATCH_TOKENS / el / 1e3:.1f}k tok/s  "
                    f"read={t_read:.0f}s fwd={t_fwd:.0f}s write={t_write:.0f}s")
        total = time.perf_counter() - t0
        row = {"layer": L, "shards": n, "total_s": total, "read_s": t_read, "fwd_s": t_fwd,
               "write_s": t_write, "first_shard_s": t_first,
               "tokens_per_s": n * t.BATCH_TOKENS / total, "src": str(src) if L else "tokens",
               "dst": str(dst), "scratch": str(scratch)}
        record["layers"].append(row)
        log(f"L{L} done: {total / 60:.2f} min  read {t_read / total * 100:.0f}%  "
            f"fwd {t_fwd / total * 100:.0f}%  write {t_write / total * 100:.0f}%  "
            f"{row['tokens_per_s'] / 1e3:.1f}k tok/s")
        if "trainer_check" in record and L == a:
            chk = Path(record["trainer_check"]["dir"])
            same = all(
                (chk / p.name).read_bytes() == p.read_bytes()
                for p in t._shard_paths(chk)
            )
            # torch.save output may differ in pickle framing even for equal tensors;
            # compare tensors too.
            same_t = all(torch.equal(t._read_shard(chk, i), t._read_shard(dst, i))
                         for i in range(record["trainer_check"]["shards"]))
            record["trainer_check"].update(bytes_equal=same, tensors_equal=same_t)
            log(f"trainer-producer check on {record['trainer_check']['shards']} shards: "
                f"bytes_equal={same} tensors_equal={same_t}")
            t._rm_pool(chk)
            t._rm_pool(Path(str(tok_dir) + "_sub"))
            if not same_t:
                raise SystemExit("harness pool differs from trainer producer output")
        if args.consume and L >= 1 and src.exists() and src.resolve() != resume_dir.resolve():
            _t = time.perf_counter()
            t._rm_pool(src)
            row["consume_src_s"] = time.perf_counter() - _t
            log(f"  consumed {src} in {row['consume_src_s']:.1f}s")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=1))
    log("walk complete")


if __name__ == "__main__":
    main()
