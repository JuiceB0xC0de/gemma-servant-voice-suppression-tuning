#!/usr/bin/env python3
"""Tokenize a second held-out source (C4 English validation split) into token shards
with the trainer's exact shard recipe: documents tokenized with add_special_tokens
(BOS at each document start), concatenated, sliced into 16 x 2047 tokens per shard
with a BOS prepended to every 2048-token row.

Gemma 2 was not trained on C4 validation by us, and it is disjoint from FineWeb-Edu
(the SAE training distribution), so it serves as the "second general source".
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/gemma-2-2b")
    ap.add_argument("--n-shards", type=int, default=61)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-seqs", type=int, default=16)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    bos = tok.bos_token_id
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    it = iter(ds)
    per_shard = args.n_seqs * (args.seq_len - 1)
    buf = []
    n_docs = 0
    t0 = time.time()
    for i in range(args.n_shards):
        while sum(len(b) for b in buf) < per_shard:
            row = next(it)
            n_docs += 1
            text = row.get("text", "")
            if not text.strip():
                continue
            ids = tok(text, truncation=True, max_length=4096, return_tensors="pt",
                      add_special_tokens=True).input_ids[0]
            if len(ids) < 8:
                continue
            buf.append(ids)
        flat = torch.cat(buf)
        real = flat[:per_shard].view(args.n_seqs, args.seq_len - 1)
        rest = flat[per_shard:]
        buf = [rest] if rest.numel() else []
        shard = torch.cat([torch.full((args.n_seqs, 1), bos, dtype=real.dtype), real], dim=1)
        torch.save(shard.cpu(), out / f"shard_{i:05d}.pt")
    (out / "manifest.json").write_text(json.dumps({
        "source": "allenai/c4 en validation (streaming, in order)", "n_shards": args.n_shards,
        "n_docs": n_docs, "tokens_per_shard": args.n_seqs * args.seq_len,
        "model_id": args.model_id, "seconds": time.time() - t0}, indent=1))
    print(f"[c4] {args.n_shards} shards from {n_docs} docs in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
