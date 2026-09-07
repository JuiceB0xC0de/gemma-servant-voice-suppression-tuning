#!/usr/bin/env python3
"""Join step for the Gemma 4 E4B SAE atlas.

1. Collect layer_NN_s0/{sae.pt,meta.json} from the three chain-job output trees
   (staged read-only under $SILICO_INPUT_ARTIFACTS_DIR) into one atlas directory
   under the experiment outputs. checkpoint_full.pt and *_latest dirs are never
   copied: only sae.pt + meta.json are retained/published.
2. Run verify_atlas.py locally (no HF check yet). Layers that miss the EV floor or
   L0 window are flagged, never excluded.
3. Write atlas_summary.csv (+ the EV/L0 figure bundle) from the 42 metas.
4. Create the private HF dataset repo and push all 42 layers ONCE with the
   trainer's own push_layer / push_summary, plus atlas_summary.csv.
5. Re-run verify_atlas.py with the HF check.

Usage (inside the job, from the worktree root):
  python3 src/join_atlas.py --sources <dir> [<dir> ...] --dest <atlas_dir> \
      --hf-repo juiceb0xc0de/gemma-4-e4b-it-SAE [--no-push] [--results <dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "trainer"))


def find_layers(sources: list[Path]) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for src in sources:
        for meta in sorted(src.rglob("layer_*_s0/meta.json")):
            d = meta.parent
            if not (d / "sae.pt").is_file():
                print(f"  [skip] {d}: sae.pt missing")
                continue
            L = int(d.name[len("layer_"):len("layer_") + 2])
            if L in found:
                print(f"  [dup] layer {L}: {d} (keeping {found[L]})")
                continue
            found[L] = d
    return found


def copy_layer(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("sae.pt", "meta.json"):
        s, d = src / name, dst / name
        if d.is_file() and d.stat().st_size == s.stat().st_size:
            continue
        tmp = d.with_suffix(d.suffix + ".part")
        shutil.copyfile(s, tmp)
        os.replace(tmp, d)


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--dest", required=True, help="atlas dir: .../saes/google_gemma-4-e4b-it")
    ap.add_argument("--results", default=None, help="dir for verify/summary outputs (default: <dest>/../..)")
    ap.add_argument("--hf-repo", default="juiceb0xc0de/gemma-4-e4b-it-SAE")
    ap.add_argument("--hf-repo-type", default="dataset")
    ap.add_argument("--n-layers", type=int, default=42)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--verify-args", default="--d-in 2560 --n-features 81920 --k 50 --ev-floor 0.85 --l0-window 40,60 --dead-max 0.01")
    args = ap.parse_args()

    dest = Path(args.dest)
    results = Path(args.results) if args.results else dest.parent.parent
    results.mkdir(parents=True, exist_ok=True)
    sources = [Path(s) for s in args.sources]

    print(f"== join: sources={sources} dest={dest}")
    found = find_layers(sources)
    missing = [L for L in range(args.n_layers) if L not in found]
    print(f"   found {len(found)}/{args.n_layers} layers; missing={missing}")
    if missing:
        json.dump({"missing_layers": missing, "found": {L: str(p) for L, p in found.items()}},
                  open(results / "join_status.json", "w"), indent=1)
        print("== join ABORTED: incomplete atlas (nothing pushed)")
        return 3

    t0 = time.time()
    for L in range(args.n_layers):
        copy_layer(found[L], dest / f"layer_{L:02d}_s0")
        print(f"   copied layer {L:02d} from {found[L]}")
    print(f"   copy done in {time.time() - t0:.0f}s")

    # metas bundle + per-layer wall minutes (from the chain logs' "[ok] L.. done ... elapsed X min")
    bundle = {f"layer_{L:02d}": json.load(open(dest / f"layer_{L:02d}_s0" / "meta.json"))
              for L in range(args.n_layers)}
    json.dump(bundle, open(results / "atlas_metas.json", "w"), indent=1)
    wall: dict[str, float] = {}
    pat = re.compile(r"\[ok\] L(\d+) done:.*\|\s*elapsed\s+([\d.]+)\s*min")
    for src in sources:
        for log in sorted(src.rglob("logs/chain_*.log")):
            for line in open(log, errors="replace"):
                m = pat.search(line)
                if m:
                    wall[f"{int(m.group(1)):02d}"] = float(m.group(2))
    json.dump(wall, open(results / "wall_minutes.json", "w"), indent=1)
    print(f"   wall minutes parsed for {len(wall)} layers")

    # summary CSV from the assembled metas (figure is built pod-side; image lacks silico-figures)
    rc = run([sys.executable, str(HERE / "make_summary.py"), "--metas", str(results / "atlas_metas.json"),
              "--out", str(results / "atlas_summary.csv"), "--wall", str(results / "wall_minutes.json"),
              "--no-figure"])
    if rc:
        print(f"   [warn] make_summary rc={rc}")

    verify = [sys.executable, str(EXP / "verify_atlas.py"), "--root", str(dest),
              "--hf-repo", args.hf_repo, "--hf-repo-type", args.hf_repo_type,
              "--n-layers", str(args.n_layers), *args.verify_args.split()]
    rc_local = run(verify + ["--skip-hf", "--out", str(results / "verify_atlas_local.json")])
    print(f"== local verify rc={rc_local}")

    if args.no_push:
        return rc_local

    import run_atlas  # trainer's own upload path
    from huggingface_hub import HfApi, create_repo
    token = run_atlas.resolve_hf_token()
    if not token:
        print("== no HF token; cannot push"); return 4
    create_repo(args.hf_repo, repo_type=args.hf_repo_type, private=True, exist_ok=True, token=token)
    # never publish the scratch checkpoint even if one slipped through
    for p in dest.rglob("checkpoint_full.pt"):
        print(f"   [guard] removing {p} before push"); p.unlink()
    t0 = time.time()
    pushed = []
    for L in range(args.n_layers):
        ok = run_atlas.push_layer(dest, args.hf_repo, L, 0, repo_type=args.hf_repo_type)
        pushed.append(ok)
        print(f"   pushed {sum(pushed)}/{L + 1} elapsed {time.time() - t0:.0f}s", flush=True)
    summary = {
        "model_id": "google/gemma-4-E4B-it", "n_layers": args.n_layers, "seed": 0,
        "trainer_commit": os.environ.get("SAE_TRAINER_COMMIT"),
        "image_ref": os.environ.get("SAE_IMAGE_REF"),
        "join_job_id": os.environ.get("SAE_JOB_ID"),
        "layers_pushed": int(sum(pushed)),
        "layers": {f"layer_{L:02d}": json.load(open(dest / f"layer_{L:02d}_s0" / "meta.json")).get("final_metrics")
                   for L in range(args.n_layers)},
    }
    run_atlas.push_summary(dest, args.hf_repo, summary, repo_type=args.hf_repo_type)
    csv = results / "atlas_summary.csv"
    if csv.is_file():
        HfApi(token=token).upload_file(path_or_fileobj=str(csv), path_in_repo="atlas_summary.csv",
                                       repo_id=args.hf_repo, repo_type=args.hf_repo_type)
        print("[HF] uploaded atlas_summary.csv")
    print(f"== push done: {sum(pushed)}/{args.n_layers} layers in {time.time() - t0:.0f}s")

    rc_full = run(verify + ["--out", str(results / "verify_atlas.json")])
    print(f"== full verify rc={rc_full}")
    return rc_full


if __name__ == "__main__":
    sys.exit(main())
