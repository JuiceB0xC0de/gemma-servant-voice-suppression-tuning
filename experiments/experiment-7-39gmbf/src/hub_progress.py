#!/usr/bin/env python3
"""Pod-side progress poll: list juiceb0xc0de/gemma-2-2b-SAE and read every layer's
meta.json provenance (written by this job's driver). Prints a per-layer table of the
layers whose provenance was written after --since (UTC ISO) and writes results/hub_progress.json."""
import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "juiceb0xc0de/gemma-2-2b-SAE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-11T19:50:00Z")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "hub_progress.json"))
    a = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.dataset_info(REPO, files_metadata=True)
    files = {s.rfilename: s.size for s in info.siblings}
    layers = sorted({f.split("/")[0] for f in files if f.startswith("layer_") and f.endswith("/meta.json")})
    rows = []
    for ld in layers:
        p = hf_hub_download(REPO, f"{ld}/meta.json", repo_type="dataset", token=os.environ.get("HF_TOKEN"),
                            force_download=True)
        m = json.load(open(p))
        prov = m.get("provenance") or {}
        fm = m.get("final_metrics") or {}
        rows.append({"layer_dir": ld, "written_utc": prov.get("written_utc"), "job_id": prov.get("job_id"),
                     "ev": fm.get("ev"), "l0": fm.get("mean_l0"), "dead_pct": fm.get("dead_pct"),
                     "stop_step": m.get("n_steps"), "early_stopped": m.get("early_stopped"),
                     "stop_reason": prov.get("stop_reason"), "masked_fraction": prov.get("masked_fraction_observed"),
                     "tokens_seen": prov.get("tokens_seen"), "sae_pt_bytes": files.get(f"{ld}/sae.pt"),
                     "new": bool(prov.get("written_utc") and prov["written_utc"] >= a.since)})
    out = {"repo": REPO, "sha": info.sha, "last_modified": str(info.last_modified), "n_files": len(files), "layers": rows}
    Path(a.out).write_text(json.dumps(out, indent=1))
    new = [r for r in rows if r["new"]]
    print(f"repo sha {info.sha}  files {len(files)}  layer dirs {len(rows)}  new since {a.since}: {len(new)}")
    print("layer        ev      l0   dead%  stop  early reason                       masked   written")
    for r in rows:
        flag = "*" if r["new"] else " "
        ev = f"{r['ev']:.4f}" if r["ev"] is not None else "-"
        print(f"{flag}{r['layer_dir']:<12} {ev:>6} {str(r['l0']):>6} {str(r['dead_pct']):>6} {str(r['stop_step']):>5} "
              f"{str(r['early_stopped']):>5} {str(r['stop_reason'])[:28]:<28} {str(r['masked_fraction'])[:8]:<8} {r['written_utc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
