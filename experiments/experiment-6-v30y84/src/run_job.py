#!/usr/bin/env python3
"""Produce-only, in-RAM pool timing run: Gemma 2 2B layers 0-3 through the trainer's
own orchestrator loop (run_atlas.py -> run_atlas_rolling), with the wrapper owning the
single W&B run, wall-clock timers around each pool production, the special-token
count over the token shards, memory sampling, and the Hub before/after listing.

The trainer is invoked in-process exactly as
    python run_atlas.py --config configs/gemma2_2b.yaml --layer-range 0,3 \
        --capture rolling-hf --pool-retention 1 --no-push
with SAE_PRODUCE_ONLY=1 SAE_POOL_BACKEND=ram (and SAE_SEQ_LEN=1024 from the config
env block) so that run_atlas.main() drives the orchestrator; the wrapper only wraps
module-level functions (_capture_token_pool, _produce_pool_hf_rolling, _rm_pool) to
time and record them. Trainer source changes are confined to results/trainer.patch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

POOL_BATCHES = 500
SEQ_LEN = 1024
BATCH_TOKENS = 32768
D_IN = 2304
POOL_BYTES = POOL_BATCHES * BATCH_TOKENS * D_IN * 2          # bf16, 75.5 GB
RAM_ESTIMATE_BYTES = 2 * POOL_BYTES + 10 * 1e9               # two resident pools + model/CPU overhead
MODEL_ID = "google/gemma-2-2b"
CORPUS_ID = "monology/pile-uncopyrighted"
HUB_REPO = "juiceb0xc0de/gemma-2-2b-SAE"
WANDB_PROJECT = "gemma-2-2b-SAE-timing"
WANDB_NAME = "produce_only_L00-03_ram"

T_START = time.time()


def log(msg):
    print(f"[run_job {time.strftime('%H:%M:%S')} +{time.time()-T_START:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------------
# memory helpers
# ---------------------------------------------------------------------------------
def meminfo():
    d = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                d[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return d


def cgroup_mem():
    out = {}
    for name, path in (("max", "/sys/fs/cgroup/memory.max"), ("current", "/sys/fs/cgroup/memory.current"),
                       ("max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
                       ("current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        if name in out:
            continue
        try:
            v = Path(path).read_text().strip()
            out[name] = None if v == "max" else int(v)
        except OSError:
            pass
    return out


_RSS_PEAK_SEEN = 0


def proc_status():
    """VmRSS from psutil (works under gVisor, which omits VmHWM in /proc/self/status);
    VmHWM = max(ru_maxrss, largest RSS sampled so far)."""
    global _RSS_PEAK_SEEN
    import resource
    d = {}
    try:
        import psutil
        d["VmRSS"] = psutil.Process().memory_info().rss
    except Exception:
        try:
            with open("/proc/self/statm") as f:
                d["VmRSS"] = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except OSError:
            d["VmRSS"] = 0
    _RSS_PEAK_SEEN = max(_RSS_PEAK_SEEN, d["VmRSS"])
    d["VmHWM"] = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, _RSS_PEAK_SEEN)
    return d


def effective_available_bytes():
    """min(MemAvailable, cgroup headroom) -- whichever binds first."""
    mi = meminfo().get("MemAvailable")
    cg = cgroup_mem()
    cands = [x for x in (mi, (cg["max"] - cg.get("current", 0)) if cg.get("max") else None) if x]
    return min(cands) if cands else None


def gpu_query():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        u, m, t = [float(x) for x in r.stdout.strip().split(",")]
        return {"gpu_util_pct": u, "gpu_mem_used_gb": m / 1024, "gpu_mem_total_gb": t / 1024}
    except Exception:
        return {}


class Sampler(threading.Thread):
    """Every `every` s: MemAvailable, cgroup usage, RSS/HWM, GPU -> W&B + in-memory series."""

    def __init__(self, wb, every=15.0):
        super().__init__(daemon=True)
        self.wb, self.every, self.stop = wb, every, threading.Event()
        self.series = []
        self.peak_rss = 0

    def sample(self, tag=None):
        mi, cg, ps = meminfo(), cgroup_mem(), proc_status()
        row = {"t_s": round(time.time() - T_START, 1),
               "mem_available_gb": mi.get("MemAvailable", 0) / 1e9,
               "cgroup_current_gb": (cg.get("current") or 0) / 1e9,
               "rss_gb": ps.get("VmRSS", 0) / 1e9, "rss_peak_gb": ps.get("VmHWM", 0) / 1e9}
        row.update(gpu_query())
        if tag:
            row["tag"] = tag
        self.peak_rss = max(self.peak_rss, ps.get("VmHWM", 0))
        self.series.append(row)
        if self.wb is not None:
            self.wb.log({f"sys/{k}": v for k, v in row.items() if k != "tag"})
        return row

    def run(self):
        while not self.stop.is_set():
            try:
                self.sample()
            except Exception as e:  # noqa
                print(f"  [sampler] {e}")
            self.stop.wait(self.every)


# ---------------------------------------------------------------------------------
def hub_listing(token):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    info = api.dataset_info(HUB_REPO, files_metadata=True)
    files = sorted((s.rfilename, s.size) for s in info.siblings)
    return {"repo": HUB_REPO, "sha": info.sha, "last_modified": str(info.last_modified),
            "n_files": len(files), "files": files, "queried_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-batches", type=int, default=POOL_BATCHES)
    ap.add_argument("--layer-range", default="0,3")
    ap.add_argument("--tag", default="produce_only_L00-03_ram", help="output subdir under the artifacts dir")
    ap.add_argument("--ram-request-gib", type=int, default=200)
    ap.add_argument("--skip-ram-check", action="store_true", help="smoke only")
    args = ap.parse_args()
    pool_batches = args.pool_batches
    pool_bytes = pool_batches * BATCH_TOKENS * D_IN * 2
    ram_estimate = 2 * pool_bytes + 10e9

    exp = Path(os.environ["SILICO_EXPERIMENT_RELATIVE_DIR"]).resolve()
    trainer = exp / "trainer"
    art = Path(os.environ["SILICO_EXPERIMENT_ARTIFACTS_DIR"]) / args.tag
    res = art / "results"
    res.mkdir(parents=True, exist_ok=True)
    (art / "logs").mkdir(exist_ok=True)
    # Token shards (131 MB) and the marker json live on the job-local worktree copy; the
    # activation pools live in process memory (SAE_POOL_BACKEND=ram).
    scratch = exp / "scratch" / "job_rollcache"
    data_dir = exp / "scratch" / "job_data"
    for d in (scratch, data_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Cache check does not compare shape: never reuse a tokens_* cache from another SEQ_LEN.
    for stale in scratch.glob("tokens_*"):
        shutil.rmtree(stale, ignore_errors=True)
        log(f"deleted stale token cache {stale}")

    # ---- environment (before any trainer import: SEQ_LEN/backend are module-level) ----
    hf_home = exp / "scratch" / "hf_home"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONUNBUFFERED="1", HF_HOME=str(hf_home), HF_XET_HIGH_PERFORMANCE="1",
                      SAE_DATA_DIR=str(data_dir), SAE_SCRATCH_DIR=str(scratch),
                      SAE_BATCH_TOKENS=str(BATCH_TOKENS), SAE_SEQ_LEN=str(SEQ_LEN),
                      SAE_PRODUCE_ONLY="1", SAE_POOL_BACKEND="ram",
                      PYTORCH_ALLOC_CONF="expandable_segments:True",
                      WANDB_MODE=os.environ.get("WANDB_MODE", "online"))
    for k in ("WANDB_RUN_ID", "WANDB_NAME", "WANDB_PROJECT"):
        os.environ.pop(k, None)
    hf_token = os.environ.get("HF_TOKEN")

    commit = (trainer / "TRAINER_COMMIT.txt").read_text().split()[-1]
    patch_path = exp / "results" / "trainer.patch"
    patch_sha = hashlib.sha256(patch_path.read_bytes()).hexdigest() if patch_path.exists() else None
    image_ref = os.environ.get("SAE_IMAGE_REF", "unknown")
    job_id = os.environ.get("SAE_JOB_ID") or os.environ.get("MODAL_TASK_ID", "unknown")

    # ---- RAM preflight (before model load) ----------------------------------------
    mi, cg = meminfo(), cgroup_mem()
    eff = effective_available_bytes()
    log(f"RAM: MemTotal={mi.get('MemTotal',0)/1e9:.1f}GB MemAvailable={mi.get('MemAvailable',0)/1e9:.1f}GB "
        f"cgroup.max={(cg.get('max') or 0)/1e9:.1f}GB cgroup.current={(cg.get('current') or 0)/1e9:.2f}GB "
        f"effective_available={(eff or 0)/1e9:.1f}GB; estimate(2 pools+10GB)={ram_estimate/1e9:.1f}GB "
        f"requested={args.ram_request_gib}GiB cpus={os.cpu_count()} shm={shutil.disk_usage('/dev/shm').total/1e9:.1f}GB")
    record = {"job_id": job_id, "trainer_commit": commit, "patch_sha256": patch_sha, "image_ref": image_ref,
              "model_id": MODEL_ID, "corpus": CORPUS_ID, "seq_len": SEQ_LEN, "batch_tokens": BATCH_TOKENS,
              "pool_batches": pool_batches, "pool_retention": 1, "layer_range": args.layer_range,
              "pool_bytes": pool_bytes, "ram_estimate_bytes": ram_estimate, "ram_requested_gib": args.ram_request_gib,
              "mem": {"MemTotal": mi.get("MemTotal"), "MemAvailable_start": mi.get("MemAvailable"),
                      "cgroup_max": cg.get("max"), "cgroup_current_start": cg.get("current"),
                      "effective_available_start": eff, "shm_total": shutil.disk_usage("/dev/shm").total},
              "cpu_count": os.cpu_count(), "layers": [], "status": "started"}

    def save():
        (res / "job_record.json").write_text(json.dumps(record, indent=1, default=str))
    save()
    if eff is not None and eff < ram_estimate and not args.skip_ram_check:
        record["status"] = "aborted_insufficient_ram"
        save()
        raise SystemExit(f"ABORT before model load: effective available RAM {eff/1e9:.1f}GB < "
                         f"estimate {ram_estimate/1e9:.1f}GB (no disk fallback by design)")

    # ---- Hub state before ------------------------------------------------------------
    try:
        before = hub_listing(hf_token)
        (res / "hub_listing_before.json").write_text(json.dumps(before, indent=1))
        log(f"hub before: {before['n_files']} files sha={before['sha']}")
    except Exception as e:  # noqa
        before = None
        log(f"WARNING hub listing before failed: {e}")
    try:
        from huggingface_hub import HfApi
        ds_sha = HfApi(token=hf_token).dataset_info(CORPUS_ID).sha
    except Exception as e:  # noqa
        ds_sha = f"unresolved: {e}"
    record["corpus_commit"] = ds_sha
    log(f"corpus {CORPUS_ID} @ {ds_sha}")

    # ---- W&B (the wrapper owns the single run) ------------------------------------
    wb = None
    try:
        import wandb
        wb = wandb.init(project=WANDB_PROJECT, name=WANDB_NAME if args.tag == "produce_only_L00-03_ram" else args.tag,
                        config={"trainer_commit": commit, "patch_sha256": patch_sha, "image_ref": image_ref,
                                "corpus": CORPUS_ID, "corpus_commit": ds_sha, "model_id": MODEL_ID,
                                "seq_len": SEQ_LEN, "batch_tokens": BATCH_TOKENS, "pool_batches": pool_batches,
                                "pool_retention": 1, "layer_range": args.layer_range,
                                "pool_backend": "ram", "produce_only": True,
                                "ram_requested_gib": args.ram_request_gib, "mem_start": record["mem"],
                                "job_id": job_id, "cpu_count": os.cpu_count(),
                                "config_yaml": (exp / "configs/gemma2_2b.yaml").read_text()})
        wb.define_metric("sys/*", step_metric="sys/t_s")
        wb.define_metric("layer/*", step_metric="layer/L")
        record["wandb_url"] = wb.url
        log(f"wandb run: {wb.url}")
    except Exception as e:  # noqa
        log(f"WARNING wandb init failed: {e}")
    save()
    sampler = Sampler(wb)
    sampler.start()

    # ---- import the trainer and wrap the functions we time -----------------------
    sys.path.insert(0, str(trainer))
    os.chdir(trainer)   # run_atlas resolves configs/ relative to itself; cwd only matters for ./data defaults
    import torch
    import sae_trainer_rolling as T
    import run_atlas
    assert T.SEQ_LEN == SEQ_LEN and T._RAM, (T.SEQ_LEN, T._RAM)
    record["env"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        import transformers
        record["env"]["transformers"] = transformers.__version__
    except Exception:
        pass
    log(f"env {record['env']}")

    def resident_pool_bytes():
        return sum(t.numel() * t.element_size() for t in T._RAM_POOLS.values())

    def resident_pools():
        return sorted({d.rsplit("/", 1)[-1] for (d, _) in T._RAM_POOLS})

    timing_rows = []
    csv_path = res / "pool_timing.csv"
    fields = ["layer", "wall_s", "tokens_per_s", "tokens", "shards", "resident_pool_gb_after_produce",
              "resident_pool_gb_after_cleanup", "resident_pools_after_cleanup", "peak_rss_gb",
              "mem_available_gb_after", "start_utc", "end_utc"]

    def write_timing():
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in timing_rows:
                w.writerow({k: r.get(k) for k in fields})
        lines = ["| layer | wall s | tokens/s | resident pool GB after produce | after cleanup | peak RSS GB | MemAvailable GB after |",
                 "|---|---|---|---|---|---|---|"]
        for r in timing_rows:
            ac = r.get("resident_pool_gb_after_cleanup")
            ac_s = "" if ac is None else f"{ac:.1f}"
            lines.append(f"| {r['layer']} | {r['wall_s']:.1f} | {r['tokens_per_s']:,.0f} | "
                         f"{r['resident_pool_gb_after_produce']:.1f} | {ac_s} | "
                         f"{r['peak_rss_gb']:.1f} | {r['mem_available_gb_after']:.1f} |")
        (res / "pool_timing.md").write_text(
            f"# Pool production timing: {MODEL_ID}, layers {args.layer_range}, {pool_batches} shards x "
            f"[{BATCH_TOKENS // SEQ_LEN},{SEQ_LEN}], corpus {CORPUS_ID}, in-process RAM pool, retention 1\n\n"
            "Performance under this configuration only (Pile, SEQ_LEN 1024, RAM backend); no RAM-vs-disk claim.\n\n"
            + "\n".join(lines) + "\n")

    orig_produce = T._produce_pool_hf_rolling

    def timed_produce(model, text_model, decoder_layers, layer, tok_dir, src_dir, dst_dir, device):
        sampler.sample(tag=f"L{layer}_start")
        start = time.time()
        start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        orig_produce(model, text_model, decoder_layers, layer, tok_dir, src_dir, dst_dir, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - start
        n = len(T._shard_paths(dst_dir))
        ps = proc_status()
        row = {"layer": layer, "wall_s": wall, "tokens": n * BATCH_TOKENS, "shards": n,
               "tokens_per_s": n * BATCH_TOKENS / wall,
               "resident_pool_gb_after_produce": resident_pool_bytes() / 1e9,
               "resident_pool_gb_after_cleanup": None, "resident_pools_after_cleanup": None,
               "peak_rss_gb": ps.get("VmHWM", 0) / 1e9,
               "mem_available_gb_after": meminfo().get("MemAvailable", 0) / 1e9,
               "start_utc": start_utc, "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        timing_rows.append(row)
        record["layers"].append(row)
        write_timing(); save()
        sampler.sample(tag=f"L{layer}_end")
        log(f"L{layer}: {wall:.1f}s  {row['tokens_per_s']/1e3:.1f}k tok/s  resident pools "
            f"{row['resident_pool_gb_after_produce']:.1f}GB ({resident_pools()})  peak RSS {row['peak_rss_gb']:.1f}GB")
        if wb is not None:
            wb.log({"layer/L": layer, "layer/wall_s": wall, "layer/tokens_per_s": row["tokens_per_s"],
                    "layer/resident_pool_gb_after_produce": row["resident_pool_gb_after_produce"],
                    "layer/peak_rss_gb": row["peak_rss_gb"], "layer/mem_available_gb_after": row["mem_available_gb_after"]})
    T._produce_pool_hf_rolling = timed_produce

    orig_rm = T._rm_pool

    def logged_rm(dir_path):
        orig_rm(dir_path)
        gb = resident_pool_bytes() / 1e9
        if timing_rows:
            timing_rows[-1]["resident_pool_gb_after_cleanup"] = gb
            timing_rows[-1]["resident_pools_after_cleanup"] = ";".join(resident_pools())
            write_timing(); save()
        sampler.sample(tag=f"rm_{Path(str(dir_path)).name}")
        log(f"rm {Path(str(dir_path)).name}: resident pools now {gb:.1f}GB ({resident_pools()})")
    T._rm_pool = logged_rm

    orig_tokens = T._capture_token_pool

    def counted_tokens(hf_token_, seed, pool_batches_, use_pretok, tok_dir, bos_token_id, model_id=None, vocab_size=None):
        t0 = time.time()
        orig_tokens(hf_token_, seed, pool_batches_, use_pretok, tok_dir, bos_token_id, model_id=model_id, vocab_size=vocab_size)
        tok_wall = time.time() - t0
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(model_id or MODEL_ID, token=hf_token)
        ids = {"bos": tk.bos_token_id, "eos": tk.eos_token_id, "pad": tk.pad_token_id}
        paths = T._shard_paths(tok_dir)
        counts = {k: 0 for k in ("total", "bos", "bos_pos0", "bos_mid", "eos", "pad")}
        shapes = set()
        for i in range(len(paths)):
            s = T._read_shard(tok_dir, i)
            shapes.add(tuple(s.shape))
            counts["total"] += s.numel()
            b = s == ids["bos"]
            counts["bos"] += int(b.sum()); counts["bos_pos0"] += int(b[:, 0].sum()); counts["bos_mid"] += int(b[:, 1:].sum())
            counts["eos"] += int((s == ids["eos"]).sum())
            counts["pad"] += int((s == ids["pad"]).sum()) if ids["pad"] is not None else 0
        rows = [("total_ids", None, counts["total"]),
                ("bos", ids["bos"], counts["bos"]), ("bos_position0", ids["bos"], counts["bos_pos0"]),
                ("bos_mid_window", ids["bos"], counts["bos_mid"]),
                ("eos", ids["eos"], counts["eos"]), ("pad", ids["pad"], counts["pad"])]
        with open(res / "special_tokens.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["token", "id", "count", "fraction"])
            for name, tid, c in rows:
                w.writerow([name, tid, c, f"{c / counts['total']:.6f}"])
        record["tokens"] = {"shards": len(paths), "shapes": sorted(shapes), "wall_s": tok_wall, "ids": ids, "counts": counts}
        save()
        log(f"token shards: {len(paths)} shapes={sorted(shapes)} in {tok_wall:.0f}s; special tokens {counts} ids={ids}")
        if wb is not None:
            import wandb
            wb.log({"tokens/wall_s": tok_wall, "tokens/n_shards": len(paths),
                    **{f"tokens/{k}": v for k, v in counts.items()},
                    "tokens/special_tokens": wandb.Table(columns=["token", "id", "count", "fraction"],
                                                          data=[[n, tid, c, c / counts["total"]] for n, tid, c in rows])})
    T._capture_token_pool = counted_tokens

    def forbid(name):
        def f(*a, **k):
            raise RuntimeError(f"{name} called in produce-only mode")
        return f
    T.train_sae_on_activations = forbid("train_sae_on_activations")
    T.RollingActivationProvider = forbid("RollingActivationProvider")

    # ---- run the trainer's orchestrator exactly as the CLI would --------------------
    argv = ["run_atlas.py", "--config", str(exp / "configs/gemma2_2b.yaml"), "--layer-range", args.layer_range,
            "--capture", "rolling-hf", "--pool-retention", "1", "--no-push"]
    if pool_batches != POOL_BATCHES:
        argv += ["--pool-batches", str(pool_batches)]
    record["trainer_argv"] = argv
    log(f"launching trainer: {' '.join(argv)}  env: SAE_PRODUCE_ONLY=1 SAE_POOL_BACKEND=ram SAE_SEQ_LEN(config)=1024")
    sys.argv = argv
    t_tr = time.time()
    status = "ok"
    try:
        run_atlas.main()
    except BaseException as e:  # noqa
        status = f"failed: {type(e).__name__}: {e}"
        log(status)
        raise
    finally:
        record["trainer_wall_s"] = time.time() - t_tr
        record["status"] = status
        record["final_resident_pool_gb"] = resident_pool_bytes() / 1e9
        record["final_resident_pools"] = resident_pools()
        record["peak_rss_gb"] = max(sampler.peak_rss, proc_status().get("VmHWM", 0)) / 1e9
        record["total_wall_s"] = time.time() - T_START
        sampler.stop.set(); sampler.sample(tag="end")
        (res / "mem_series.json").write_text(json.dumps(sampler.series))
        with open(res / "mem_series.csv", "w", newline="") as f:
            keys = ["t_s", "tag", "mem_available_gb", "cgroup_current_gb", "rss_gb", "rss_peak_gb",
                    "gpu_util_pct", "gpu_mem_used_gb"]
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
            for r in sampler.series:
                w.writerow(r)
        # SAE dir must hold no SAE outputs (produce-only)
        sae_dir = Path(os.environ["SAE_DATA_DIR"]) / "saes"
        record["sae_dir_files"] = sorted(str(p.relative_to(sae_dir)) for p in sae_dir.rglob("*") if p.is_file()) if sae_dir.exists() else []
        try:
            after = hub_listing(hf_token)
            (res / "hub_listing_after.json").write_text(json.dumps(after, indent=1))
            record["hub_unchanged"] = (before is not None and after["sha"] == before["sha"] and after["files"] == before["files"])
            log(f"hub after: {after['n_files']} files sha={after['sha']} unchanged={record['hub_unchanged']}")
        except Exception as e:  # noqa
            log(f"WARNING hub listing after failed: {e}")
        save()
        if wb is not None:
            wb.summary.update({"status": status, "trainer_wall_s": record["trainer_wall_s"],
                               "total_wall_s": record["total_wall_s"], "peak_rss_gb": record["peak_rss_gb"],
                               "hub_unchanged": record.get("hub_unchanged")})
            if timing_rows:
                import wandb
                wb.log({"pool_timing": wandb.Table(columns=fields, data=[[r.get(k) for k in fields] for r in timing_rows])})
            wb.finish()
        log(f"done status={status} total {record['total_wall_s']/60:.1f} min; results -> {res}")


if __name__ == "__main__":
    main()
