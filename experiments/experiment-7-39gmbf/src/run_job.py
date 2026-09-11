#!/usr/bin/env python3
"""Full Gemma 2 2B SAE atlas, layers 0-25, one sequential job, in-RAM pools.

Drives the trainer's own orchestrator in-process exactly as
    SAE_POOL_BACKEND=ram SAE_SEQ_LEN=1024 SAE_NO_PREPRODUCE=1 SAE_EAGER_POOL_CLEANUP=1 \
    SAE_NO_RESUME_POOL=1 SAE_EXCLUDE_SPECIAL=1 SAE_WANDB_SINGLE_RUN=1 \
    SAE_WANDB_RUN_NAME=gemma-2-2b_L00-25_pile1024_ram \
    python run_atlas.py --config configs/gemma2_2b.yaml --layer-range 0,25 \
        --capture rolling-hf --pool-retention 1
(SAE_SEQ_LEN comes from the config env block). The wrapper owns the single W&B run (the
trainer reuses it under SAE_WANDB_SINGLE_RUN=1 and logs each layer under Lnn/), wraps
module-level trainer functions to time pool production / training / push, counts special
tokens over the token shards, amends each layer's meta.json with job provenance before the
trainer pushes it, verifies the Hub listing after every layer, samples memory, and writes
results/atlas_summary.csv. Trainer source changes are confined to results/trainer.patch.
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
MODEL_ID = "google/gemma-2-2b"
CORPUS_ID = "monology/pile-uncopyrighted"
HUB_REPO = "juiceb0xc0de/gemma-2-2b-SAE"
WANDB_PROJECT = "gemma-2-2b-SAE-atlas"
WANDB_NAME = "gemma-2-2b_L00-25_pile1024_ram"
OTHER_WANDB_PROJECTS = ("gemma-2-2b-SAE", "gemma-2-2b-SAE-timing")   # must stay untouched

T_START = time.time()


def log(msg):
    print(f"[run_job {time.strftime('%H:%M:%S')} +{time.time()-T_START:7.1f}s] {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    """Every `every` s: MemAvailable, cgroup usage, RSS/HWM, GPU -> W&B (sys/*) + series."""

    def __init__(self, wb, every=15.0):
        super().__init__(daemon=True)
        self.wb, self.every, self.stop = wb, every, threading.Event()
        self.series = []
        self.peak_rss = 0
        self.phase = "setup"

    def sample(self, tag=None):
        mi, cg, ps = meminfo(), cgroup_mem(), proc_status()
        row = {"t_s": round(time.time() - T_START, 1),
               "mem_available_gb": mi.get("MemAvailable", 0) / 1e9,
               "cgroup_current_gb": (cg.get("current") or 0) / 1e9,
               "rss_gb": ps.get("VmRSS", 0) / 1e9, "rss_peak_gb": ps.get("VmHWM", 0) / 1e9,
               "phase": self.phase}
        row.update(gpu_query())
        if tag:
            row["tag"] = tag
        self.peak_rss = max(self.peak_rss, ps.get("VmHWM", 0))
        self.series.append(row)
        if self.wb is not None:
            try:
                self.wb.log({f"sys/{k}": v for k, v in row.items() if k not in ("tag", "phase")})
            except Exception:
                pass
        return row

    def run(self):
        while not self.stop.is_set():
            try:
                self.sample()
            except Exception as e:  # noqa
                print(f"  [sampler] {e}")
            self.stop.wait(self.every)


# ---------------------------------------------------------------------------------
def with_retries(fn, what, attempts=4, base_delay=10.0):
    """Transient Hub/network errors (connection resets, 5xx) must not kill a 2 h job:
    retry a few times with backoff, then re-raise the last error."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa
            last = e
            if i + 1 < attempts:
                delay = base_delay * (2 ** i)
                log(f"WARNING {what} failed ({type(e).__name__}: {e}); retry {i+1}/{attempts-1} in {delay:.0f}s")
                time.sleep(delay)
    raise last


def hub_listing(token):
    from huggingface_hub import HfApi

    def _q():
        api = HfApi(token=token)
        info = api.dataset_info(HUB_REPO, files_metadata=True)
        files = sorted((s.rfilename, s.size) for s in info.siblings)
        return {"repo": HUB_REPO, "sha": info.sha, "last_modified": str(info.last_modified),
                "n_files": len(files), "files": files, "queried_at": utc()}
    return with_retries(_q, "hub listing")


def wandb_project_counts(entity=None):
    """Run counts in the pre-existing projects (must be unchanged before/after)."""
    out = {}
    try:
        import wandb
        api = wandb.Api(timeout=60)
        ent = entity or api.default_entity
        for proj in OTHER_WANDB_PROJECTS + (WANDB_PROJECT,):
            try:
                runs = api.runs(f"{ent}/{proj}")
                out[proj] = len(list(runs))
            except Exception as e:  # noqa
                out[proj] = f"unavailable: {type(e).__name__}"
        out["_entity"] = ent
    except Exception as e:  # noqa
        out["_error"] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-batches", type=int, default=POOL_BATCHES)
    ap.add_argument("--layer-range", default="0,25")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke only: override config max_steps")
    ap.add_argument("--tag", default="atlas_L00-25_ram", help="output subdir under the artifacts dir")
    ap.add_argument("--ram-request-gib", type=int, default=200)
    ap.add_argument("--skip-ram-check", action="store_true", help="smoke only")
    ap.add_argument("--no-push", action="store_true", help="smoke only")
    ap.add_argument("--wandb-project", default=None, help="smoke only: override the W&B project")
    ap.add_argument("--wandb-name", default=None)
    args = ap.parse_args()
    pool_batches = args.pool_batches
    pool_bytes = pool_batches * BATCH_TOKENS * D_IN * 2
    ram_estimate = 2 * pool_bytes + 10e9
    push = not args.no_push
    wandb_project = args.wandb_project or WANDB_PROJECT
    wandb_name = args.wandb_name or WANDB_NAME
    start_layer, end_layer = map(int, args.layer_range.split(","))

    exp = Path(os.environ["SILICO_EXPERIMENT_RELATIVE_DIR"]).resolve()
    trainer = exp / "trainer"
    art = Path(os.environ["SILICO_EXPERIMENT_ARTIFACTS_DIR"]) / args.tag
    res = art / "results"
    res.mkdir(parents=True, exist_ok=True)
    (art / "logs").mkdir(exist_ok=True)
    # SAE outputs (sae.pt, meta.json, checkpoint_full.pt per layer) go to the durable
    # artifacts dir: <art>/saes/google_gemma-2-2b/layer_LL_s0/. Token shards (131 MB) and
    # the marker json live on the job-local worktree copy; activation pools live in RAM.
    data_dir = art
    scratch = exp / "scratch" / "job_rollcache"
    scratch.mkdir(parents=True, exist_ok=True)
    for stale in scratch.glob("tokens_*"):
        shutil.rmtree(stale, ignore_errors=True)
        log(f"deleted stale token cache {stale}")

    # ---- environment (before any trainer import: SEQ_LEN/backend/single-run are module-level)
    hf_home = exp / "scratch" / "hf_home"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONUNBUFFERED="1", HF_HOME=str(hf_home), HF_XET_HIGH_PERFORMANCE="1",
                      SAE_DATA_DIR=str(data_dir), SAE_SCRATCH_DIR=str(scratch),
                      SAE_BATCH_TOKENS=str(BATCH_TOKENS), SAE_SEQ_LEN=str(SEQ_LEN),
                      SAE_POOL_BACKEND="ram", SAE_NO_PREPRODUCE="1", SAE_EAGER_POOL_CLEANUP="1",
                      SAE_NO_RESUME_POOL="1", SAE_EXCLUDE_SPECIAL="1",
                      SAE_WANDB_SINGLE_RUN="1", SAE_WANDB_RUN_NAME=wandb_name,
                      PYTORCH_ALLOC_CONF="expandable_segments:True",
                      WANDB_MODE=os.environ.get("WANDB_MODE", "online"))
    os.environ.pop("SAE_PRODUCE_ONLY", None)
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
        f"cgroup.max={(cg.get('max') or 0)/1e9:.1f}GB effective_available={(eff or 0)/1e9:.1f}GB; "
        f"estimate(2 pools+10GB)={ram_estimate/1e9:.1f}GB requested={args.ram_request_gib}GiB "
        f"cpus={os.cpu_count()}")
    record = {"job_id": job_id, "trainer_commit": commit, "patch_sha256": patch_sha, "image_ref": image_ref,
              "model_id": MODEL_ID, "corpus": CORPUS_ID, "seq_len": SEQ_LEN, "batch_tokens": BATCH_TOKENS,
              "pool_batches": pool_batches, "pool_retention": 1, "layer_range": args.layer_range,
              "push": push, "hub_repo": HUB_REPO, "wandb_project": wandb_project, "wandb_name": wandb_name,
              "env_flags": {k: os.environ[k] for k in sorted(os.environ) if k.startswith("SAE_")},
              "pool_bytes": pool_bytes, "ram_estimate_bytes": ram_estimate, "ram_requested_gib": args.ram_request_gib,
              "mem": {"MemTotal": mi.get("MemTotal"), "MemAvailable_start": mi.get("MemAvailable"),
                      "cgroup_max": cg.get("max"), "cgroup_current_start": cg.get("current"),
                      "effective_available_start": eff},
              "cpu_count": os.cpu_count(), "layers": [], "status": "started", "start_utc": utc()}

    def save():
        (res / "job_record.json").write_text(json.dumps(record, indent=1, default=str))
    save()
    if eff is not None and eff < ram_estimate and not args.skip_ram_check:
        record["status"] = "aborted_insufficient_ram"
        save()
        raise SystemExit(f"ABORT before model load: effective available RAM {eff/1e9:.1f}GB < "
                         f"estimate {ram_estimate/1e9:.1f}GB (no disk fallback by design)")

    # ---- Hub / W&B state before -------------------------------------------------------
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
    wb_before = wandb_project_counts()
    record["wandb_run_counts_before"] = wb_before
    log(f"wandb run counts before: {wb_before}")
    save()

    # ---- W&B: the wrapper owns the single run; the trainer reuses it -----------------
    wb = None
    try:
        import wandb
        wb = wandb.init(project=wandb_project, name=wandb_name,
                        config={"trainer_commit": commit, "patch_sha256": patch_sha, "image_ref": image_ref,
                                "corpus": CORPUS_ID, "corpus_commit": ds_sha, "model_id": MODEL_ID,
                                "seq_len": SEQ_LEN, "batch_tokens": BATCH_TOKENS, "pool_batches": pool_batches,
                                "pool_retention": 1, "layer_range": args.layer_range, "pool_backend": "ram",
                                "no_preproduce": True, "eager_pool_cleanup": True, "no_resume_pool": True,
                                "exclude_special": True, "push": push,
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
    os.chdir(trainer)
    import torch
    import sae_trainer_rolling as T
    import sae_scheduler as S
    import run_atlas
    assert T.SEQ_LEN == SEQ_LEN and T._RAM and T._WANDB_SINGLE_RUN, (T.SEQ_LEN, T._RAM, T._WANDB_SINGLE_RUN)
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

    # per-layer rows -------------------------------------------------------------------
    rows: dict[int, dict] = {}
    fields = ["layer", "ev", "l0", "dead_pct", "stop_step", "stop_reason", "early_stopped",
              "produce_wall_s", "train_wall_s", "push_wall_s", "tokens_seen", "tokens_seen_nominal",
              "eligible_rows_per_batch_mean", "masked_fraction", "produce_tokens_per_s",
              "resident_pool_gb_during_train", "peak_rss_gb", "hub_verified", "start_utc", "end_utc"]
    csv_path = res / "atlas_summary.csv"

    def row_for(L):
        return rows.setdefault(L, {"layer": L})

    def write_summary():
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for L in sorted(rows):
                w.writerow({k: rows[L].get(k) for k in fields})
        record["layers"] = [rows[L] for L in sorted(rows)]
        save()

    # pool production timing ------------------------------------------------------------
    orig_produce = T._produce_pool_hf_rolling

    def timed_produce(model, text_model, decoder_layers, layer, tok_dir, src_dir, dst_dir, device):
        sampler.phase = f"produce_L{layer:02d}"
        sampler.sample(tag=f"L{layer}_produce_start")
        r = row_for(layer)
        r["start_utc"] = utc()
        start = time.time()
        orig_produce(model, text_model, decoder_layers, layer, tok_dir, src_dir, dst_dir, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - start
        n = len(T._shard_paths(dst_dir))
        r["produce_wall_s"] = wall
        r["produce_tokens_per_s"] = n * BATCH_TOKENS / wall
        r["peak_rss_gb"] = proc_status().get("VmHWM", 0) / 1e9
        write_summary()
        sampler.sample(tag=f"L{layer}_produce_end")
        log(f"L{layer}: produced in {wall:.1f}s ({r['produce_tokens_per_s']/1e3:.1f}k tok/s); resident pools "
            f"{resident_pool_bytes()/1e9:.1f}GB ({resident_pools()}); peak RSS {r['peak_rss_gb']:.1f}GB")
        if wb is not None:
            wb.log({"layer/L": layer, "layer/produce_wall_s": wall, "layer/produce_tokens_per_s": r["produce_tokens_per_s"],
                    "layer/resident_pool_gb_after_produce": resident_pool_bytes() / 1e9, "layer/peak_rss_gb": r["peak_rss_gb"]})
    T._produce_pool_hf_rolling = timed_produce

    orig_rm = T._rm_pool

    def logged_rm(dir_path):
        orig_rm(dir_path)
        gb = resident_pool_bytes() / 1e9
        sampler.sample(tag=f"rm_{Path(str(dir_path)).name}")
        log(f"rm {Path(str(dir_path)).name}: resident pools now {gb:.1f}GB ({resident_pools()})")
    T._rm_pool = logged_rm

    # special-token count over the token shards ----------------------------------------
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
        trows = [("total_ids", None, counts["total"]),
                 ("bos", ids["bos"], counts["bos"]), ("bos_position0", ids["bos"], counts["bos_pos0"]),
                 ("bos_mid_window", ids["bos"], counts["bos_mid"]),
                 ("eos", ids["eos"], counts["eos"]), ("pad", ids["pad"], counts["pad"])]
        with open(res / "special_tokens.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["token", "id", "count", "fraction"])
            for name, tid, c in trows:
                w.writerow([name, tid, c, f"{c / counts['total']:.6f}"])
        record["tokens"] = {"shards": len(paths), "shapes": sorted(shapes), "wall_s": tok_wall, "ids": ids, "counts": counts,
                            "excluded_fraction": (counts["bos"] + counts["eos"] + counts["pad"]) / counts["total"]}
        save()
        log(f"token shards: {len(paths)} shapes={sorted(shapes)} in {tok_wall:.0f}s; special tokens {counts} ids={ids}")
        if wb is not None:
            import wandb
            wb.log({"tokens/wall_s": tok_wall, "tokens/n_shards": len(paths),
                    **{f"tokens/{k}": v for k, v in counts.items()},
                    "tokens/special_tokens": wandb.Table(columns=["token", "id", "count", "fraction"],
                                                          data=[[n, tid, c, c / counts["total"]] for n, tid, c in trows])})
    T._capture_token_pool = counted_tokens

    # capture the live scheduler / provider of the layer being trained ------------------
    live = {"scheduler": None, "provider": None}

    class RecordingScheduler(S.SAEEventControlScheduler):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            live["scheduler"] = self
    S.SAEEventControlScheduler = RecordingScheduler

    OrigProvider = T.RollingActivationProvider

    class RecordingProvider(OrigProvider):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            live["provider"] = self
            log(f"provider: {len(self.paths)} shards, exclude_ids={self.exclude_ids}, tok_dir={'set' if self.tok_dir else None}")
    T.RollingActivationProvider = RecordingProvider

    # amend meta.json with provenance right before the trainer pushes the layer folder.
    # _save_full_checkpoint runs every 1000 steps (no meta.json yet) and once after the
    # final save (meta.json present): only the latter is amended.
    orig_save_ckpt = T._save_full_checkpoint

    def amending_save_ckpt(out_dir, step, sae, optimizer, scheduler, rng_states, *a, **k):
        orig_save_ckpt(out_dir, step, sae, optimizer, scheduler, rng_states, *a, **k)
        meta_path = Path(out_dir) / "meta.json"
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text())
            prov = live["provider"]
            masked_frac = (prov.masked_rows / prov.total_rows) if (prov and prov.total_rows) else None
            fm = meta.get("final_metrics") or {}
            meta["provenance"] = {
                "job_id": job_id, "trainer_commit": commit, "patch_sha256": patch_sha, "image_ref": image_ref,
                "corpus": CORPUS_ID, "corpus_commit": ds_sha, "seq_len": SEQ_LEN, "pool_batches": pool_batches,
                "pool_backend": "ram", "exclude_ids": list(prov.exclude_ids) if prov else [],
                "exclude_special": True, "masked_fraction_observed": masked_frac,
                "tokens_seen_nominal": meta.get("total_tokens"),
                "tokens_seen": (None if masked_frac is None else int(round(meta["total_tokens"] * (1 - masked_frac)))),
                "stop_reason": getattr(scheduler, "stop_reason", ""), "stop_step": meta.get("n_steps"),
                "early_stopped": meta.get("early_stopped"),
                "final_ev": fm.get("ev"), "final_l0": fm.get("mean_l0"), "final_dead_pct": fm.get("dead_pct"),
                "wandb_project": wandb_project, "wandb_run": record.get("wandb_url"), "written_utc": utc(),
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            log(f"meta.json amended with provenance for {Path(out_dir).name}")
        except Exception as e:  # noqa
            log(f"WARNING meta.json amend failed: {e}")
    T._save_full_checkpoint = amending_save_ckpt

    # time the per-layer Hub push (the trainer calls HfApi.upload_folder after the save)
    import huggingface_hub
    orig_upload = huggingface_hub.HfApi.upload_folder
    push_walls: dict[str, float] = {}

    def timed_upload(self, *a, **k):
        t0 = time.time()
        out = with_retries(lambda: orig_upload(self, *a, **k), f"push {k.get('path_in_repo')}")
        push_walls[k.get("path_in_repo") or "?"] = time.time() - t0
        log(f"push {k.get('path_in_repo')}: {time.time()-t0:.1f}s")
        return out
    huggingface_hub.HfApi.upload_folder = timed_upload

    # training wall + per-layer verification -------------------------------------------
    orig_train = T.train_sae_on_activations

    def timed_train(layer, d_in, seed, provider, **kw):
        sampler.phase = f"train_L{layer:02d}"
        r = row_for(layer)
        r["resident_pool_gb_during_train"] = resident_pool_bytes() / 1e9
        sampler.sample(tag=f"L{layer}_train_start")
        log(f"L{layer}: training starts; resident pools {r['resident_pool_gb_during_train']:.1f}GB ({resident_pools()})")
        t0 = time.time()
        out = orig_train(layer, d_in, seed, provider, **kw)
        r["train_wall_s"] = time.time() - t0
        sched, prov = live["scheduler"], live["provider"]
        fm = out if isinstance(out, dict) else {}
        r["ev"], r["l0"], r["dead_pct"] = fm.get("ev"), fm.get("mean_l0"), fm.get("dead_pct")
        r["stop_step"] = getattr(sched, "total_steps", None)
        r["stop_reason"] = getattr(sched, "stop_reason", "")
        r["early_stopped"] = getattr(sched, "should_stop", None)
        meta_path = Path(T.SAE_DIR) / f"layer_{layer:02d}_s{seed}" / "meta.json"
        try:
            meta = json.loads(meta_path.read_text())
            r["stop_step"] = meta.get("n_steps")
            r["tokens_seen_nominal"] = meta.get("total_tokens")
        except Exception:
            pass
        if prov and prov.total_rows:
            mf = prov.masked_rows / prov.total_rows
            r["masked_fraction"] = mf
            r["eligible_rows_per_batch_mean"] = BATCH_TOKENS * (1 - mf)
            if r.get("tokens_seen_nominal"):
                r["tokens_seen"] = int(round(r["tokens_seen_nominal"] * (1 - mf)))
        r["push_wall_s"] = push_walls.get(f"layer_{layer:02d}_s{seed}")
        r["peak_rss_gb"] = max(r.get("peak_rss_gb") or 0, proc_status().get("VmHWM", 0) / 1e9)
        # verify the Hub listing shows this layer before the orchestrator advances
        if push:
            try:
                listing = hub_listing(hf_token)
                names = {f for f, _ in listing["files"]}
                need = {f"layer_{layer:02d}_s{seed}/sae.pt", f"layer_{layer:02d}_s{seed}/meta.json"}
                r["hub_verified"] = need <= names
                (res / "hub_listing_latest.json").write_text(json.dumps(listing, indent=1))
                if not r["hub_verified"]:
                    raise RuntimeError(f"Hub listing lacks {sorted(need - names)} after push of layer {layer}")
            except Exception as e:  # noqa
                r["hub_verified"] = False
                write_summary()
                raise
        else:
            r["hub_verified"] = None
        r["end_utc"] = utc()
        write_summary()
        sampler.sample(tag=f"L{layer}_train_end")
        log(f"L{layer}: ev={r['ev']} l0={r['l0']} dead={r['dead_pct']} stop_step={r['stop_step']} "
            f"reason={r['stop_reason']!r} train {r['train_wall_s']:.0f}s push {r['push_wall_s']} "
            f"masked_frac={r.get('masked_fraction')} hub_verified={r['hub_verified']}")
        if wb is not None:
            wb.log({"layer/L": layer, "layer/train_wall_s": r["train_wall_s"], "layer/push_wall_s": r["push_wall_s"] or 0,
                    "layer/final_ev": r["ev"], "layer/final_l0": r["l0"], "layer/final_dead_pct": r["dead_pct"],
                    "layer/stop_step": r["stop_step"], "layer/masked_fraction": r.get("masked_fraction"),
                    "layer/resident_pool_gb_during_train": r["resident_pool_gb_during_train"]})
        return out
    T.train_sae_on_activations = timed_train

    # ---- run the trainer's orchestrator exactly as the CLI would --------------------
    argv = ["run_atlas.py", "--config", str(exp / "configs/gemma2_2b.yaml"), "--layer-range", args.layer_range,
            "--capture", "rolling-hf", "--pool-retention", "1"]
    if not push:
        argv.append("--no-push")
    if pool_batches != POOL_BATCHES:
        argv += ["--pool-batches", str(pool_batches)]
    if args.max_steps is not None:
        argv += ["--max-steps", str(args.max_steps)]
    if args.wandb_project:
        argv += ["--wandb-project", args.wandb_project]
    record["trainer_argv"] = argv
    log(f"launching trainer: {' '.join(argv)}  env flags: {record['env_flags']}")
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
        record["end_utc"] = utc()
        sampler.stop.set(); sampler.sample(tag="end")
        (res / "mem_series.json").write_text(json.dumps(sampler.series))
        with open(res / "mem_series.csv", "w", newline="") as f:
            keys = ["t_s", "tag", "phase", "mem_available_gb", "cgroup_current_gb", "rss_gb", "rss_peak_gb",
                    "gpu_util_pct", "gpu_mem_used_gb"]
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
            for rr in sampler.series:
                w.writerow(rr)
        sae_dir = Path(T.SAE_DIR)
        record["sae_dir_files"] = sorted(str(p.relative_to(sae_dir)) for p in sae_dir.rglob("*") if p.is_file()) if sae_dir.exists() else []
        # no pool ever touched disk: the scratch dir must hold only token shards + markers
        record["scratch_files"] = sorted(str(p.relative_to(scratch)) for p in scratch.rglob("*") if p.is_file())
        record["pool_files_on_disk"] = [p for p in record["scratch_files"] if p.startswith("pool_") or "resume_pool" in p]
        try:
            after = hub_listing(hf_token)
            (res / "hub_listing_after.json").write_text(json.dumps(after, indent=1))
            layer_dirs = sorted({f.split("/")[0] for f, _ in after["files"] if f.startswith("layer_")})
            record["hub_layer_dirs_after"] = layer_dirs
            log(f"hub after: {after['n_files']} files sha={after['sha']} layer dirs={len(layer_dirs)}")
        except Exception as e:  # noqa
            log(f"WARNING hub listing after failed: {e}")
        write_summary()
        if wb is not None:
            try:
                import wandb
                wb.summary.update({"status": status, "trainer_wall_s": record["trainer_wall_s"],
                                   "total_wall_s": record["total_wall_s"], "peak_rss_gb": record["peak_rss_gb"],
                                   "n_layers_done": len([r for r in rows.values() if r.get("ev") is not None])})
                if rows:
                    wb.log({"atlas_summary": wandb.Table(columns=fields,
                                                         data=[[rows[L].get(k) for k in fields] for L in sorted(rows)])})
                wb.finish()
            except Exception as e:  # noqa
                log(f"WARNING wandb finalize failed: {e}")
        wb_after = wandb_project_counts()
        record["wandb_run_counts_after"] = wb_after
        record["other_wandb_projects_unchanged"] = all(
            wb_before.get(p) == wb_after.get(p) for p in OTHER_WANDB_PROJECTS)
        log(f"wandb run counts after: {wb_after} (other projects unchanged={record['other_wandb_projects_unchanged']})")
        save()
        log(f"done status={status} total {record['total_wall_s']/60:.1f} min; results -> {res}")


if __name__ == "__main__":
    main()
