#!/usr/bin/env python3
"""Job orchestrator: scratch A/B, rolling check, chain walk + training at layers 5/12/20,
comparison against Gemma Scope, HF push, restart pool, and W&B timing record.

Everything the trainer prints is streamed through here with a wall-clock stamp per
line, so per-phase timings come from the trainer's own marker lines without touching
its source. Runs inside the researcher's image via src/run_job.sh.

W&B projects:
  <showcase>         trainer-native per-layer runs (L05_s0, L12_s0, L20_s0) written by
                     the trainer itself, plus one `summary` run with the calibration and
                     decoder_overlap tables (written here).
  <showcase>-timing  one parent run with every time/* metric, system metrics, disk
                     sampling, throughput and the timing table; plus the trainer's two
                     short A/B layer-3 runs (redirected with --wandb-project).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

T_START = time.time()


def log(msg):
    print(f"[job {time.strftime('%H:%M:%S')} +{(time.time() - T_START) / 60:6.1f}m] {msg}",
          flush=True)


# --------------------------------------------------------------------------- helpers
class Timing:
    """time/<phase>_s records + a W&B mirror."""

    def __init__(self, wb):
        self.wb = wb
        self.phases = []

    def record(self, name, seconds, **tags):
        row = {"phase": name, "seconds": seconds, "start_utc": tags.pop("start_utc", None), **tags}
        self.phases.append(row)
        if self.wb is not None:
            self.wb.log({f"time/{name}_s": seconds, **{f"tag/{k}": v for k, v in tags.items()
                                                      if isinstance(v, (int, float))}})
        log(f"time/{name}_s = {seconds:.1f}  {tags if tags else ''}")

    def timed(self, name, fn, **tags):
        t0 = time.time()
        start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))
        out = fn()
        self.record(name, time.time() - t0, start_utc=start, **tags)
        return out


def run_streamed(cmd, log_path, env, cwd=None, marker_cb=None):
    """Run a subprocess, tee stdout to log_path with wall-clock stamps, and hand every
    line to marker_cb(line, t). Returns (rc, lines) where lines = [(t, text)]."""
    lines = []
    with open(log_path, "a") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=env, cwd=cwd)
        for raw in p.stdout:
            t = time.time()
            line = raw.rstrip("\n")
            f.write(f"{t:.3f} {line}\n")
            f.flush()
            lines.append((t, line))
            if marker_cb:
                marker_cb(line, t)
            # keep the job log readable: print trainer lines except the per-step spam
            if not line.startswith("  [STEP-TIME") and "Loading weights" not in line:
                print(line, flush=True)
        rc = p.wait()
    return rc, lines


def parse_trainer_lines(lines):
    """Derive per-layer phase timings and training stats from stamped trainer output."""
    out = {"layers": {}, "step_times_ms": {}, "resume_lines": []}
    cur = None
    prod_start = {}
    pending_wandb = None
    for t, line in lines:
        m = re.search(r"\[produce L(\d+)\] HF rolling block", line)
        if m:
            prod_start[int(m.group(1))] = t
        m = re.search(r"\[produce L(\d+)\] done in ([\d.]+)min", line)
        if m:
            L = int(m.group(1))
            out["layers"].setdefault(L, {})["produce_s"] = float(m.group(2)) * 60
        m = re.search(r"\[produce L(\d+)\] pool already present", line)
        if m:
            out["layers"].setdefault(int(m.group(1)), {})["produce_s"] = 0.0
            out["layers"][int(m.group(1))]["pool_reused"] = True
        if "[resume]" in line or "[bootstrap]" in line or "[cleanup]" in line:
            out["resume_lines"].append(line.strip())
        m = re.search(r"ROLLING SAE  layer=(\d+)", line)
        if m:
            cur = int(m.group(1))
            d = out["layers"].setdefault(cur, {})
            d["train_start_t"] = t
            out["step_times_ms"][cur] = []
            if pending_wandb:
                d["wandb_url"] = pending_wandb
                pending_wandb = None
        m = re.search(r"WandB: (https?://\S+)", line)
        if m:
            pending_wandb = m.group(1)
        m = re.search(r"\[STEP-TIME\s+(\d+)\] total=\s*([\d.]+)ms", line)
        if m and cur is not None:
            out["step_times_ms"][cur].append((int(m.group(1)), float(m.group(2))))
        m = re.search(r"b_dec set from", line)
        if m and cur is not None:
            out["layers"][cur]["bdec_done_t"] = t
        if cur is not None and re.search(r"^\s*step=", strip_ansi(line)):
            s = strip_ansi(line)
            d = out["layers"][cur]
            if "first_log_step_t" not in d:
                d["first_log_step_t"] = t
            d["last_log_line"] = s.strip()
            mm = re.search(r"step=\s*(\d+).*?L0=\s*([\d.]+).*?dead=\s*([\d.]+)%.*?ev=\s*(-?[\d.]+).*?tok/s=([\d.]+)k.*?tokens=([\d.]+)M", s)
            if mm:
                d.update(last_step=int(mm.group(1)), last_l0=float(mm.group(2)),
                         last_dead_pct=float(mm.group(3)), last_ev=float(mm.group(4)),
                         last_tok_s=float(mm.group(5)) * 1e3, last_tokens_M=float(mm.group(6)))
        for pat in (r"EARLY STOP @ step (\d+): (.*)", r"\[(EV PLATEAU STOP|POST-PEAK STOP|AGGRESSIVE-K STOP|DEAD CEILING) @ (\d+)\]"):
            m = re.search(pat, strip_ansi(line))
            if m and cur is not None:
                out["layers"][cur].setdefault("stop_lines", []).append(strip_ansi(line).strip())
        if cur is not None and line.startswith("Saved:"):
            out["layers"][cur]["saved_t"] = t
        m = re.search(r"\[ok\] L(\d+) done", line)
        if m:
            out["layers"].setdefault(int(m.group(1)), {})["ok_t"] = t
    for L, d in out["layers"].items():
        if "train_start_t" in d and "saved_t" in d:
            d["train_wall_s"] = d["saved_t"] - d["train_start_t"]
            if "bdec_done_t" in d:
                d["bdec_init_s"] = d["bdec_done_t"] - d["train_start_t"]
            if "first_log_step_t" in d:
                d["first_log_window_s"] = d["first_log_step_t"] - d["train_start_t"]
        if "saved_t" in d and "ok_t" in d:
            d["post_save_resume_pool_s"] = d["ok_t"] - d["saved_t"]
        st = [ms for _, ms in out["step_times_ms"].get(L, [])]
        if st:
            st_sorted = sorted(st)
            d["step_ms_mean"] = sum(st) / len(st)
            d["step_ms_p50"] = st_sorted[len(st) // 2]
            d["step_ms_p95"] = st_sorted[min(len(st) - 1, int(0.95 * len(st)))]
            d["step_ms_n"] = len(st)
            d["steps_per_s_from_step_time"] = 1000.0 / d["step_ms_mean"]
    return out


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(s):
    return ANSI.sub("", s)


class GpuSampler(threading.Thread):
    """nvidia-smi utilization every few seconds; per-phase means via marks."""

    def __init__(self, period=5.0):
        super().__init__(daemon=True)
        self.period = period
        self.samples = []   # (t, util%, mem_used_MiB)
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True,
                                   text=True, timeout=10).stdout.strip().split(",")
                self.samples.append((time.time(), float(q[0]), float(q[1])))
            except Exception:
                pass
            self._stop.wait(self.period)

    def mean_between(self, t0, t1):
        xs = [u for t, u, _ in self.samples if t0 <= t <= t1]
        return (sum(xs) / len(xs)) if xs else None

    def busy_seconds(self, thresh=10.0):
        return sum(self.period for _, u, _ in self.samples if u >= thresh)

    def stop(self):
        self._stop.set()


class DiskSampler(threading.Thread):
    def __init__(self, paths, wb, period=60.0):
        super().__init__(daemon=True)
        self.paths, self.wb, self.period = paths, wb, period
        self._stop = threading.Event()
        self.samples = []

    def run(self):
        while not self._stop.is_set():
            row = {}
            for name, p in self.paths.items():
                try:
                    st = shutil.disk_usage(p)
                    row[f"disk/{name}_free_gb"] = st.free / 1e9
                    row[f"disk/{name}_used_gb"] = du_gb(p)
                except Exception:
                    pass
            try:
                import psutil
                row["sys/ram_used_gb"] = psutil.virtual_memory().used / 1e9
            except Exception:
                pass
            self.samples.append((time.time(), row))
            if self.wb is not None and row:
                self.wb.log(row)
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()


def du_gb(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot / 1e9


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--pool-batches", type=int, default=500)
    ap.add_argument("--train-layers", default="5,12,20")
    ap.add_argument("--ab-layers", default="0,3")
    ap.add_argument("--ab-steps", type=int, default=200)
    ap.add_argument("--heldout-shards", type=int, default=61)   # 61 x 32768 = 2.0M tokens
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--skip-ab", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--wandb-showcase", default="gemma-2-2b-SAE")
    ap.add_argument("--fast-scratch", default="/local-scratch")
    args = ap.parse_args()

    exp = Path(os.environ["SILICO_EXPERIMENT_RELATIVE_DIR"]).resolve()
    trainer = exp / "trainer"
    art = Path(os.environ["SILICO_EXPERIMENT_ARTIFACTS_DIR"])
    sub = "gemma2_2b" if args.mode == "full" else "gemma2_2b_smoke"
    out = art / sub
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    model_id = "google/gemma-2-2b"
    slug = model_id.replace("/", "_").lower()
    commit = re.sub(r"^commit ", "", (trainer / "TRAINER_COMMIT.txt").read_text().strip())
    image_ref = os.environ.get("SAE_IMAGE_REF", "unknown")
    job_id = os.environ.get("SAE_JOB_ID") or os.environ.get("MODAL_TASK_ID", "unknown")
    fast = Path(args.fast_scratch) / sub
    fast.mkdir(parents=True, exist_ok=True)
    vol_scratch = out / "ab_volume_scratch"           # A/B "volume" placement
    train_layers = [int(x) for x in args.train_layers.split(",")]
    ab_a, ab_b = map(int, args.ab_layers.split(","))
    timing_project = args.wandb_showcase + "-timing"

    base_env = dict(os.environ)
    base_env.update(PYTHONUNBUFFERED="1", SAE_TRAINER_DIR=str(trainer),
                    PYTHONPATH=f"{trainer}:{os.environ.get('PYTHONPATH', '')}",
                    HF_HOME=str(fast / "hf_home"), TMPDIR=str(fast / "tmp"),
                    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                    SAE_TRAINER_COMMIT=commit, SAE_IMAGE_REF=image_ref, SAE_JOB_ID=str(job_id),
                    HF_XET_HIGH_PERFORMANCE="1")
    for k in ("WANDB_RUN_ID", "WANDB_NAME", "WANDB_PROJECT"):
        base_env.pop(k, None)
    Path(base_env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(base_env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    # ---- timing W&B run --------------------------------------------------------------
    wb = None
    try:
        import wandb
        wb = wandb.init(project=timing_project, name=f"{args.mode}_{job_id}",
                        config={"mode": args.mode, "pool_batches": args.pool_batches,
                                "train_layers": train_layers, "ab_layers": [ab_a, ab_b],
                                "ab_steps": args.ab_steps, "trainer_commit": commit,
                                "image_ref": image_ref, "job_id": job_id,
                                "fast_scratch_kind": "container-local overlay disk",
                                "fast_scratch_dir": str(fast), "volume_scratch_dir": str(vol_scratch),
                                "config_yaml": (trainer / "configs/gemma2_2b.yaml").read_text()})
        log(f"timing run: {wb.url}")
    except Exception as e:  # noqa
        log(f"WARNING wandb init failed: {e}")
    timing = Timing(wb)
    gpu = GpuSampler(); gpu.start()
    disk = DiskSampler({"fast": str(fast), "volume": str(out)}, wb); disk.start()
    record = {"mode": args.mode, "job_id": job_id, "trainer_commit": commit, "image_ref": image_ref,
              "fast_scratch": str(fast), "volume_scratch": str(vol_scratch),
              "timing_wandb_url": getattr(wb, "url", None), "phases": timing.phases,
              "ab": {}, "chain": {}, "layers": {}, "env": {}}

    def save_record():
        (out / "results" / "job_record.json").write_text(json.dumps(record, indent=1, default=str))

    # ---- setup -------------------------------------------------------------------------
    def setup():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"],
                       cwd=trainer, check=True)
        r = subprocess.run([sys.executable, "-c",
                            "import torch, transformers, sae_trainer_rolling as t, json;"
                            "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,"
                            "'transformers': transformers.__version__, 'trainer_file': t.__file__,"
                            "'gpu': torch.cuda.get_device_name(0)}))"],
                           capture_output=True, text=True, env=base_env, check=True)
        record["env"] = json.loads(r.stdout.strip().splitlines()[-1])
        record["env"]["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
        record["env"]["shm_bytes"] = shutil.disk_usage("/dev/shm").total
        record["env"]["cpu_count"] = os.cpu_count()
        try:
            import psutil
            record["env"]["ram_total_gb"] = psutil.virtual_memory().total / 1e9
        except Exception:
            pass
    timing.timed("setup", setup)
    log(f"env: {record['env']}")
    if wb is not None:
        wb.config.update({"env": record["env"]})

    # HF repo (private) for the SAEs
    if not args.no_push:
        def mkrepo():
            from huggingface_hub import create_repo
            create_repo("juiceb0xc0de/gemma-2-2b-SAE", repo_type="dataset", private=True, exist_ok=True)
        timing.timed("hf_create_repo", mkrepo)

    def dl_model():
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    timing.timed("model_download", dl_model)

    # ---- token pool + held-out --------------------------------------------------------
    tok_dir_fast = fast / "rollcache" / f"tokens_{slug}_s0"
    heldout_fw = out / "heldout" / "fineweb_edu"
    heldout_c4 = out / "heldout" / "c4_en_validation"
    env_fast = dict(base_env, SAE_SCRATCH_DIR=str(fast / "rollcache"))
    (fast / "rollcache").mkdir(parents=True, exist_ok=True)
    tok_json = out / "results" / "token_prep.json"
    rc, _ = timing.timed("token_prep", lambda: run_streamed(
        [sys.executable, "-u", str(exp / "src/walk_layers.py"), "--tokens-only",
         "--pool-batches", str(args.pool_batches), "--heldout-shards", str(args.heldout_shards),
         "--heldout-dir", str(heldout_fw), "--out", str(tok_json), "--tag", "tokens"],
        out / "logs/token_prep.log", env_fast))
    assert rc == 0, "token prep failed"
    rc, _ = timing.timed("heldout_c4_prep", lambda: run_streamed(
        [sys.executable, "-u", str(exp / "src/make_heldout_c4.py"), "--n-shards",
         str(args.heldout_shards), "--out-dir", str(heldout_c4)],
        out / "logs/heldout_c4.log", env_fast))
    assert rc == 0, "c4 held-out prep failed"

    # ---- rolling-forward check (4 real shards, all 26 layers) --------------------------
    rc, _ = timing.timed("rolling_check", lambda: run_streamed(
        [sys.executable, "-u", str(exp / "src/verify_rolling_forward.py"), "--model-id", model_id,
         "--tok-dir", str(tok_dir_fast), "--n-batches", "4", "--seqs-per-batch", "4",
         "--out", str(out / "results/rolling_check.json")],
        out / "logs/rolling_check.log", env_fast))
    record["rolling_check_rc"] = rc
    save_record()

    # ---- helper: trainer invocation ---------------------------------------------------
    def run_trainer(layer, scratch_dir, data_dir, max_steps, wandb_project, timing_every, tag):
        env = dict(base_env, SAE_SCRATCH_DIR=str(scratch_dir), SAE_DATA_DIR=str(data_dir))
        cmd = [sys.executable, "-u", str(trainer / "run_atlas.py"), "--config",
               str(trainer / "configs/gemma2_2b.yaml"), "--layer-range", f"{layer},{layer}",
               "--pool-batches", str(args.pool_batches), "--no-push",
               "--timing-every", str(timing_every), "--wandb-project", wandb_project]
        if max_steps is not None:
            cmd += ["--max-steps", str(max_steps)]
        logp = out / f"logs/trainer_{tag}_L{layer:02d}.log"
        t0 = time.time()
        rc, lines = run_streamed(cmd, logp, env)
        parsed = parse_trainer_lines(lines)
        parsed["rc"] = rc
        parsed["wall_s"] = time.time() - t0
        parsed["log"] = str(logp)
        parsed["cmd"] = cmd
        d = parsed["layers"].get(layer, {})
        for k in ("train_start_t", "saved_t"):
            if k in d:
                d[f"gpu_util_mean_train"] = gpu.mean_between(d["train_start_t"], d["saved_t"]) \
                    if "train_start_t" in d and "saved_t" in d else None
        return parsed

    def run_walk(a, b, scratch_dir, tag, src_dir=None, check=0):
        env = dict(base_env, SAE_SCRATCH_DIR=str(scratch_dir))
        js = out / f"results/walk_{tag}_L{a:02d}-{b:02d}.json"
        cmd = [sys.executable, "-u", str(exp / "src/walk_layers.py"), "--layers", f"{a},{b}",
               "--pool-batches", str(args.pool_batches), "--consume", "--out", str(js), "--tag", tag]
        if src_dir:
            cmd += ["--src-dir", str(src_dir)]
        if check:
            cmd += ["--check-against-trainer", str(check)]
        t0 = time.time()
        rc, lines = run_streamed(cmd, out / f"logs/walk_{tag}_L{a:02d}-{b:02d}.log", env)
        assert rc == 0, f"walk {tag} {a}-{b} failed"
        rec = json.loads(js.read_text())
        rec["wall_s"] = time.time() - t0
        for k in ("model_load_cpu_s", "model_to_gpu_s"):
            if k in rec.get("phases", {}):
                timing.record(f"{tag}_{k[:-2]}", rec["phases"][k])
        # GPU utilization per layer from the sampler using log stamps
        starts = {}
        for t, line in lines:
            m = re.search(r"L(\d+): producing", line)
            if m:
                starts[int(m.group(1))] = t
            m = re.search(r"L(\d+) done:", line)
            if m and int(m.group(1)) in starts:
                L = int(m.group(1))
                for row in rec["layers"]:
                    if row["layer"] == L:
                        row["gpu_util_mean"] = gpu.mean_between(starts[L], t)
        return rec

    # ---- A/B: volume vs fast scratch -------------------------------------------------
    if not args.skip_ab:
        for kind, sdir in (("volume", vol_scratch / "rollcache"), ("fast", fast / "rollcache")):
            sdir.mkdir(parents=True, exist_ok=True)
            if kind == "volume":
                t0 = time.time()
                shutil.copytree(tok_dir_fast, sdir / tok_dir_fast.name, dirs_exist_ok=True)
                timing.record("ab_token_pool_copy_to_volume", time.time() - t0)
            ab_data = out / f"ab_{kind}_data"
            walk = timing.timed(f"ab_walk_{kind}", lambda: run_walk(ab_a, ab_b, sdir, f"ab_{kind}",
                                                                    check=(5 if kind == "fast" else 0)))
            tr = timing.timed(f"ab_train_{kind}", lambda: run_trainer(
                ab_b, sdir, ab_data, args.ab_steps, timing_project, 1, f"ab_{kind}"))
            record["ab"][kind] = {"scratch_dir": str(sdir), "walk": walk, "train": tr}
            if wb is not None:
                for row in walk["layers"]:
                    wb.log({f"ab/{kind}/produce_s": row["total_s"], f"ab/{kind}/read_s": row["read_s"],
                            f"ab/{kind}/fwd_s": row["fwd_s"], f"ab/{kind}/write_s": row["write_s"],
                            f"ab/{kind}/tokens_per_s": row["tokens_per_s"], "ab/layer": row["layer"]})
                d = tr["layers"].get(ab_b, {})
                wb.log({f"ab/{kind}/train_wall_s": d.get("train_wall_s"),
                        f"ab/{kind}/step_ms_mean": d.get("step_ms_mean"),
                        f"ab/{kind}/step_ms_p95": d.get("step_ms_p95"),
                        f"ab/{kind}/steps_per_s": d.get("steps_per_s_from_step_time"),
                        f"ab/{kind}/resume_pool_copy_s": d.get("post_save_resume_pool_s")})
            save_record()
            # tear down: pools, resume pool, junk SAE
            t0 = time.time()
            for p in sdir.glob("pool_*"):
                shutil.rmtree(p, ignore_errors=True)
            shutil.rmtree(sdir / "resume_pool_s0", ignore_errors=True)
            shutil.rmtree(ab_data, ignore_errors=True)
            if kind == "volume":
                shutil.rmtree(vol_scratch, ignore_errors=True)
            timing.record(f"ab_cleanup_{kind}", time.time() - t0)

    # ---- chain: walk 0 -> 20, train 5, 12, 20 ------------------------------------------
    scratch = fast / "rollcache"
    data_dir = out / "sae_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    restart = out / "restart_pool"
    (restart / "resume_pool_s0").mkdir(parents=True, exist_ok=True)
    link = scratch / "resume_pool_s0"
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            shutil.rmtree(link)
    link.symlink_to(restart / "resume_pool_s0", target_is_directory=True)
    log(f"resume pool symlink {link} -> {restart / 'resume_pool_s0'}")

    def pool_dir(L):
        return scratch / f"pool_{slug}_L{L:02d}_s0"

    prev = -1
    chain_rows = []
    for T in train_layers:
        a = prev + 1
        walk = timing.timed(f"chain_walk_L{a:02d}-{T:02d}", lambda: run_walk(
            a, T, scratch, "chain", src_dir=(pool_dir(prev) if prev >= 0 else None)))
        chain_rows.extend(walk["layers"])
        tr = timing.timed(f"chain_train_L{T:02d}", lambda: run_trainer(
            T, scratch, data_dir, args.max_steps, args.wandb_showcase, 10, "chain"))
        assert tr["rc"] == 0, f"trainer failed at layer {T}"
        record["layers"][T] = tr
        d = tr["layers"].get(T, {})
        # meta.json provenance (added after the trainer writes it; no trainer edit)
        ldir = data_dir / "saes" / slug / f"layer_{T:02d}_s0"
        meta = json.loads((ldir / "meta.json").read_text())
        meta.update(trainer_commit=commit, image_ref=image_ref, job_id=str(job_id),
                    scratch_dir_kind="container-local overlay disk", scratch_dir=str(scratch),
                    capture="rolling-hf", pool_walk="src/walk_layers.py (trainer primitives)",
                    wandb_url=d.get("wandb_url"))
        (ldir / "meta.json").write_text(json.dumps(meta, indent=2))
        if wb is not None:
            for row in walk["layers"]:
                wb.log({"chain/produce_s": row["total_s"], "chain/read_s": row["read_s"],
                        "chain/fwd_s": row["fwd_s"], "chain/write_s": row["write_s"],
                        "chain/tokens_per_s": row["tokens_per_s"], "chain/layer": row["layer"],
                        "chain/gpu_util_mean": row.get("gpu_util_mean")})
            wb.log({"chain/train_wall_s": d.get("train_wall_s"), "chain/step_ms_mean": d.get("step_ms_mean"),
                    "chain/step_ms_p95": d.get("step_ms_p95"), "chain/steps_per_s": d.get("steps_per_s_from_step_time"),
                    "chain/train_layer": T, "chain/final_ev": meta["final_metrics"].get("ev"),
                    "chain/final_l0": meta["final_metrics"].get("mean_l0"),
                    "chain/final_dead_pct": meta["final_metrics"].get("dead_pct"),
                    "chain/stop_step": meta["n_steps"], "chain/train_tokens": meta["total_tokens"],
                    "chain/resume_pool_copy_s": d.get("post_save_resume_pool_s"),
                    "chain/gpu_util_mean_train": d.get("gpu_util_mean_train")})
        # durable copy of the weights (sae.pt + meta.json) and push
        dst = out / "saes" / f"layer_{T:02d}_s0"
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("sae.pt", "meta.json"):
            shutil.copyfile(ldir / f, dst / f)
        if not args.no_push:
            def push():
                r = subprocess.run(["hf", "upload", "juiceb0xc0de/gemma-2-2b-SAE", str(dst),
                                    f"layer_{T:02d}_s0", "--repo-type", "dataset", "--private"],
                                   capture_output=True, text=True, env=base_env)
                log(r.stdout[-500:] + r.stderr[-500:])
                return r.returncode
            record["layers"][T]["push_rc"] = timing.timed(f"hf_push_L{T:02d}", push)
        prev = T
        save_record()
    record["chain"]["walk_layers"] = chain_rows

    # restart manifest
    rp = restart / "resume_pool_s0"
    man_trainer = json.loads((rp / "manifest.json").read_text()) if (rp / "manifest.json").exists() else {}
    shard_files = sorted(p.name for p in rp.glob("shard_*.pt"))
    manifest = {"layer_index": man_trainer.get("last_completed_layer"), "shard_count": len(shard_files),
                "token_shard_ids": list(range(args.pool_batches)),
                "token_pool_dir_name": tok_dir_fast.name, "seed": 0,
                "config_sha256": __import__("hashlib").sha256(
                    (trainer / "configs/gemma2_2b.yaml").read_bytes()).hexdigest(),
                "trainer_commit": commit, "trainer_manifest": man_trainer,
                "layer_21_pool_present": False,
                "note": "resume_pool_s0 is the trainer's own resume pool (written through a symlink); "
                        "the token pool is also copied here so the cascade can reuse identical tokens.",
                "job_id": str(job_id)}
    shutil.copytree(tok_dir_fast, restart / tok_dir_fast.name, dirs_exist_ok=True)
    (restart / "restart_manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"restart manifest: layer {manifest['layer_index']} shards {manifest['shard_count']}")

    # ---- comparison vs Gemma Scope -----------------------------------------------------
    eval_out = out / "results" / "calibration"
    rc, _ = timing.timed("calibration_eval", lambda: run_streamed(
        [sys.executable, "-u", str(exp / "src/eval_calibration.py"), "--layers", args.train_layers,
         "--sae-root", str(data_dir / "saes" / slug), "--heldout", f"fineweb_edu={heldout_fw}",
         "--heldout", f"c4_en_validation={heldout_c4}", "--out-dir", str(eval_out)],
        out / "logs/calibration_eval.log", env_fast))
    record["calibration_rc"] = rc

    # ---- timing table + wrap-up --------------------------------------------------------
    if wb is not None:
        import wandb as _w
        cols = ["placement", "phase", "layer", "seconds", "read_s", "fwd_s", "write_s", "tokens_per_s",
                "step_ms_mean", "step_ms_p95", "steps_per_s", "gpu_util_mean"]
        data = []
        for kind, ab in record["ab"].items():
            for row in ab["walk"]["layers"]:
                data.append([kind, "produce", row["layer"], row["total_s"], row["read_s"], row["fwd_s"],
                             row["write_s"], row["tokens_per_s"], None, None, None, row.get("gpu_util_mean")])
            d = ab["train"]["layers"].get(ab_b, {})
            data.append([kind, f"train_{args.ab_steps}steps", ab_b, d.get("train_wall_s"), None, None, None,
                         None, d.get("step_ms_mean"), d.get("step_ms_p95"),
                         d.get("steps_per_s_from_step_time"), d.get("gpu_util_mean_train")])
            data.append([kind, "resume_pool_copy", ab_b, d.get("post_save_resume_pool_s"), None, None,
                         None, None, None, None, None, None])
        for row in chain_rows:
            data.append(["fast", "chain_produce", row["layer"], row["total_s"], row["read_s"], row["fwd_s"],
                         row["write_s"], row["tokens_per_s"], None, None, None, row.get("gpu_util_mean")])
        for T in train_layers:
            d = record["layers"][T]["layers"].get(T, {})
            data.append(["fast", "chain_train", T, d.get("train_wall_s"), None, None, None, None,
                         d.get("step_ms_mean"), d.get("step_ms_p95"), d.get("steps_per_s_from_step_time"),
                         d.get("gpu_util_mean_train")])
        wb.log({"timing": _w.Table(columns=cols, data=data)})
        wb.log({"time/job_total_s": time.time() - T_START, "time/gpu_busy_total_s": gpu.busy_seconds()})

    # ---- showcase summary run (tables only, no timing) --------------------------------
    if wb is not None and rc == 0:
        try:
            import csv
            import wandb as _w
            rows = list(csv.DictReader(open(eval_out / "calibration.csv")))
            ov = json.loads((eval_out / "decoder_overlap.json").read_text())
            srun = _w.init(project=args.wandb_showcase, name="summary", reinit="create_new",
                           config={"model_id": model_id, "trainer_commit": commit, "image_ref": image_ref,
                                   "job_id": job_id, "config_yaml": (trainer / "configs/gemma2_2b.yaml").read_text(),
                                   "reference": ov["reference"], "heldout_fineweb_shard_ids":
                                   json.loads(tok_json.read_text()).get("heldout", {}).get("shard_ids"),
                                   "layer_runs": {str(L): record["layers"][L]["layers"].get(L, {}).get("wandb_url")
                                                  for L in train_layers}})
            tbl = _w.Table(columns=list(rows[0].keys()), data=[list(r.values()) for r in rows])
            srun.log({"calibration": tbl})
            import torch
            cols = ["layer", "direction", "feature_index", "max_cosine"]
            data = []
            for L in train_layers:
                d = torch.load(eval_out / f"decoder_overlap_L{L:02d}.pt")
                for k, name in (("ours_to_ref_max_cos", "ours_to_ref"), ("ref_to_ours_max_cos", "ref_to_ours")):
                    v = d[k].tolist()
                    data += [[L, name, i, x] for i, x in enumerate(v)]
            otbl = _w.Table(columns=cols, data=data)
            srun.log({"decoder_overlap": otbl})
            for L in train_layers:
                for name in ("ours_to_ref", "ref_to_ours"):
                    sub_t = _w.Table(columns=["max_cosine"], data=[[r[3]] for r in data if r[0] == L and r[1] == name])
                    srun.log({f"decoder_overlap_hist/L{L:02d}_{name}": _w.plot.histogram(
                        sub_t, "max_cosine", title=f"L{L} {name} max decoder cosine")})
            record["summary_wandb_url"] = srun.url
            srun.finish()
        except Exception as e:  # noqa
            log(f"WARNING summary run failed: {e}")

    record["gpu_samples"] = gpu.samples
    record["disk_samples"] = disk.samples
    record["job_total_s"] = time.time() - T_START
    record["gpu_busy_total_s"] = gpu.busy_seconds()
    save_record()

    # ---- cleanup local scratch (keep nothing large outside the artifacts dir) --------
    for p in scratch.glob("pool_*"):
        shutil.rmtree(p, ignore_errors=True)
    gpu.stop(); disk.stop()
    if wb is not None:
        wb.finish()
    log(f"done in {(time.time() - T_START) / 60:.1f} min; results in {out}")


if __name__ == "__main__":
    main()
