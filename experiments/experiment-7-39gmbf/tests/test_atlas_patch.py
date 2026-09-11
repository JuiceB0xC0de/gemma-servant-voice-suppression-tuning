"""CPU checks for the four env-gated additions on top of Task 6's trainer patch:
special-token exclusion in RollingActivationProvider (RAM and disk backends), the
accum_steps guard, single-run W&B prefixing, and the orchestrator's env gates."""
import importlib
import re
import sys
from pathlib import Path

import pytest
import torch

TRAINER = Path(__file__).resolve().parents[1] / "trainer"


def _load(monkeypatch, **env):
    for k in ("SAE_POOL_BACKEND", "SAE_WANDB_SINGLE_RUN", "SAE_SEQ_LEN"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.syspath_prepend(str(TRAINER))
    sys.modules.pop("sae_trainer_rolling", None)
    return importlib.import_module("sae_trainer_rolling")


def _make_pool(T, tok_dir, pool_dir, n_shards=3, n_seqs=4, seq_len=8, d=6, bos=2, eos=1, pad=0):
    g = torch.Generator().manual_seed(0)
    n_special = 0
    for i in range(n_shards):
        ids = torch.randint(3, 50, (n_seqs, seq_len), generator=g)
        ids[:, 0] = bos                       # BOS at position 0 of every sequence
        ids[0, 3] = eos                       # one mid-window EOS
        ids[1, 5] = bos                       # one mid-window BOS
        if i == 0:
            ids[2, 7] = pad
        n_special += int(((ids == bos) | (ids == eos) | (ids == pad)).sum())
        tok_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ids, tok_dir / f"shard_{i:05d}.pt")   # token shards always live on disk
        # activation row (s, p) encodes the token id so alignment is checkable
        acts = ids.unsqueeze(-1).expand(n_seqs, seq_len, d).to(torch.bfloat16).clone()
        T._write_shard(pool_dir, i, acts)
    return n_special, n_shards * n_seqs * seq_len


@pytest.mark.parametrize("backend", ["ram", "disk"])
def test_provider_excludes_special_tokens(tmp_path, monkeypatch, backend):
    T = _load(monkeypatch, SAE_POOL_BACKEND=backend)
    T._RAM_POOLS.clear()
    tok_dir, pool_dir = tmp_path / "tokens", tmp_path / "pool"
    n_special, n_total = _make_pool(T, tok_dir, pool_dir)
    assert n_special > 0
    dev = torch.device("cpu")
    # without exclusion: every row survives, special ids are present
    prov = T.RollingActivationProvider(pool_dir, dev, seed=0, queue_size=2, n_workers=1)
    b = prov.next_batch()
    assert b.shape == (4 * 8, 6)
    assert bool(((b[:, 0] == 2) | (b[:, 0] == 1) | (b[:, 0] == 0)).any())
    prov.close()
    # with exclusion: no special id appears in any row of any shard
    prov = T.RollingActivationProvider(pool_dir, dev, seed=0, queue_size=2, n_workers=1,
                                       tok_dir=tok_dir, exclude_ids=(2, 1, 0))
    seen_rows = 0
    for _ in range(3):
        b = prov.next_batch()
        assert b.dtype == torch.bfloat16 and b.shape[1] == 6
        assert not bool(((b[:, 0] == 2) | (b[:, 0] == 1) | (b[:, 0] == 0)).any())
        assert b.shape[0] < 4 * 8
        seen_rows += b.shape[0]
    assert seen_rows == n_total - n_special      # exactly the special positions were dropped
    assert prov.masked_rows >= n_special and prov.total_rows >= n_total
    prov.close()
    T._RAM_POOLS.clear()


def test_provider_exclude_none_ids_ignored(tmp_path, monkeypatch):
    T = _load(monkeypatch, SAE_POOL_BACKEND="ram")
    T._RAM_POOLS.clear()
    tok_dir, pool_dir = tmp_path / "tokens", tmp_path / "pool"
    _make_pool(T, tok_dir, pool_dir)
    prov = T.RollingActivationProvider(pool_dir, torch.device("cpu"), seed=0, queue_size=1,
                                       n_workers=1, tok_dir=tok_dir, exclude_ids=(None,))
    assert prov.exclude_ids == []
    assert prov.next_batch().shape[0] == 32          # no mask applied
    prov.close()
    T._RAM_POOLS.clear()


def test_accum_guard_and_single_run_wandb(tmp_path, monkeypatch):
    """exclusion + accum_steps>1 must assert before any training; the assert lives
    right after accum_steps is computed."""
    T = _load(monkeypatch, SAE_WANDB_SINGLE_RUN="1")
    assert T._WANDB_SINGLE_RUN is True
    src = (TRAINER / "sae_trainer_rolling.py").read_text()
    body = src.split("def train_sae_on_activations", 1)[1]
    assert body.index("accum_steps = BATCH_TOKENS // microbatch_tokens") < \
        body.index("SAE_EXCLUDE_SPECIAL requires microbatch_tokens == BATCH_TOKENS")
    # the two per-step log sites and the early-stop log go through _wlog
    train_body = body.split("def run_atlas_rolling", 1)[0]
    assert len(re.findall(r"\bwandb\.log\(", train_body)) == 2      # both inside _wlog only
    assert train_body.count("_wlog(") == 3                           # def + early-stop + per-LOG_EVERY site
    # per-layer finish is skipped in single-run mode
    assert "if not _single_run:" in train_body and "wandb.finish()" in train_body

    class FakeProv:
        exclude_ids = [2]

    with pytest.raises(AssertionError, match="SAE_EXCLUDE_SPECIAL"):
        T.train_sae_on_activations(0, 8, 0, FakeProv(), microbatch_tokens=T.BATCH_TOKENS // 2,
                                   push=False, cpu=True)


def test_orchestrator_env_gates_present():
    src = (TRAINER / "sae_trainer_rolling.py").read_text()
    orch = src.split("def run_atlas_rolling", 1)[1]
    for flag in ("SAE_NO_PREPRODUCE", "SAE_EAGER_POOL_CLEANUP", "SAE_NO_RESUME_POOL",
                 "SAE_EXCLUDE_SPECIAL"):
        assert f'os.environ.get("{flag}", "0") == "1"' in orch, flag
    assert "and not produce_only and not no_preproduce:" in orch
    assert "deleted pool L{old_L} (eager)" in orch
    assert "if L >= start_layer and not no_resume_pool:" in orch
    assert "tok_dir=tok_dir if exclude_ids else None" in orch
