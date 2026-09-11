"""Clean the Bella corpus, build matched pairs, splits and canary sets.

Outputs (all small JSON/JSONL, committed under results/data/):
  pairs.jsonl        600 rows: {pid, prompt, bella, split in {train,val,test}, crisis}
  eval_prompts.jsonl 100 held-out corpus prompts for the steering sweep (disjoint from pairs)
  crisis_eval.jsonl  30 held-out crisis prompts (disjoint from pairs and eval_prompts)
  red_team.jsonl     160 red-team stems
  neutral.jsonl      100 neutral stems
  authentic.jsonl / corporate.jsonl  secondary contrast (copied, ids added)
  manifest.json      row counts at every cleaning step, seeds, regexes
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw"
OUT = HERE / "results" / "data"
SEED = 42

AI_DISCLAIMER = re.compile(
    r"(as an ai\b|language model|as an assistant|as a helpful assistant|"
    r"i'?m an ai\b|i am an ai\b|ai assistant|artificial intelligence|just an ai\b|only an ai\b)",
    re.I,
)
TEA = re.compile(r"\btea\b", re.I)
CRISIS = re.compile(
    r"(suicid\w*|kill myself|killing myself|end it all|end my life|ending my life|self[- ]harm|"
    r"hurt myself|hurting myself|cut myself|cutting myself|don'?t want to (be alive|live|wake up|exist)|"
    r"want to die|wanna die|overdos\w*|no reason to live|not worth living|"
    r"take my own life|unalive|disappear forever|better off dead)",
    re.I,
)
# The corpus has only 18 unique strict-crisis prompts after cleaning, so the crisis pool is
# strict crisis plus short first-person emotional-distress prompts (lonely, scared, crying,
# hopeless, depressed, relapsed ...). Strict rows are preferred for the held-out crisis set.
DISTRESS = re.compile(
    r"(lonely|so alone|being alone|scared|crying|can'?t (do this|go on|take (it|this)) anymore|falling apart|"
    r"give up|hopeless|worthless|hate myself|nobody cares|no one cares|depress|panic|anxious|anxiety|"
    r"breaking down|numb|empty inside|relapse)",
    re.I,
)
FIRST_PERSON = re.compile(r"^(i|i'm|im|i’m|i've|everything|nothing|my)\b", re.I)
DISTRESS_EXCLUDE = re.compile(r"(tattoo|spaghetti|parameters|career|bar life|clingy|chaos|cat\b|learn something new|jealous)", re.I)


def is_crisis(prompt: str) -> tuple[bool, bool]:
    """(in crisis pool, strict)"""
    if CRISIS.search(prompt) and len(prompt) < 220:
        return True, True
    if DISTRESS.search(prompt) and FIRST_PERSON.search(prompt) and len(prompt) < 160 and not DISTRESS_EXCLUDE.search(prompt):
        return True, False
    return False, False


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.open() if l.strip()]


def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def h(s: str) -> str:
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()


def main():
    rng = random.Random(SEED)
    manifest = {"seed": SEED, "steps": [], "regex": {
        "ai_disclaimer": AI_DISCLAIMER.pattern, "tea": TEA.pattern, "crisis": CRISIS.pattern}}
    rows = read_jsonl(RAW / "bella-corpus.jsonl")
    manifest["steps"].append({"step": "raw", "rows": len(rows)})

    # first user / first assistant turn (15 rows are multi-turn; we keep their first exchange)
    ex = []
    multi = 0
    for r in rows:
        m = r["messages"]
        if len(m) > 2:
            multi += 1
        u = next(x["content"] for x in m if x["role"] == "user")
        a = next(x["content"] for x in m if x["role"] == "assistant")
        ex.append({"prompt": u.strip(), "bella": a.strip()})
    manifest["steps"].append({"step": "first_exchange", "rows": len(ex), "multi_turn_rows": multi})

    n0 = len(ex)
    ex = [e for e in ex if not AI_DISCLAIMER.search(e["bella"])]
    manifest["steps"].append({"step": "drop_ai_disclaimer", "dropped": n0 - len(ex), "rows": len(ex)})
    n0 = len(ex)
    ex = [e for e in ex if not (TEA.search(e["bella"]) or TEA.search(e["prompt"]))]
    manifest["steps"].append({"step": "drop_tea", "dropped": n0 - len(ex), "rows": len(ex)})
    n0 = len(ex)
    seen = set()
    ex2 = []
    for e in ex:
        if e["bella"] in seen:
            continue
        seen.add(e["bella"])
        ex2.append(e)
    ex = ex2
    manifest["steps"].append({"step": "drop_duplicate_assistant", "dropped": n0 - len(ex), "rows": len(ex)})
    n0 = len(ex)
    seen = set()
    ex2 = []
    for e in ex:
        k = e["prompt"].lower()
        if k in seen:
            continue
        seen.add(k)
        ex2.append(e)
    ex = ex2
    manifest["steps"].append({"step": "drop_duplicate_prompt_keep_first", "dropped": n0 - len(ex), "rows": len(ex)})
    # drop empty / trivially short
    n0 = len(ex)
    ex = [e for e in ex if len(e["prompt"]) >= 3 and len(e["bella"]) >= 3]
    manifest["steps"].append({"step": "drop_empty", "dropped": n0 - len(ex), "rows": len(ex)})

    for e in ex:
        e["crisis"], e["crisis_strict"] = is_crisis(e["prompt"])
        e["pid"] = h(e["prompt"])
    crisis = [e for e in ex if e["crisis"]]
    normal = [e for e in ex if not e["crisis"]]
    manifest["cleaned"] = {"rows": len(ex), "crisis_pool": len(crisis),
                           "crisis_strict": sum(e["crisis_strict"] for e in crisis), "non_crisis": len(normal)}
    rng.shuffle(crisis)
    rng.shuffle(normal)
    # held-out crisis set first (30, strict rows preferred), remaining crisis rows go into the pairs
    crisis.sort(key=lambda e: not e["crisis_strict"])
    crisis_eval = crisis[:30]
    crisis_rest = crisis[30:]
    rng.shuffle(crisis_rest)

    # matched pairs: 600 prompts, at most 40 crisis
    n_crisis_pairs = min(40, len(crisis_rest))
    pairs = crisis_rest[:n_crisis_pairs] + normal[: 600 - n_crisis_pairs]
    rng.shuffle(pairs)
    # splits by prompt: 400 / 100 / 100, stratified on crisis
    for grp in (True, False):
        g = [p for p in pairs if p["crisis"] == grp]
        n = len(g)
        n_train = round(n * 400 / 600)
        n_val = round(n * 100 / 600)
        for i, p in enumerate(g):
            p["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    # steering eval prompts: 100 held-out non-crisis, disjoint; 30 held-out crisis
    used = {p["pid"] for p in pairs}
    eval_prompts = [e for e in normal[600 - n_crisis_pairs:] if e["pid"] not in used][:100]
    assert len(eval_prompts) == 100, len(eval_prompts)
    assert len(crisis_eval) == 30, (len(crisis_eval), len(crisis))
    assert not ({e["pid"] for e in eval_prompts} & used)
    assert not ({e["pid"] for e in crisis_eval} & used)
    assert not ({e["pid"] for e in crisis_eval} & {e["pid"] for e in eval_prompts})

    red = [{"pid": f"rt{i:03d}", "prompt": (r.get("text") or r.get("prompt")).strip()} for i, r in enumerate(read_jsonl(RAW / "red_team_stems.jsonl"))]
    neutral_all = [{"pid": f"ns{i:03d}", "text": r["text"].strip()} for i, r in enumerate(read_jsonl(RAW / "neutral_stems.jsonl"))]
    rng.shuffle(neutral_all)
    neutral = sorted(neutral_all[:100], key=lambda r: r["pid"])
    authentic = [{"id": f"au{i:03d}", "text": r["text"].strip()} for i, r in enumerate(read_jsonl(RAW / "authentic.jsonl"))]
    corporate = [{"id": f"co{i:03d}", "text": r["text"].strip()} for i, r in enumerate(read_jsonl(RAW / "corporate.jsonl"))]

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "pairs.jsonl", pairs)
    write_jsonl(OUT / "eval_prompts.jsonl", eval_prompts)
    write_jsonl(OUT / "crisis_eval.jsonl", crisis_eval)
    write_jsonl(OUT / "red_team.jsonl", red)
    write_jsonl(OUT / "neutral.jsonl", neutral)
    write_jsonl(OUT / "authentic.jsonl", authentic)
    write_jsonl(OUT / "corporate.jsonl", corporate)
    from collections import Counter
    manifest["pairs"] = {
        "n": len(pairs), "crisis": sum(p["crisis"] for p in pairs),
        "splits": dict(Counter(p["split"] for p in pairs)),
        "crisis_by_split": {f"{s}_{c}": n for (s, c), n in Counter((p["split"], p["crisis"]) for p in pairs).items()},
    }
    manifest["canaries"] = {"eval_prompts": len(eval_prompts), "crisis_eval": len(crisis_eval),
                            "crisis_eval_strict": sum(e["crisis_strict"] for e in crisis_eval),
                            "red_team": len(red), "neutral": len(neutral), "neutral_pool": len(neutral_all),
                            "authentic": len(authentic), "corporate": len(corporate)}
    import statistics
    manifest["length_words"] = {
        "bella_reply_median": statistics.median(len(p["bella"].split()) for p in pairs),
        "prompt_median": statistics.median(len(p["prompt"].split()) for p in pairs),
        "authentic_median": statistics.median(len(a["text"].split()) for a in authentic),
        "corporate_median": statistics.median(len(c["text"].split()) for c in corporate),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
