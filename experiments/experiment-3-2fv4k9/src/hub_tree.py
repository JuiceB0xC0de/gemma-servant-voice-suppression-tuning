"""Print which layer_NN_s0 folders (with both sae.pt and meta.json) exist on the Hub dataset."""
import re
import sys

from huggingface_hub import HfApi

REPO = "juiceb0xc0de/gemma-4-e4b-SAE"
files = HfApi().list_repo_files(REPO, repo_type="dataset")
have = {}
for f in files:
    m = re.match(r"(layer_(\d\d)_s0)/(sae\.pt|meta\.json)$", f)
    if m:
        have.setdefault(int(m.group(2)), set()).add(m.group(3))
complete = sorted(n for n, s in have.items() if s == {"sae.pt", "meta.json"})
partial = sorted(n for n, s in have.items() if s != {"sae.pt", "meta.json"})
missing = [n for n in range(42) if n not in complete]
print("complete:", complete)
print("partial:", partial)
print("missing:", missing)
print("summary_csv:", "atlas_summary.csv" in files)
print(f"{len(complete)}/42")
if "--json" in sys.argv:
    import json
    print(json.dumps({"complete": complete, "partial": partial, "missing": missing}))
