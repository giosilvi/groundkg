# tools/build_manifest.py
import json, sys, yaml
from pathlib import Path
META_P = Path("data/meta.jsonl")
OUT_P = Path("docs.yaml")

def main():
    if not META_P.exists():
        print("data/meta.jsonl not found", file=sys.stderr); sys.exit(2)
    docs, seen = [], set()
    for line in META_P.open("r", encoding="utf-8"):
        m = json.loads(line)
        if m["doc_id"] in seen: continue
        seen.add(m["doc_id"])
        docs.append({
            "doc_id": m["doc_id"],
            "title": m.get("title") or m["doc_id"],
            "url": m["url"],
            "license": m.get("license","UNKNOWN"),
            "sha256": m["sha256_raw"]
        })
    with OUT_P.open("w", encoding="utf-8") as f:
        yaml.safe_dump(docs, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {OUT_P} with {len(docs)} entries")

if __name__ == "__main__":
    main()

