# groundkg/dedupe_edges.py
import sys, json

def key(e):
    ev = e.get("evidence", {})
    return (
        e.get("subject","").strip().lower(),
        e.get("predicate","").strip(),
        e.get("object","").strip().lower(),
        ev.get("quote","").strip()
    )

def main():
    in_path = sys.argv[1]
    seen = set()
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            k = key(e)
            if k in seen:
                continue
            seen.add(k)
            sys.stdout.write(json.dumps(e, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

