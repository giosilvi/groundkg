# tools/select_training_from_scored.py
import sys, json, os, random
random.seed(0)

POS_THR = float(os.environ.get("GK_POS_THR", "0.95"))
NEG_THR = float(os.environ.get("GK_NEG_THR", "0.95"))
MAX_PER_CLASS = int(os.environ.get("GK_MAX_PER_CLASS", "500"))
MIN_PER_CLASS = int(os.environ.get("GK_MIN_PER_CLASS", "20"))

def mark(text, s, o):
    s0, s1 = s["start"], s["end"]; o0, o1 = o["start"], o["end"]
    if s0 > o0: s0, s1, o0, o1 = o0, o1, s0, s1
    return text[:s0]+"[E1]"+text[s0:s1]+"[/E1]"+text[s1:o0]+"[E2]"+text[o0:o1]+"[/E2]"+text[o1:]

def main():
    inp = sys.argv[1] if len(sys.argv)>1 else "out/pack.scored.jsonl"
    pos, pos_all, neg = {}, {}, []
    with open(inp,"r",encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lbl, p = r["pred"], float(r["prob"])
            if lbl != "none" and p >= POS_THR:
                pos.setdefault(lbl, []).append({"text": mark(r["text"], r["subject"], r["object"]), "label": lbl})
            if lbl != "none":
                pos_all.setdefault(lbl, []).append({
                    "text": mark(r["text"], r["subject"], r["object"]),
                    "label": lbl,
                    "prob": p
                })
            elif lbl == "none" and p >= NEG_THR:
                neg.append({"text": mark(r["text"], r["subject"], r["object"]), "label": "none"})
    # cap per class and ensure a minimum per class if available
    out = []
    labels = set(pos_all.keys())
    for lbl in labels:
        # items above threshold
        items = pos.get(lbl, [])
        # if not enough, backfill with top-scoring examples regardless of threshold
        if len(items) < MIN_PER_CLASS:
            backfill = sorted(pos_all.get(lbl, []), key=lambda x: x["prob"], reverse=True)
            # drop prob
            backfill = [{"text": it["text"], "label": it["label"]} for it in backfill]
            need = MIN_PER_CLASS - len(items)
            items = items + backfill[:max(0, need)]
        random.shuffle(items)
        out.extend(items[:MAX_PER_CLASS])
    # If still underrepresented classes exist, use mined patterns to backfill from candidates
    patt_file = os.path.join("out", "patterns.jsonl")
    cand_file = os.path.join("out", "pack.candidates.jsonl")
    patt_map = {}
    if os.path.exists(patt_file):
        try:
            with open(patt_file, "r", encoding="utf-8") as pf:
                for line in pf:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    lbl = r.get("label") or r.get("suggested")
                    surf = r.get("surface", "")
                    if not lbl or lbl == "unknown" or not surf:
                        continue
                    patt_map.setdefault(lbl, set()).add(surf)
        except Exception as e:
            print(f"[warn] failed to read patterns: {e}")
    # backfill by matching patterns between E1/E2
    if patt_map and os.path.exists(cand_file):
        try:
            with open(cand_file, "r", encoding="utf-8") as cf:
                for line in cf:
                    if not line.strip():
                        continue
                    c = json.loads(line)
                    text = c["text"]; s=c["subject"]; o=c["object"]
                    s0,s1 = s["start"], s["end"]; o0,o1 = o["start"], o["end"]
                    lo, hi = (s1, o0) if s1 <= o0 else (o1, s0)
                    mid = text[lo:hi].strip().lower()
                    mid_clean = ' '.join([t for t in mid.replace('\n',' ').split() if t])
                    for lbl, surfaces in patt_map.items():
                        if len([r for r in out if r["label"]==lbl]) >= MIN_PER_CLASS:
                            continue
                        # exact substring match of mined surface phrase
                        if any(surf in mid_clean for surf in surfaces):
                            marked = mark(text, s, o)
                            out.append({"text": marked, "label": lbl})
        except Exception as e:
            print(f"[warn] pattern backfill failed: {e}")
    random.shuffle(neg); out.extend(neg[:MAX_PER_CLASS])
    random.shuffle(out)
    # append persistent seeds if present
    seed_path = os.path.join("training", "seed.jsonl")
    if os.path.exists(seed_path):
        try:
            with open(seed_path, "r", encoding="utf-8") as sf:
                for line in sf:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if "text" in r and "label" in r:
                        out.append({"text": r["text"], "label": r["label"]})
        except Exception as e:
            print(f"[warn] failed to read seeds: {e}")
    n = len(out); n_dev = max(1, int(0.2*n))
    dev, train = out[:n_dev], out[n_dev:]
    os.makedirs("training", exist_ok=True)
    with open("training/re_train.jsonl","w",encoding="utf-8") as f:
        for r in train: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    with open("training/re_dev.jsonl","w",encoding="utf-8") as f:
        for r in dev: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"Auto-selected {len(train)} train / {len(dev)} dev (POS_THR={POS_THR}, NEG_THR={NEG_THR})")

if __name__ == "__main__":
    main()

