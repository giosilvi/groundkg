# tools/select_training_from_scored.py
import sys, json, os, random
random.seed(0)

POS_THR = float(os.environ.get("GK_POS_THR", "0.95"))
NEG_THR = float(os.environ.get("GK_NEG_THR", "0.95"))
MAX_PER_CLASS = int(os.environ.get("GK_MAX_PER_CLASS", "500"))
MIN_PER_CLASS = int(os.environ.get("GK_MIN_PER_CLASS", "20"))

def adaptive_thresholds(inp_path, initial_pos_thr, initial_neg_thr, min_examples=30, min_threshold=0.60):
    """Adaptively lower thresholds if too few examples are selected.
    
    Args:
        inp_path: Path to scored predictions
        initial_pos_thr: Initial positive threshold
        initial_neg_thr: Initial negative threshold
        min_examples: Minimum examples needed
        min_threshold: Minimum threshold floor (default 0.60)
    """
    # First pass: count examples with initial thresholds
    pos_count = {}
    pos_all = {}
    neg_count = 0
    
    with open(inp_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            lbl, p = r["pred"], float(r["prob"])
            if lbl != "none" and p >= initial_pos_thr:
                pos_count[lbl] = pos_count.get(lbl, 0) + 1
            if lbl != "none":
                pos_all.setdefault(lbl, []).append(p)
            elif lbl == "none" and p >= initial_neg_thr:
                neg_count += 1
    
    total_pos = sum(pos_count.values())
    
    # If we have enough examples, use original thresholds
    if total_pos + neg_count >= min_examples:
        return initial_pos_thr, initial_neg_thr
    
    # Otherwise, adaptively lower thresholds
    # Calculate percentiles for each class
    pos_thr = initial_pos_thr
    for lbl, probs in pos_all.items():
        if len(probs) > 0:
            probs_sorted = sorted(probs, reverse=True)
            # Use p75 if we don't have enough high-confidence examples
            if pos_count.get(lbl, 0) < MIN_PER_CLASS and len(probs_sorted) > 3:
                p75 = probs_sorted[len(probs_sorted) * 3 // 4]
                # Lower threshold to p75, but enforce minimum floor
                pos_thr = min(pos_thr, max(min_threshold, p75))
    
    # Lower neg threshold if needed, but enforce minimum floor
    neg_thr = initial_neg_thr
    if neg_count < MIN_PER_CLASS:
        neg_thr = max(min_threshold + 0.10, initial_neg_thr * 0.9)  # Neg threshold slightly higher
    
    return pos_thr, neg_thr

def mark(text, s, o):
    s0, s1 = s["start"], s["end"]; o0, o1 = o["start"], o["end"]
    if s0 > o0: s0, s1, o0, o1 = o0, o1, s0, s1
    return text[:s0]+"[E1]"+text[s0:s1]+"[/E1]"+text[s1:o0]+"[E2]"+text[o0:o1]+"[/E2]"+text[o1:]

def main():
    inp = sys.argv[1] if len(sys.argv)>1 else "out/pack.scored.jsonl"
    
    # Adaptively adjust thresholds if needed (with minimum floor of 0.60)
    MIN_THRESHOLD = float(os.environ.get("GK_MIN_THRESHOLD", "0.60"))
    pos_thr, neg_thr = adaptive_thresholds(inp, POS_THR, NEG_THR, min_examples=30, min_threshold=MIN_THRESHOLD)
    if pos_thr != POS_THR or neg_thr != NEG_THR:
        print(f"Adaptive thresholds: POS_THR={pos_thr:.3f} (was {POS_THR:.3f}), NEG_THR={neg_thr:.3f} (was {NEG_THR:.3f}) [min={MIN_THRESHOLD:.2f}]")
    
    pos, pos_all, neg = {}, {}, []
    with open(inp,"r",encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lbl, p = r["pred"], float(r["prob"])
            if lbl != "none" and p >= pos_thr:
                pos.setdefault(lbl, []).append({"text": mark(r["text"], r["subject"], r["object"]), "label": lbl})
            if lbl != "none":
                pos_all.setdefault(lbl, []).append({
                    "text": mark(r["text"], r["subject"], r["object"]),
                    "label": lbl,
                    "prob": p
                })
            elif lbl == "none" and p >= neg_thr:
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
    print(f"Auto-selected {len(train)} train / {len(dev)} dev (POS_THR={pos_thr:.3f}, NEG_THR={neg_thr:.3f})")

if __name__ == "__main__":
    main()

