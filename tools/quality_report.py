import json, sys, os, collections, statistics

def safe_load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def main():
    # Inputs
    scored_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join('out','pack.scored.jsonl')
    edges_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join('out','edges.jsonl')
    train_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join('training','re_train.jsonl')
    thr_path = sys.argv[4] if len(sys.argv) > 4 else os.path.join('models','thresholds.json')

    scored = safe_load_jsonl(scored_path)
    edges = safe_load_jsonl(edges_path)
    train = safe_load_jsonl(train_path)
    thresholds = {}
    if os.path.exists(thr_path):
        try:
            thresholds = json.load(open(thr_path,'r',encoding='utf-8'))
        except Exception:
            thresholds = {}

    # Scored stats
    pred_counts = collections.Counter()
    pred_probs = collections.defaultdict(list)
    for r in scored:
        lbl = r.get('pred','none')
        p = float(r.get('prob',0))
        pred_counts[lbl] += 1
        pred_probs[lbl].append(p)

    # Edge stats
    edge_counts = collections.Counter(e.get('predicate','') for e in edges)
    total_edges = len(edges)

    # Train label distribution
    train_counts = collections.Counter(r.get('label','') for r in train)

    # Print report
    print('=== Quality Report ===')
    print('- Scored predictions per class:')
    for lbl, cnt in pred_counts.most_common():
        probs = pred_probs.get(lbl, [])
        avg = statistics.mean(probs) if probs else 0.0
        print(f"  {lbl:16s} count={cnt:6d} avg_prob={avg:.3f} thr={thresholds.get(lbl,'-')}")
    print('- Edges emitted (post-threshold):')
    print(f"  total_edges={total_edges}")
    for lbl, cnt in edge_counts.most_common():
        print(f"  {lbl:16s} count={cnt}")
    non_none = sum(cnt for lbl,cnt in edge_counts.items() if lbl and lbl!='none')
    print(f"  non_none_ratio={(non_none/(total_edges or 1)):.3f}")
    print('- Training label distribution:')
    for lbl, cnt in train_counts.most_common():
        print(f"  {lbl:16s} count={cnt}")
    print('- Thresholds:')
    for lbl, thr in thresholds.items():
        print(f"  {lbl:16s} thr={thr}")

if __name__ == '__main__':
    main()


