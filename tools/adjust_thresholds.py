#!/usr/bin/env python3
"""Automatically adjust thresholds if no edges are emitted."""
import json
import sys
import os
from collections import defaultdict

def analyze_predictions(scored_path):
    """Analyze prediction distribution."""
    pred_probs = defaultdict(list)
    with open(scored_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pred = r.get('pred', 'none')
            prob = float(r.get('prob', 0.0))
            if pred != 'none':
                pred_probs[pred].append(prob)
    
    stats = {}
    for pred, probs in pred_probs.items():
        if not probs:
            continue
        probs_sorted = sorted(probs, reverse=True)
        stats[pred] = {
            'mean': sum(probs) / len(probs),
            'median': probs_sorted[len(probs_sorted) // 2],
            'p75': probs_sorted[len(probs_sorted) * 3 // 4] if len(probs_sorted) > 3 else probs_sorted[-1],
            'p90': probs_sorted[len(probs_sorted) * 9 // 10] if len(probs_sorted) > 9 else probs_sorted[-1],
            'max': probs_sorted[0],
            'count': len(probs)
        }
    return stats

def adjust_thresholds(thresholds_path, scored_path, min_edges=10, min_threshold=0.60):
    """Adjust thresholds to ensure at least min_edges are emitted.
    
    Args:
        thresholds_path: Path to thresholds.json
        scored_path: Path to scored predictions
        min_edges: Minimum number of edges to emit
        min_threshold: Minimum threshold floor (default 0.60)
    """
    # Load current thresholds
    if os.path.exists(thresholds_path):
        thresholds = json.load(open(thresholds_path, 'r', encoding='utf-8'))
    else:
        thresholds = {}
    
    # Count edges that would be emitted with current thresholds
    edges_count = defaultdict(int)
    with open(scored_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pred = r.get('pred', 'none')
            prob = float(r.get('prob', 0.0))
            thr = float(thresholds.get(pred, 0.85))
            if pred != 'none' and prob >= thr:
                edges_count[pred] += 1
    
    total_edges = sum(edges_count.values())
    
    # Analyze prediction distribution
    stats = analyze_predictions(scored_path)
    
    # Adjust thresholds to ensure we get some edges AND enforce minimum floor
    adjusted = thresholds.copy()
    adjusted_any = False
    
    # First, enforce minimum floor on all existing thresholds
    for pred in adjusted.keys():
        if pred != 'none':
            current_thr = float(adjusted.get(pred, 0.85))
            if current_thr < min_threshold:
                adjusted[pred] = min_threshold
                adjusted_any = True
    
    # If we have enough edges with current thresholds (after floor enforcement), return
    if total_edges >= min_edges:
        return adjusted, adjusted_any
    
    # Otherwise, lower thresholds if needed (but still enforce minimum floor)
    for pred, stat in stats.items():
        current_thr = float(adjusted.get(pred, 0.85))
        
        # If current threshold is too high, lower it
        if current_thr > stat['p90']:
            # Set to p90 (top 10% of predictions), but enforce minimum floor
            new_thr = max(min_threshold, min(stat['p90'], current_thr * 0.9))
            if new_thr != current_thr:
                adjusted[pred] = round(new_thr, 3)
                adjusted_any = True
        elif current_thr > stat['median'] and edges_count.get(pred, 0) == 0:
            # If no edges for this class, lower to median, but enforce minimum floor
            new_thr = max(min_threshold, min(stat['median'], current_thr * 0.85))
            if new_thr != current_thr:
                adjusted[pred] = round(new_thr, 3)
                adjusted_any = True
    
    # Ensure "none" threshold is high (we don't want to emit "none" edges)
    adjusted['none'] = 1.0
    
    return adjusted, adjusted_any

def main():
    scored_path = sys.argv[1] if len(sys.argv) > 1 else "out/pack.scored.jsonl"
    thresholds_path = sys.argv[2] if len(sys.argv) > 2 else "models/thresholds.json"
    min_edges = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    min_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.60
    
    if not os.path.exists(scored_path):
        print(f"Error: {scored_path} not found", file=sys.stderr)
        sys.exit(1)
    
    adjusted_thresholds, was_adjusted = adjust_thresholds(thresholds_path, scored_path, min_edges, min_threshold)
    
    if was_adjusted:
        os.makedirs(os.path.dirname(thresholds_path) or ".", exist_ok=True)
        with open(thresholds_path, 'w', encoding='utf-8') as f:
            json.dump(adjusted_thresholds, f, indent=2, ensure_ascii=False)
        print(f"Adjusted thresholds in {thresholds_path} (min_threshold={min_threshold})")
        print("New thresholds:")
        for pred, thr in sorted(adjusted_thresholds.items()):
            print(f"  {pred}: {thr}")
    else:
        print("Thresholds are appropriate, no adjustment needed")

if __name__ == "__main__":
    main()

