#!/usr/bin/env python3
"""Check quality of generated seeds."""
import json
import re
import sys

INVALID_WORDS = {
    "what", "this", "that", "these", "those", "it", "they", "he", "she", "we", "you",
    "a", "an", "the"
}

def check_seeds(seed_file):
    """Check seed quality."""
    seeds = []
    with open(seed_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seeds.append(json.loads(line))
    
    print(f"Total seeds: {len(seeds)}\n")
    
    issues_found = {
        "malformed_markers": [],
        "invalid_entities": [],
        "cross_sentence": [],
        "duplicates": []
    }
    
    seen_pairs = set()
    
    for i, seed in enumerate(seeds, 1):
        text = seed["text"]
        label = seed["label"]
        
        # Extract entities
        e1_match = re.search(r'\[E1\](.*?)\[/E1\]', text)
        e2_match = re.search(r'\[E2\](.*?)\[/E2\]', text)
        
        e1_text = e1_match.group(1).strip().lower() if e1_match else ""
        e2_text = e2_match.group(1).strip().lower() if e2_match else ""
        
        # Check for issues
        issues = []
        
        # Malformed markers
        if "G[E1]" in text or re.search(r'\[E[12]\][^[]*\[E[12]\]', text):
            issues.append("malformed_marker")
            issues_found["malformed_markers"].append(i)
        
        # Invalid entities
        if e1_text in INVALID_WORDS:
            issues.append(f"invalid_e1: {e1_text}")
            issues_found["invalid_entities"].append(i)
        if e2_text in INVALID_WORDS:
            issues.append(f"invalid_e2: {e2_text}")
            issues_found["invalid_entities"].append(i)
        
        # Cross-sentence
        if "\n" in text and re.search(r'\[E1\].*\n.*\[E2\]', text):
            issues.append("cross_sentence")
            issues_found["cross_sentence"].append(i)
        
        # Duplicates
        pair_key = (e1_text, e2_text, label)
        if pair_key in seen_pairs:
            issues.append("duplicate")
            issues_found["duplicates"].append(i)
        seen_pairs.add(pair_key)
        
        status = "✓ OK" if not issues else f"✗ {', '.join(issues)}"
        print(f"{i}. [{label}] {status}")
        print(f"   E1: {e1_text[:40]}")
        print(f"   E2: {e2_text[:40]}")
        print(f"   Text: {text[:100]}...")
        print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print(f"  Total seeds: {len(seeds)}")
    print(f"  Malformed markers: {len(issues_found['malformed_markers'])}")
    print(f"  Invalid entities: {len(issues_found['invalid_entities'])}")
    print(f"  Cross-sentence: {len(issues_found['cross_sentence'])}")
    print(f"  Duplicates: {len(issues_found['duplicates'])}")
    
    if all(len(v) == 0 for v in issues_found.values()):
        print("\n✓ All seeds are valid!")
        return 0
    else:
        print("\n✗ Issues found")
        return 1

if __name__ == "__main__":
    seed_file = sys.argv[1] if len(sys.argv) > 1 else "training/seed.jsonl"
    sys.exit(check_seeds(seed_file))

