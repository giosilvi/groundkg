# tools/mine_patterns.py
import sys, json, re, collections, argparse,os

# Simple tokenization
TOKEN_RE = re.compile(r"\w+|[^\w\s]")

STOP = set(['the','a','an','to','of','in','for','on','with','and','or','is','are','was','were','be','been','by','at','as','from','that','this','these','those'])

def tokens(s):
    return TOKEN_RE.findall(s)

def window_between(text, s0, s1, o0, o1, max_len=8):
    # Returns token window between subject and object spans
    lo, hi = (s1, o0) if s1 <= o0 else (o1, s0)
    mid = text[lo:hi]
    toks = [t.lower() for t in tokens(mid) if t.strip()]
    return toks[:max_len]

def key_normalize(toks):
    toks = [t for t in toks if t.isalpha() and t not in STOP]
    return ' '.join(toks)

def mine_from_candidates(cand_path, min_count=5):
    counts = collections.Counter()
    examples = {}
    with open(cand_path, 'r', encoding='utf-8') as f:
        for line in f:
            c = json.loads(line)
            text = c['text']; s=c['subject']; o=c['object']
            s0,s1 = s['start'], s['end']; o0,o1 = o['start'], o['end']
            toks = window_between(text, s0, s1, o0, o1)
            k = key_normalize(toks)
            if not k: 
                continue
            counts[k] += 1
            examples.setdefault(k, (text, s, o))
    patts = [(k, n, examples[k]) for k, n in counts.items() if n >= min_count]
    patts.sort(key=lambda x: -x[1])
    return patts

def mine_from_scored(scored_path, min_count=3, min_prob=0.9):
    counts = {}  # {(label, key): count}
    examples = {}
    with open(scored_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if float(r.get('prob',0)) < min_prob: 
                continue
            text = r['text']; s=r['subject']; o=r['object']; lbl = r.get('pred','')
            s0,s1 = s['start'], s['end']; o0,o1 = o['start'], o['end']
            toks = window_between(text, s0, s1, o0, o1)
            k = key_normalize(toks)
            if not k:
                continue
            key = (lbl, k)
            counts[key] = counts.get(key, 0) + 1
            examples.setdefault(key, (text, s, o))
    out = []
    for (lbl, k), n in counts.items():
        if n >= min_count:
            out.append((lbl, k, n, examples[(lbl,k)]))
    out.sort(key=lambda x: -x[2])
    return out


def suggest_predicate_id(surface):
    # crude mapping from surface phrase to predicate id suggestion
    m = surface
    if 'headquarter' in m or 'located in' in m or 'in ' == m[:3]:
        return 'headquartered_in'
    if 'consist' in m or 'compose' in m or 'made of' in m:
        return 'consists_of'
    if 'use' in m or 'used for' in m or 'for' == m:
        return 'performs_task'
    if m in ('is','are') or m.startswith('is '):
        return 'type'
    return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', help='out/pack.candidates.jsonl or out/candidates.jsonl')
    ap.add_argument('--scored', help='out/pack.scored.jsonl (preferred)')
    ap.add_argument('--min-count', type=int, default=5)
    ap.add_argument('--min-prob', type=float, default=0.9)
    ap.add_argument('--json', action='store_true', help='emit JSONL instead of TSV')
    args = ap.parse_args()

    if args.scored and os.path.exists(args.scored):
        rows = mine_from_scored(args.scored, min_count=args.min_count, min_prob=args.min_prob)
        if args.json:
            for lbl, surf, n, ex in rows:
                rec = {"label": lbl, "surface": surf, "count": n}
                print(json.dumps(rec, ensure_ascii=False))
        else:
            print('# label\tsurface_between\tcount\texample')
            for lbl, surf, n, ex in rows:
                text, s, o = ex
                print(f"{lbl}\t{surf}\t{n}\t{text}")
    elif args.candidates and os.path.exists(args.candidates):
        rows = mine_from_candidates(args.candidates, min_count=args.min_count)
        if args.json:
            for surf, n, ex in rows:
                rec = {"surface": surf, "count": n, "suggested": suggest_predicate_id(surf)}
                print(json.dumps(rec, ensure_ascii=False))
        else:
            print('# surface_between\tcount\texample\tsuggested_predicate')
            for surf, n, ex in rows:
                text, s, o = ex
                print(f"{surf}\t{n}\t{text}\t{suggest_predicate_id(surf)}")
    else:
        print('Provide --scored or --candidates input', file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    main()
