import sys, json

def main():
    scored_path, thr_path = sys.argv[1:3]
    thresholds = json.load(open(thr_path, 'r', encoding='utf-8'))
    with open(scored_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            pred = r.get('pred', 'none')
            prob = float(r.get('prob', 0.0))
            thr = float(thresholds.get(pred, 0.85))
            if pred == 'none' or prob < thr:
                continue
            s = r['subject']; o = r['object']
            edge = {
                'subject': s.get('text','').strip(),
                'predicate': pred,
                'object': o.get('text','').strip(),
                'evidence': {
                    'doc_id': r.get('doc_id'),
                    'quote': r.get('text',''),
                    'char_start': r.get('sent_start', 0),
                    'char_end': r.get('sent_start', 0) + len(r.get('text',''))
                }
            }
            sys.stdout.write(json.dumps(edge, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()


