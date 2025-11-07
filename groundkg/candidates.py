# groundkg/candidates.py
import sys
import json
import itertools
import re

SUBJ_LABELS = {"ORG", "PRODUCT", "PERSON", "FAC", "GPE", "EVENT", "LAW", "NORP"}
OBJ_LABELS = {
    "ORG",
    "PRODUCT",
    "PERSON",
    "FAC",
    "GPE",
    "EVENT",
    "LAW",
    "NORP",
    "LOC",
    "WORK_OF_ART",
}
MAX_CHAR_DIST = 150
MAX_PAIRS_PER_SENT = 10  # cap to reduce noise

NP_REGEX = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")


def non_overlapping_chunks(sent_text, ents):
    """ 
    Purpose: Extract noun phrases from a sentence that are not overlapping with any entity.
    """
    chunks = []
    for m in NP_REGEX.finditer(sent_text):
        start, end = m.start(), m.end()
        if end - start < 3:
            continue
        if any(not (end <= e["start"] or start >= e["end"]) for e in ents):
            continue
        chunks.append(
            {
                "text": sent_text[start:end],
                "start": start,
                "end": end,
                "label": "NOUNPHRASE",
            }
        )
    return chunks


def main():
    ner_path = sys.argv[1]
    with open(ner_path, "r", encoding="utf-8") as f:
        for line in f:
            sent = json.loads(line)
            ents = sent.get("entities", [])
            chunks = non_overlapping_chunks(sent["text"], ents)
            objs_pool = ents + chunks
            subs = [e for e in ents if e["label"] in SUBJ_LABELS] or ents or chunks
            objs = [
                e
                for e in objs_pool
                if (e["label"] in OBJ_LABELS or e["label"] == "NOUNPHRASE")
            ] or objs_pool

            # Build candidates with a simple proximity score (shorter span distance first)
            pairs = []
            seen_pairs = set()  # deduplicate by (subject text, object text)
            for s, o in itertools.product(subs, objs):
                if s is o:
                    continue
                # enforce subject before object to reduce symmetric noise
                if s["start"] >= o["start"]:
                    continue
                span_min = min(s["start"], o["start"])
                span_max = max(s["end"], o["end"])
                width = span_max - span_min
                if width > MAX_CHAR_DIST:
                    continue
                # deduplicate: same subject-object text pair
                pair_key = (s["text"].strip().lower(), o["text"].strip().lower())
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                prox = width  # lower is better
                pairs.append((prox, s, o))
            pairs.sort(key=lambda x: x[0])
            for _, s, o in pairs[:MAX_PAIRS_PER_SENT]:
                out = {
                    "doc_id": sent["doc_id"],
                    "sent_idx": sent["sent_idx"],
                    "sent_start": sent["sent_start"],
                    "text": sent["text"],
                    "subject": s,
                    "object": o,
                }
                sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
