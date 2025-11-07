import json
import os
import re
import sys


INP_DEFAULT = "out/pack.candidates.jsonl"
OUT_DEFAULT = "training/seed.jsonl"

# Invalid entity words to filter out
INVALID_ENTITY_WORDS = {
    # Pronouns
    "what",
    "this",
    "that",
    "these",
    "those",
    "it",
    "they",
    "he",
    "she",
    "we",
    "you",
    # Determiners
    "a",
    "an",
    "the",
    # Common stop words that shouldn't be entities
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
}


# Mid-surface patterns between E1/E2 (high precision, domain-agnostic)
MID = {
    "headquartered_in": re.compile(r"\bis\s+headquartered\s+in\b", re.I),
    "subsidiary_of": re.compile(r"\bis\s+(?:a\s+)?subsidiary\s+of\b", re.I),
    "member_of": re.compile(r"\bis\s+(?:a\s+)?member\s+of\b", re.I),
    "part_of": re.compile(r"\bis\s+part\s+of\b", re.I),
    "operates_in": re.compile(r"\boperates\s+in\b", re.I),
    "covered_by": re.compile(r"\bis\s+covered\s+by\b", re.I),
    "provides": re.compile(r"\bprovides\b", re.I),
    "requires": re.compile(r"\brequires\b", re.I),
    "prohibits": re.compile(r"\bprohibits\b", re.I),
    "uses": re.compile(r"\buses\b", re.I),
    # keep 'type' very conservative - require noun phrase after "is a/an/the"
    "type": re.compile(r"\bis\s+(?:an?|the)\s+\w+", re.I),
}


# Type guards (aligned with groundkg/re_infer.py ALLOWED_TYPES)
SUB_ALLOWED = {
    "headquartered_in": {"ORG", "PERSON", "FAC", "NORP"},
    "operates_in": {"ORG", "PRODUCT"},
    "subsidiary_of": {"ORG"},
    "member_of": {"ORG", "PERSON", "PRODUCT"},
    "part_of": {"ORG", "PRODUCT", "FAC", "WORK_OF_ART", "LOC", "GPE"},
    "covered_by": {
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
        "NOUNPHRASE",
    },
    "requires": {"LAW", "ORG"},
    "prohibits": {"LAW", "ORG"},
    "provides": {"ORG", "PRODUCT"},
    "uses": {"ORG", "PERSON", "PRODUCT"},
    "type": {
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
        "NOUNPHRASE",
    },
}

OBJ_ALLOWED = {
    "headquartered_in": {"GPE", "LOC", "FAC"},
    "operates_in": {"GPE", "LOC", "NORP"},
    "subsidiary_of": {"ORG"},
    "member_of": {"ORG"},
    "part_of": {"ORG", "PRODUCT", "FAC", "WORK_OF_ART", "LOC", "GPE"},
    "covered_by": {"LAW"},
    "requires": {"NOUNPHRASE", "WORK_OF_ART"},
    "prohibits": {"NOUNPHRASE", "WORK_OF_ART"},
    "provides": {"NOUNPHRASE", "PRODUCT", "WORK_OF_ART"},
    "uses": {"PRODUCT", "WORK_OF_ART", "LAW", "NOUNPHRASE"},
    "type": {"NOUNPHRASE", "ORG", "PRODUCT", "WORK_OF_ART"},
}


def type_ok(lbl: str, s_lab: str, o_lab: str) -> bool:
    return (s_lab in SUB_ALLOWED.get(lbl, set())) and (o_lab in OBJ_ALLOWED.get(lbl, set()))


def is_valid_entity(entity: dict) -> bool:
    """Check if entity text is valid (not a pronoun, determiner, etc.)."""
    text = entity.get("text", "").strip().lower()
    if not text:
        return False
    # Minimum length: 2 characters
    if len(text) < 2:
        return False
    # Exclude invalid words
    if text in INVALID_ENTITY_WORDS:
        return False
    # Exclude single characters (unless it's a valid acronym - but we'll be conservative)
    if len(text) == 1 and not text.isupper():
        return False
    return True


def validate_entities_in_sentence(text: str, s: dict, o: dict, sent_start: int) -> bool:
    """Verify that both entities are within the same sentence boundaries."""
    s0, s1 = s["start"], s["end"]
    o0, o1 = o["start"], o["end"]
    text_len = len(text)
    # Check that entity positions are valid relative to sentence text
    if s0 < 0 or s1 > text_len or o0 < 0 or o1 > text_len:
        return False
    if s0 >= s1 or o0 >= o1:
        return False
    # Ensure entities don't overlap
    if not (s1 <= o0 or o1 <= s0):
        return False
    # Validate entity text matches what's actually at those positions
    s_text_actual = text[s0:s1].strip()
    o_text_actual = text[o0:o1].strip()
    s_text_expected = s.get("text", "").strip()
    o_text_expected = o.get("text", "").strip()
    # Allow some flexibility (whitespace, case) but text should match
    if s_text_actual.lower() != s_text_expected.lower():
        return False
    if o_text_actual.lower() != o_text_expected.lower():
        return False
    # Check for sentence boundaries between entities (cross-sentence matching)
    # Find the span between entities
    lo, hi = (s1, o0) if s1 <= o0 else (o1, s0)
    if lo < hi:
        between_text = text[lo:hi]
        # Check for sentence-ending punctuation followed by whitespace/newline
        if re.search(r'[.!?]\s+', between_text):
            return False
        # Check for newlines (often indicate sentence boundaries)
        if '\n' in between_text:
            return False
    return True


def mark(text: str, s: dict, o: dict) -> str:
    """Mark entities in text with [E1]...[/E1] and [E2]...[/E2] tags."""
    s0, s1, o0, o1 = s["start"], s["end"], o["start"], o["end"]
    # Validate boundaries
    if s0 < 0 or s1 > len(text) or o0 < 0 or o1 > len(text):
        raise ValueError(f"Entity boundaries out of range: s=({s0},{s1}), o=({o0},{o1}), text_len={len(text)}")
    if s0 >= s1 or o0 >= o1:
        raise ValueError(f"Invalid entity boundaries: s=({s0},{s1}), o=({o0},{o1})")
    # Ensure E1 comes before E2
    if s0 > o0:
        s0, s1, o0, o1 = o0, o1, s0, s1
    # Check for overlap (shouldn't happen, but be safe)
    if s1 > o0:
        raise ValueError(f"Overlapping entities: s=({s0},{s1}), o=({o0},{o1})")
    return (
        text[:s0]
        + "[E1]"
        + text[s0:s1]
        + "[/E1]"
        + text[s1:o0]
        + "[E2]"
        + text[o0:o1]
        + "[/E2]"
        + text[o1:]
    )


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else INP_DEFAULT
    outp = sys.argv[2] if len(sys.argv) > 2 else OUT_DEFAULT
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)

    kept = 0
    neg_examples = []  # Collect negative examples for "none" class
    seen_per_sentence = {}  # Track seen pairs per (doc_id, sent_idx)
    with open(outp, "w", encoding="utf-8") as w:
        with open(inp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)
                text = c["text"]
                s, o = c["subject"], c["object"]
                
                # Filter invalid entities
                if not is_valid_entity(s) or not is_valid_entity(o):
                    continue
                
                # Validate entities are in sentence
                sent_start = c.get("sent_start", 0)
                if not validate_entities_in_sentence(text, s, o, sent_start):
                    continue
                
                s0, s1, o0, o1 = s["start"], s["end"], o["start"], o["end"]
                lo, hi = (s1, o0) if s1 <= o0 else (o1, s0)
                if lo < 0 or hi > len(text) or lo >= hi:
                    continue
                mid = " ".join(text[lo:hi].strip().split())

                # Sentence-level deduplication
                sent_key = (c["doc_id"], c["sent_idx"])
                pair_key = (s["text"].strip().lower(), o["text"].strip().lower())
                if sent_key not in seen_per_sentence:
                    seen_per_sentence[sent_key] = set()
                if pair_key in seen_per_sentence[sent_key]:
                    continue  # Skip duplicate pair in same sentence
                seen_per_sentence[sent_key].add(pair_key)

                matched = None
                matched_distance = float('inf')
                for lbl, rx in MID.items():
                    if not type_ok(lbl, s.get("label", ""), o.get("label", "")):
                        continue
                    if rx.search(mid):
                        # extra guard for 'type' to avoid overly broad matches
                        if lbl == "type":
                            # Require at least 3 words between entities for type
                            mid_words = len(mid.split())
                            if mid_words > 6 or mid_words < 3:
                                continue
                        # Prefer closer matches (shorter distance between entities)
                        distance = abs(s1 - o0) if s1 <= o0 else abs(o1 - s0)
                        if distance < matched_distance:
                            matched = lbl
                            matched_distance = distance
                
                # Mark entities with validation
                try:
                    marked_text = mark(text, s, o)
                except ValueError as e:
                    # Skip malformed entities
                    continue

                if matched:
                    # Positive example
                    w.write(
                        json.dumps({"text": marked_text, "label": matched}, ensure_ascii=False)
                        + "\n"
                    )
                    kept += 1
                else:
                    # Collect negative examples (no pattern matched)
                    # Only keep if entities are reasonably close (within 50 chars)
                    distance = abs(s1 - o0) if s1 <= o0 else abs(o1 - s0)
                    # Increase limit to get more negatives for better training
                    if distance <= 50 and len(neg_examples) < 20:  # Increased limit
                        neg_examples.append(marked_text)

    # Add negative examples to ensure at least 2 classes
    # Add more negatives to ensure balanced training (aim for ~40% negatives)
    if kept > 0 and len(neg_examples) > 0:
        # Ensure at least 3 negatives, up to 40% of positives
        num_neg = min(max(3, int(kept * 0.4)), len(neg_examples))
        with open(outp, "a", encoding="utf-8") as w:
            for neg_text in neg_examples[:num_neg]:
                w.write(
                    json.dumps({"text": neg_text, "label": "none"}, ensure_ascii=False)
                    + "\n"
                )
                kept += 1

    num_neg_added = min(len(neg_examples), max(3, int((kept - len(neg_examples)) * 0.4))) if kept > 0 and len(neg_examples) > 0 else 0
    print(f"Wrote {outp} with {kept} seeds (including {num_neg_added} negative examples)")


if __name__ == "__main__":
    main()



