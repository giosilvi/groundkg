import argparse
import json
import re
import uuid
from pathlib import Path

# Very lightweight sentence split (keeps runtime/dep minimal)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Naive org and money/date patterns (good enough for bootstrapping)
ORG = r"([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*)"
MONEY = r"([$€£]\s?\d[\d.,\s]*(?:\s?(?:billion|million|bn|m|B|M))?)"
DATE = r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}|\d{4})"

PATTERNS = [
    (
        "Acquisition",
        re.compile(
            rf"\b{ORG}\s+(?:acquired|bought|purchased)\s+{ORG}(?:\s+for\s+{MONEY})?(?:\s+on\s+{DATE})?",
            re.I,
        ),
    ),
    (
        "Funding",
        re.compile(
            rf"\b{ORG}\s+(?:raised|secured)\s+{MONEY}(?:\s+(Series\s?[A-K]))?(?:\s+from\s+{ORG})?(?:\s+on\s+{DATE})?",
            re.I,
        ),
    ),
    (
        "Appointment",
        re.compile(
            rf"\b{ORG}\s+(?:appointed|named|hired)\s+{ORG}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s+as\s+[A-Z][A-Za-z]+)?(?:\s+on\s+{DATE})?",
            re.I,
        ),
    ),
    (
        "Launch",
        re.compile(
            rf"\b{ORG}\s+(?:launched|released|unveiled)\s+(.+?)(?:\s+on\s+{DATE})?(?=[\.!\?]|$)",
            re.I,
        ),
    ),
    (
        "Founding",
        re.compile(
            rf"\b(?:{ORG}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:founded|established)\s+{ORG}(?:\s+in\s+{DATE})?",
            re.I,
        ),
    ),
]


def _read_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _sentences(text: str):
    return _SENT_SPLIT.split(text.strip()) if text else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with open(outp, "w", encoding="utf-8") as w:
        for rec in _read_manifest(args.manifest):
            doc_id = rec.get("doc_id") or rec.get("id") or rec.get("url")
            for s in _sentences(rec.get("text", "")):
                s_clean = s.strip()
                if not s_clean:
                    continue
                for ev_type, rx in PATTERNS:
                    m = rx.search(s_clean)
                    if not m:
                        continue

                    ev_id = "E_" + uuid.uuid4().hex[:12]
                    roles = {}
                    trigger = None

                    # Heuristic role mapping per event type
                    if ev_type == "Acquisition":
                        acquirer, target = m.group(1), m.group(2)
                        roles["acquirer"] = acquirer.strip()
                        roles["target"] = target.strip()
                        trigger = "acquired"
                    elif ev_type == "Funding":
                        org = m.group(1)
                        roles["recipient"] = org.strip()
                        trigger = "raised"
                    elif ev_type == "Appointment":
                        # Simplified: first ORG-like as org, second proper noun chunk as person
                        roles["actor"] = rec.get("source_org") or ""
                        trigger = "appointed"
                    elif ev_type == "Launch":
                        org = m.group(1)
                        roles["actor"] = org.strip()
                        trigger = "launched"
                    elif ev_type == "Founding":
                        founder_or_org, new_org = m.group(1), m.group(2)
                        roles["founder_or_actor"] = founder_or_org.strip()
                        roles["entity"] = new_org.strip()
                        trigger = "founded"

                    # Optional captures for amount/date
                    amount_text = None
                    date_text = None

                    # scan sentence for MONEY/DATE regardless of grouping
                    m_money = re.search(MONEY, s_clean, re.I)
                    if m_money:
                        amount_text = m_money.group(0)
                    m_date = re.search(DATE, s_clean, re.I)
                    if m_date:
                        date_text = m_date.group(0)

                    event = {
                        "event_id": ev_id,
                        "type": ev_type,
                        "doc_id": doc_id,
                        "trigger": trigger,
                        "roles": roles,
                        "date_text": date_text,
                        "amount_text": amount_text,
                        "confidence": 0.65,  # conservative prior
                        "source": f"{doc_id}#s",
                    }
                    w.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
