import argparse
import json
from pathlib import Path


def _iter_events(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with open(outp, "w", encoding="utf-8") as w:
        for ev in _iter_events(args.events):
            evnode = f"event:{ev['event_id']}"
            score = float(ev.get("confidence", 0.5))
            src = ev.get("source")

            def emit(subj, pred, obj):
                w.write(
                    json.dumps(
                        {
                            "subject": subj,
                            "predicate": pred,
                            "object": obj,
                            "score": score,
                            "source": src,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            emit(evnode, "type", ev["type"])
            if ev.get("trigger"):
                emit(evnode, "trigger", ev["trigger"])
            if ev.get("date_text"):
                emit(evnode, "date", ev["date_text"])
            if ev.get("amount_text"):
                emit(evnode, "amount", ev["amount_text"])

            for role, val in (ev.get("roles") or {}).items():
                if val:
                    emit(evnode, role, val)

    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
