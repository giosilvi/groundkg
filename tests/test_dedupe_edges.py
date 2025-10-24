import io
import json

from groundkg import dedupe_edges


def test_key_normalizes_fields():
    edge = {
        "subject": " Alice ",
        "predicate": "uses",
        "object": " Gadget ",
        "evidence": {"quote": "Alice uses the gadget."},
    }
    assert dedupe_edges.key(edge) == ("alice", "uses", "gadget", "Alice uses the gadget.")


def test_main_filters_duplicates(tmp_path, monkeypatch):
    edge = {
        "subject": "Alice",
        "predicate": "uses",
        "object": "Gadget",
        "evidence": {"quote": "Alice uses the gadget."},
    }
    dup_path = tmp_path / "edges.jsonl"
    dup_path.write_text("\n".join(json.dumps(e) for e in (edge, edge)) + "\n", encoding="utf-8")

    buf = io.StringIO()
    monkeypatch.setattr("sys.argv", ["dedupe_edges.py", str(dup_path)])
    monkeypatch.setattr("sys.stdout", buf)

    dedupe_edges.main()

    lines = buf.getvalue().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == edge


def test_key_handles_missing_evidence_quote():
    edge = {"subject": "Alice", "predicate": "uses", "object": "Gadget"}
    assert dedupe_edges.key(edge) == ("alice", "uses", "gadget", "")


def test_main_dedupes_whitespace_only_quotes(tmp_path, monkeypatch):
    edges = [
        {
            "subject": "Alice",
            "predicate": "uses",
            "object": "Gadget",
            "evidence": {"quote": "  Alice uses the gadget.  "},
        },
        {
            "subject": "alice ",
            "predicate": "uses",
            "object": "gadget",
            "evidence": {"quote": "Alice uses the gadget."},
        },
    ]
    dup_path = tmp_path / "edges.jsonl"
    dup_path.write_text("\n".join(json.dumps(e) for e in edges) + "\n", encoding="utf-8")

    buf = io.StringIO()
    monkeypatch.setattr("sys.argv", ["dedupe_edges.py", str(dup_path)])
    monkeypatch.setattr("sys.stdout", buf)

    dedupe_edges.main()

    lines = [json.loads(line) for line in buf.getvalue().splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["subject"].strip().lower() == "alice"
