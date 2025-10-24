import io
import json

from groundkg import candidates


def test_non_overlapping_chunks_filters_overlaps():
    sent = "Alice and Bob met Charlie"
    ents = [
        {"text": "Alice", "start": 0, "end": 5, "label": "PERSON"},
        {"text": "Charlie", "start": 18, "end": 25, "label": "PERSON"},
    ]

    chunks = candidates.non_overlapping_chunks(sent, ents)

    # "Alice" and "Charlie" should be excluded because they overlap entities
    chunk_texts = {c["text"] for c in chunks}
    assert chunk_texts == {"Bob"}
    # ensure chunks carry the expected metadata
    for chunk in chunks:
        assert chunk["label"] == "NOUNPHRASE"
        assert chunk["end"] - chunk["start"] >= 3


def test_main_emits_subject_object_pairs(tmp_path, monkeypatch):
    record = {
        "doc_id": "d1",
        "sent_idx": 0,
        "sent_start": 0,
        "text": "Alice visited Paris with Charlie",
        "entities": [
            {"text": "Alice", "start": 0, "end": 5, "label": "PERSON"},
            {"text": "Paris", "start": 13, "end": 18, "label": "GPE"},
        ],
    }
    ner_path = tmp_path / "ner.jsonl"
    ner_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    monkeypatch.setenv("PYTHONHASHSEED", "0")  # ensure deterministic iteration if needed
    monkeypatch.setattr("sys.argv", ["candidates.py", str(ner_path)])

    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)

    candidates.main()

    lines = [json.loads(line) for line in buf.getvalue().splitlines() if line]
    assert any(
        rec["subject"]["text"] == "Alice" and rec["object"]["text"] == "Paris"
        for rec in lines
    )
    for rec in lines:
        assert rec["doc_id"] == "d1"


def test_main_respects_char_distance_limit(tmp_path, monkeypatch):
    long_text = "Alice" + " " * 151 + "Paris"
    record = {
        "doc_id": "d1",
        "sent_idx": 0,
        "sent_start": 0,
        "text": long_text,
        "entities": [
            {"text": "Alice", "start": 0, "end": 5, "label": "PERSON"},
            {"text": "Paris", "start": 156, "end": 161, "label": "GPE"},
        ],
    }
    ner_path = tmp_path / "ner.jsonl"
    ner_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    monkeypatch.setattr("sys.argv", ["candidates.py", str(ner_path)])

    candidates.main()

    assert buf.getvalue().strip() == ""


def test_main_caps_pairs_at_limit(tmp_path, monkeypatch):
    tokens = ["Alice"] + [f"Obj{i}" for i in range(12)]
    text = " ".join(tokens)

    ents = []
    cursor = 0
    for token in tokens:
        start = cursor
        end = start + len(token)
        label = "PERSON" if token == "Alice" else "PRODUCT"
        ents.append({"text": token, "start": start, "end": end, "label": label})
        cursor = end + 1  # account for spaces

    record = {
        "doc_id": "d1",
        "sent_idx": 0,
        "sent_start": 0,
        "text": text,
        "entities": ents,
    }
    ner_path = tmp_path / "ner.jsonl"
    ner_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    monkeypatch.setattr("sys.argv", ["candidates.py", str(ner_path)])

    candidates.main()

    lines = [line for line in buf.getvalue().splitlines() if line]
    assert len(lines) == candidates.MAX_PAIRS_PER_SENT


def test_main_falls_back_to_chunks_without_entities(tmp_path, monkeypatch):
    record = {
        "doc_id": "d2",
        "sent_idx": 0,
        "sent_start": 0,
        "text": "Solar Panel helps Bright Homes",
        "entities": [],
    }
    ner_path = tmp_path / "ner.jsonl"
    ner_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    monkeypatch.setattr("sys.argv", ["candidates.py", str(ner_path)])

    candidates.main()

    lines = [json.loads(line) for line in buf.getvalue().splitlines() if line]
    assert lines, "expected chunk-derived candidates"
    for rec in lines:
        assert rec["subject"]["label"] == "NOUNPHRASE"
        assert rec["object"]["label"] == "NOUNPHRASE"
