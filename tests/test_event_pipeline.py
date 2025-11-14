import json
import sys

import pytest

np = pytest.importorskip("numpy")

from groundkg import event_extract, events_to_edges, re_score


def test_event_extract_main_generates_events(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    events_out = tmp_path / "events.jsonl"

    records = [
        {
            "doc_id": "doc1",
            "text": "MegaCorp acquired StartUp for $5 million on Jan 2, 2022.",
        },
        {
            "doc_id": "doc2",
            "text": "Bright Future secured $3M from Big VC on Feb 5, 2021.",
        },
        {
            "doc_id": "doc3",
            "text": "Tech Corp launched HyperWidget on 2023.",
        },
        {
            "doc_id": "doc4",
            "source_org": "ACME",
            "text": "ACME appointed Jane Doe as CTO on Mar 3, 2020.",
        },
        {
            "doc_id": "doc5",
            "text": "John Smith founded Future Labs in 2019.",
        },
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "event_extract",
            "--manifest",
            str(manifest),
            "--out",
            str(events_out),
        ],
    )

    event_extract.main()

    lines = [json.loads(line) for line in events_out.read_text(encoding="utf-8").splitlines()]

    event_types = {line["type"] for line in lines}
    expected_types = {"Acquisition", "Funding", "Launch", "Appointment", "Founding"}
    assert expected_types.issubset(event_types)

    acq = next(line for line in lines if line["type"] == "Acquisition")
    assert acq["roles"].get("acquirer") == "MegaCorp"
    assert acq["roles"].get("target", "").startswith("StartUp")
    assert acq["amount_text"] == "$5 million"
    assert acq["date_text"] == "Jan 2, 2022"

    funding = next(line for line in lines if line["type"] == "Funding")
    assert funding["roles"]["recipient"] == "Bright Future"

    assert any(ev["roles"].get("actor") == "ACME" for ev in lines if ev["type"] == "Appointment")

    founding = next(line for line in lines if line["type"] == "Founding")
    assert "founder_or_actor" in founding["roles"]


def test_events_to_edges_main(tmp_path, monkeypatch):
    events_file = tmp_path / "events.jsonl"
    edges_out = tmp_path / "edges.jsonl"

    event_record = {
        "event_id": "E1",
        "type": "Acquisition",
        "trigger": "acquired",
        "date_text": "Jan 2, 2022",
        "amount_text": "$5 million",
        "roles": {"acquirer": "MegaCorp", "target": "StartUp", "empty": ""},
        "confidence": 0.75,
        "source": "doc1#s",
    }
    events_file.write_text(json.dumps(event_record) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["events_to_edges", "--events", str(events_file), "--out", str(edges_out)],
    )

    events_to_edges.main()

    edges = [json.loads(line) for line in edges_out.read_text(encoding="utf-8").splitlines()]
    subjects = {edge["subject"] for edge in edges}
    assert subjects == {"event:E1"}
    predicates = {edge["predicate"] for edge in edges}
    assert predicates == {"type", "trigger", "date", "amount", "acquirer", "target"}


def test_re_score_mark_orders_entities():
    text = "Object met Subject"
    subject = {"start": 11, "end": 18}
    obj = {"start": 0, "end": 6}
    marked = re_score.mark(text, subject, obj)
    assert marked.startswith("[E1]Object[/E1] met [E2]Subject[/E2]")


def test_re_score_main_batches(tmp_path, monkeypatch, capsys):
    cand_path = tmp_path / "candidates.jsonl"
    onnx_path = tmp_path / "model.onnx"
    classes_path = tmp_path / "classes.json"

    num_candidates = 33
    candidates = []
    for i in range(num_candidates):
        candidates.append(
            {
                "doc_id": f"doc{i}",
                "sent_start": i,
                "text": f"Sentence {i}",
                "subject": {"start": 0, "end": 7},
                "object": {"start": 9, "end": 12},
            }
        )
    cand_path.write_text("\n".join(json.dumps(c) for c in candidates) + "\n", encoding="utf-8")
    onnx_path.write_text("placeholder", encoding="utf-8")
    classes_path.write_text(json.dumps(["NEG", "POS"]), encoding="utf-8")

    class DummyEmbedder:
        def __init__(self):
            self.calls = []

        def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
            self.calls.append(list(texts))
            batch = np.arange(len(texts) * re_score.EMBEDDING_DIM, dtype=np.float32)
            return batch.reshape(len(texts), re_score.EMBEDDING_DIM)

    class DummyInput:
        def __init__(self):
            self.name = "input"
            self.shape = [None, re_score.EMBEDDING_DIM]

    class DummyOutputInfo:
        def __init__(self, name):
            self.name = name
            self.shape = [None, 2]
            self.type = "tensor(float)"

    class DummySession:
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers
            self.calls = 0

        def get_inputs(self):
            return [DummyInput()]

        def get_outputs(self):
            return [DummyOutputInfo("label"), DummyOutputInfo("prob")]

        def run(self, _, feeds):
            self.calls += 1
            probs = np.zeros((1, 2), dtype=np.float32)
            probs[0, self.calls % 2] = 0.8
            return [np.array(["label"]), probs]

    dummy_embedder = DummyEmbedder()
    monkeypatch.setattr(re_score, "get_embedder", lambda: dummy_embedder)
    monkeypatch.setattr(re_score.ort, "InferenceSession", DummySession)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "re_score",
            str(cand_path),
            str(onnx_path),
            str(classes_path),
        ],
    )

    re_score.main()

    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.out.splitlines()]
    assert len(lines) == num_candidates
    assert {rec["pred"] for rec in lines} <= {"NEG", "POS"}
    assert dummy_embedder.calls  # ensure embeddings were requested
