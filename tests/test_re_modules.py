import io
import json
import sys
import types

import pytest

if "numpy" not in sys.modules:
    fake_np = types.ModuleType("numpy")

    def array(data, dtype=None):
        return data

    def argmax(seq):
        return max(range(len(seq)), key=lambda i: seq[i])

    def zeros(shape, dtype=float):
        rows, cols = shape
        return [[dtype(0) for _ in range(cols)] for _ in range(rows)]

    fake_np.array = array
    fake_np.argmax = argmax
    fake_np.zeros = zeros
    fake_np.isscalar = lambda value: isinstance(value, (int, float))
    sys.modules["numpy"] = fake_np

from groundkg import re_infer, re_score


def test_re_score_mark_orders_entities():
    text = "The gadget Alice built"
    subject = {"start": 4, "end": 10}
    obj = {"start": 11, "end": 16}
    marked = re_score.mark(text, obj, subject)  # subject starts later than object
    assert "[E1]" in marked and "[E2]" in marked
    assert marked.index("[E1]") < marked.index("[E2]")


def test_re_score_main_streams_predictions(tmp_path, monkeypatch):
    cand_path = tmp_path / "cands.jsonl"
    candidates = [
        {
            "doc_id": "d1",
            "sent_start": 0,
            "text": "Alice uses the gadget",
            "subject": {"text": "Alice", "start": 0, "end": 5},
            "object": {"text": "gadget", "start": 12, "end": 18},
        }
    ]
    cand_path.write_text("\n".join(json.dumps(c) for c in candidates) + "\n", encoding="utf-8")

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_text("", encoding="utf-8")
    classes_path = tmp_path / "classes.json"
    classes = ["none", "uses"]
    classes_path.write_text(json.dumps(classes), encoding="utf-8")

    class FakeInput:
        name = "text"

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def run(self, _outputs, feeds):
            assert isinstance(feeds, dict)
            probs = fake_np.zeros((1, len(classes)), dtype=float)
            probs[0][1] = 0.9
            return [["uses"], probs]

    monkeypatch.setattr(re_score.ort, "InferenceSession", lambda *a, **k: FakeSession())
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    monkeypatch.setattr(
        "sys.argv",
        [
            "re_score.py",
            str(cand_path),
            str(onnx_path),
            str(classes_path),
        ],
    )

    re_score.main()

    lines = [json.loads(line) for line in buf.getvalue().splitlines() if line]
    assert len(lines) == 1
    record = lines[0]
    assert record["pred"] == "uses"
    assert abs(record["prob"] - 0.9) < 1e-6
    assert record["subject"]["text"] == "Alice"
    assert record["object"]["text"] == "gadget"


def test_re_score_missing_model_exits(tmp_path, monkeypatch, capsys):
    cand_path = tmp_path / "cands.jsonl"
    cand_path.write_text("{}\n", encoding="utf-8")
    missing_model = tmp_path / "missing.onnx"
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(["none"]), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["re_score.py", str(cand_path), str(missing_model), str(classes_path)],
    )

    with pytest.raises(SystemExit) as exc:
        re_score.main()

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "missing" in captured.err.lower()


def test_type_compatible_enforces_allowed_pairs():
    assert re_infer.type_compatible("uses", "PERSON", "PRODUCT")
    assert not re_infer.type_compatible("uses", "PERSON", "GPE")


def test_mark_indicates_swapped_subject_object():
    text = "Paris is home to Alice"
    subject = {"start": 17, "end": 22}
    obj = {"start": 0, "end": 5}
    marked, swapped = re_infer.mark(text, subject, obj)
    assert swapped is True
    assert marked.startswith("[E1]Paris")


def test_main_filters_by_threshold_and_types(tmp_path, monkeypatch):
    cand_path = tmp_path / "cands.jsonl"
    preds_yaml = tmp_path / "preds.yaml"
    preds_yaml.write_text("predicates", encoding="utf-8")
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_text("", encoding="utf-8")
    thresh_path = tmp_path / "thresh.json"
    thresholds = {"uses": 0.5}
    thresh_path.write_text(json.dumps(thresholds), encoding="utf-8")

    candidates = [
        {
            "doc_id": "d1",
            "sent_start": 0,
            "text": "Alice uses the gadget",
            "subject": {"text": "Alice", "start": 0, "end": 5, "label": "PERSON"},
            "object": {"text": "gadget", "start": 12, "end": 18, "label": "PRODUCT"},
        },
        {
            "doc_id": "d1",
            "sent_start": 0,
            "text": "Alice uses the city",
            "subject": {"text": "Alice", "start": 0, "end": 5, "label": "PERSON"},
            "object": {"text": "city", "start": 12, "end": 16, "label": "GPE"},
        },
    ]
    cand_path.write_text("\n".join(json.dumps(c) for c in candidates) + "\n", encoding="utf-8")

    classes = json.load(open("models/classes.json", "r", encoding="utf-8"))
    uses_idx = classes.index("uses")

    class FakeInput:
        name = "input_text"

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def run(self, *_args, **_kwargs):
            probs = fake_np.zeros((1, len(classes)), dtype=float)
            probs[0][uses_idx] = 0.92
            labels = ["uses"]
            return [labels, probs]

    monkeypatch.setattr(re_infer.ort, "InferenceSession", lambda *a, **k: FakeSession())
    buf = io.StringIO()
    monkeypatch.setattr(
        "sys.argv",
        [
            "re_infer.py",
            str(cand_path),
            str(preds_yaml),
            str(onnx_path),
            str(thresh_path),
        ],
    )
    monkeypatch.setattr("sys.stdout", buf)

    re_infer.main()

    lines = [line for line in buf.getvalue().splitlines() if line]
    assert len(lines) == 1
    edge = json.loads(lines[0])
    assert edge["subject"] == "Alice"
    assert edge["object"] == "gadget"
    assert edge["predicate"] == "uses"


def test_main_skips_low_prob_and_allows_unknown_predicate(tmp_path, monkeypatch):
    cand_path = tmp_path / "cands.jsonl"
    preds_yaml = tmp_path / "preds.yaml"
    preds_yaml.write_text("predicates", encoding="utf-8")
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_text("", encoding="utf-8")
    thresh_path = tmp_path / "thresh.json"
    thresholds = {"uses": 0.8, "provides": 0.5}
    thresh_path.write_text(json.dumps(thresholds), encoding="utf-8")

    candidates = [
        {
            "doc_id": "d1",
            "sent_start": 0,
            "text": "Alice uses the gadget",
            "subject": {"text": "Alice", "start": 0, "end": 5, "label": "PERSON"},
            "object": {"text": "gadget", "start": 12, "end": 18, "label": "PRODUCT"},
        },
        {
            "doc_id": "d2",
            "sent_start": 10,
            "text": "Bob provides Rome with tools",
            "subject": {"text": "Bob", "start": 0, "end": 3, "label": "PERSON"},
            "object": {"text": "Rome", "start": 13, "end": 17, "label": "GPE"},
        },
    ]
    cand_path.write_text("\n".join(json.dumps(c) for c in candidates) + "\n", encoding="utf-8")

    classes = json.load(open("models/classes.json", "r", encoding="utf-8"))
    uses_idx = classes.index("uses")
    provides_idx = classes.index("provides")

    class FakeInput:
        name = "input_text"

    class FakeSession:
        def __init__(self):
            self.outputs = [
                [["uses"], [[0.0 for _ in classes]]],
                [["provides"], [[0.0 for _ in classes]]],
            ]
            self.outputs[0][1][0][uses_idx] = 0.6  # below threshold
            self.outputs[1][1][0][provides_idx] = 0.9

        def get_inputs(self):
            return [FakeInput()]

        def run(self, *_args, **_kwargs):
            return self.outputs.pop(0)

    monkeypatch.setattr(re_infer.ort, "InferenceSession", lambda *a, **k: FakeSession())
    monkeypatch.setattr(re_infer, "ALLOWED_TYPES", {k: v for k, v in re_infer.ALLOWED_TYPES.items() if k != "provides"})
    buf = io.StringIO()
    monkeypatch.setattr(
        "sys.argv",
        [
            "re_infer.py",
            str(cand_path),
            str(preds_yaml),
            str(onnx_path),
            str(thresh_path),
        ],
    )
    monkeypatch.setattr("sys.stdout", buf)

    re_infer.main()

    lines = [json.loads(line) for line in buf.getvalue().splitlines() if line]
    assert len(lines) == 1
    edge = lines[0]
    assert edge["predicate"] == "provides"
    assert edge["subject"] == "Bob"
    assert edge["object"] == "Rome"
