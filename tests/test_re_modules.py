import io
import json
import sys
import types

import pytest

if "numpy" not in sys.modules:
    fake_np = types.ModuleType("numpy")

    class FakeArray:
        def __init__(self, data, dtype=None):
            self.data = data
            self.dtype = dtype
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                self.shape = (len(data), len(data[0]))
            elif isinstance(data, list):
                self.shape = (len(data),)
            else:
                self.shape = ()

        def reshape(self, *shape):
            # Simple reshape - just return a new FakeArray with new shape
            flat = self._flatten()
            if len(shape) == 1:
                if isinstance(shape[0], tuple):
                    new_shape = shape[0]
                else:
                    # Handle reshape(1, -1) case
                    if shape[0] == 1:
                        return FakeArray([flat], self.dtype)
                    new_shape = shape[0]
            elif len(shape) == 2:
                # Handle reshape(1, -1) case
                if shape[0] == 1:
                    return FakeArray([flat], self.dtype)
                new_shape = shape
            else:
                new_shape = shape
            return FakeArray(flat, self.dtype)

        def _flatten(self):
            result = []
            for item in self.data:
                if isinstance(item, list):
                    result.extend(item)
                else:
                    result.append(item)
            return result

        def astype(self, dtype):
            return FakeArray(self.data, dtype)

        def flatten(self):
            return FakeArray(self._flatten(), self.dtype)

        def __getitem__(self, idx):
            item = self.data[idx]
            # If item is a list, wrap it in FakeArray for proper method access
            if isinstance(item, list):
                return FakeArray(item, self.dtype)
            return item

        def __setitem__(self, idx, value):
            if isinstance(idx, tuple):
                # Handle 2D indexing like probs[0][1] = 0.9
                self.data[idx[0]][idx[1]] = value
            else:
                self.data[idx] = value

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            for item in self.data:
                if isinstance(item, list):
                    yield FakeArray(item, self.dtype)
                else:
                    yield item

    def array(data, dtype=None):
        return FakeArray(data, dtype)

    def argmax(seq):
        if hasattr(seq, '__len__') and len(seq) > 0:
            if hasattr(seq[0], '__len__'):
                # 2D array, get argmax of first row
                return max(range(len(seq[0])), key=lambda i: seq[0][i])
            return max(range(len(seq)), key=lambda i: seq[i])
        return 0

    def zeros(shape, dtype=float):
        if isinstance(shape, tuple) and len(shape) == 2:
            rows, cols = shape
            return FakeArray([[dtype(0) for _ in range(cols)] for _ in range(rows)], dtype)
        elif isinstance(shape, tuple) and len(shape) == 1:
            return FakeArray([dtype(0)] * shape[0], dtype)
        else:
            rows, cols = shape
            return FakeArray([[dtype(0) for _ in range(cols)] for _ in range(rows)], dtype)

    def asarray(data, dtype=None):
        if isinstance(data, FakeArray):
            return data
        return FakeArray(data, dtype)

    fake_np.array = array
    fake_np.argmax = argmax
    fake_np.zeros = zeros
    fake_np.asarray = asarray
    fake_np.isscalar = lambda value: isinstance(value, (int, float))
    fake_np.float32 = float
    fake_np.float64 = float
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
        shape = [None, 384]  # Embedding dimension for all-MiniLM-L6-v2

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def get_outputs(self):
            return [
                types.SimpleNamespace(name="label", shape=[None], type="tensor(string)"),
                types.SimpleNamespace(name="probabilities", shape=[None, len(classes)], type="tensor(float)"),
            ]

        def run(self, _outputs, feeds):
            assert isinstance(feeds, dict)
            probs = fake_np.zeros((1, len(classes)), dtype=float)
            probs[0][1] = 0.9
            # Return FakeArray objects to match ONNX output format
            return [fake_np.array(["uses"]), probs]

    # Mock embedder to return fake embeddings (384 dims for all-MiniLM-L6-v2)
    class FakeEmbedder:
        def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
            # Return fake embeddings: one per text, each 384 dimensions
            if isinstance(texts, str):
                texts = [texts]
            return fake_np.array([[0.1] * 384 for _ in texts])

    monkeypatch.setattr(re_score, "get_embedder", lambda: FakeEmbedder())
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
    classes_path = tmp_path / "classes.json"
    classes = ["none", "uses"]
    classes_path.write_text(json.dumps(classes), encoding="utf-8")

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

    # Mock the classes.json file reading
    original_open = open
    def mock_open(path, *args, **kwargs):
        if "classes.json" in str(path):
            return original_open(classes_path, *args, **kwargs)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr("builtins.open", mock_open)
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
    classes_path = tmp_path / "classes.json"
    classes = ["none", "uses", "provides"]
    classes_path.write_text(json.dumps(classes), encoding="utf-8")

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

    # Mock the classes.json file reading
    original_open = open
    def mock_open(path, *args, **kwargs):
        if "classes.json" in str(path):
            return original_open(classes_path, *args, **kwargs)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr("builtins.open", mock_open)
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
