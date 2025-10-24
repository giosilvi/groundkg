import sys
import types
from pathlib import Path

# Ensure the repository root is importable as a package during tests
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide a lightweight onnxruntime stub so modules can be imported in tests
if "onnxruntime" not in sys.modules:
    class _StubSession:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "onnxruntime is stubbed in tests. Monkeypatch InferenceSession in individual tests."
            )

    sys.modules["onnxruntime"] = types.SimpleNamespace(InferenceSession=_StubSession)

if "spacy" not in sys.modules:
    def _stub_load(*_args, **_kwargs):
        raise RuntimeError(
            "spacy is stubbed in tests. Monkeypatch spacy.load within individual tests."
        )

    sys.modules["spacy"] = types.SimpleNamespace(load=_stub_load)
