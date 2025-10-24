import io
import runpy


def test_cli_entrypoint_prints_message(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    runpy.run_module("groundkg.cli", run_name="__main__")
    assert "Use `python -m groundkg.extract_open" in buf.getvalue()
