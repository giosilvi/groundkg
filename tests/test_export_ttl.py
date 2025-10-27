import io
import json

from groundkg import export_ttl


def test_iri_sanitizes_text():
    assert export_ttl.iri("node", "Acme, Inc./R&D") == "ex:node/Acme_Inc._R&D"


def test_emit_edge_triple_builds_expected_turtle():
    triple, subj = export_ttl.emit_edge_triple({"subject": "Alice", "predicate": "uses", "object": "Gadget"})
    assert triple == "ex:node/Alice ex:uses ex:node/Gadget .\n"
    assert subj == "ex:node/Alice"


def test_emit_attr_triples_formats_values():
    attr = {
        "name": "Battery Life",
        "valueNumber": 12,
        "unit": "hours",
        "valueBoolean": True,
        "valueString": "High capacity",
        "time": "2023-05-01",
        "evidence": {"char_start": 42},
    }
    rendered = export_ttl.emit_attr_triples(attr, "ex:node/Alice")
    assert "ex:hasAttribute" in rendered
    assert "ex:valueNumber 12" in rendered
    assert "ex:unit \"hours\"" in rendered
    assert "ex:valueBoolean true" in rendered
    assert "ex:valueString \"High capacity\"" in rendered
    assert rendered.endswith(" .\n")


def test_main_reads_edges_and_attributes(tmp_path, monkeypatch):
    edges_path = tmp_path / "edges.jsonl"
    attrs_path = tmp_path / "attributes.jsonl"
    edge = {"subject": "Alice", "predicate": "uses", "object": "Gadget"}
    edges_path.write_text(json.dumps(edge) + "\n", encoding="utf-8")
    attrs_path.write_text(json.dumps({"name": "Battery", "valueNumber": 3}) + "\n", encoding="utf-8")

    buf = io.StringIO()
    monkeypatch.setattr("sys.argv", ["export_ttl.py", str(edges_path)])
    monkeypatch.setattr("sys.stdout", buf)

    export_ttl.main()

    output = buf.getvalue()
    assert output.startswith(export_ttl.PREFIX)
    assert "ex:node/Alice ex:uses ex:node/Gadget" in output
    assert "ex:hasAttribute" in output


def test_main_ignores_malformed_attribute_lines(tmp_path, monkeypatch):
    edges_path = tmp_path / "edges.jsonl"
    attrs_path = tmp_path / "attributes.jsonl"
    edge = {"subject": "Alice", "predicate": "uses", "object": "Gadget"}
    edges_path.write_text(json.dumps(edge) + "\n", encoding="utf-8")
    attrs_path.write_text("{" + "\n", encoding="utf-8")  # malformed JSON

    buf = io.StringIO()
    monkeypatch.setattr("sys.argv", ["export_ttl.py", str(edges_path)])
    monkeypatch.setattr("sys.stdout", buf)

    export_ttl.main()

    output = buf.getvalue()
    assert "ex:node/Alice ex:uses ex:node/Gadget" in output
    assert "ex:hasAttribute" not in output
