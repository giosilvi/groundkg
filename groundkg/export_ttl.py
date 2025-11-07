import sys
import json
import os
from collections import Counter

PREFIX = """@prefix ex: <https://example.invalid/vocab#> .
@prefix schema: <http://schema.org/> .

"""


def iri(kind, name):
    safe = name.strip().replace(" ", "_").replace("/", "_").replace(",", "")
    return f"ex:{kind}/{safe}"


def emit_edge_triple(e):
    s = iri("node", e["subject"])
    o = iri("node", e["object"])
    p = "ex:" + e["predicate"]
    return f"{s} {p} {o} .\n", s


def emit_attr_triples(attr, subj_iri):
    # stable-ish id from name + evidence start
    evid = attr.get("evidence", {})
    aid = attr.get("id") or f"{attr.get('name','attr')}_{int(evid.get('char_start',0))}"
    airi = iri("attr", aid)
    lines = []
    # link from subject
    lines.append(f"{subj_iri} ex:hasAttribute {airi} .\n")
    # attribute properties
    name = attr.get("name", "").replace('"', '\\"')
    lines.append(f'{airi} a ex:Attribute ; ex:name "{name}"')
    # values
    if "valueNumber" in attr:
        lines.append(f" ; ex:valueNumber {attr['valueNumber']}")
        unit = attr.get("unit")
        if unit:
            lines.append(f' ; ex:unit "{unit}"')
    if "valueString" in attr:
        vs = attr["valueString"].replace('"', '\\"')
        lines.append(f' ; ex:valueString "{vs}"')
    if "valueBoolean" in attr:
        vb = "true" if attr["valueBoolean"] else "false"
        lines.append(f" ; ex:valueBoolean {vb}")
    # time
    if attr.get("time"):
        lines.append(f" ; ex:time \"{attr['time']}\"")
    lines.append(" .\n")
    return "".join(lines)


def main():
    edges_path = sys.argv[1]
    out_lines = [PREFIX]
    subjects = []
    edges = []
    with open(edges_path, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            triple, s_iri = emit_edge_triple(e)
            subjects.append(e["subject"])
            edges.append(e)
            out_lines.append(triple)
    # choose primary subject (most frequent in edges) for attaching attributes in v0
    primary_subj = Counter(subjects).most_common(1)[0][0] if subjects else "Unknown"
    primary_subj_iri = iri("node", primary_subj)

    # try to add attributes.jsonl from same directory
    attr_path = os.path.join(os.path.dirname(edges_path) or ".", "attributes.jsonl")
    if os.path.exists(attr_path):
        try:
            with open(attr_path, "r", encoding="utf-8") as af:
                for line in af:
                    attr = json.loads(line)
                    out_lines.append(emit_attr_triples(attr, primary_subj_iri))
        except Exception:
            # ignore attribute export errors in v0
            pass

    sys.stdout.write("".join(out_lines))


if __name__ == "__main__":
    main()
