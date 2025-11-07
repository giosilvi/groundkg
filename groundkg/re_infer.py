# groundkg/re_infer.py
import sys
import json
import os
import onnxruntime as ort
import numpy as np

ALLOWED_TYPES = {
    # predicate: (allowed_subject_labels, allowed_object_labels)
    # Labels are spaCy NER labels plus our "NOUNPHRASE" heuristic from candidates
    "headquartered_in": ({"ORG", "PERSON", "FAC", "NORP"}, {"GPE", "LOC", "FAC"}),
    "operates_in": ({"ORG", "PRODUCT"}, {"GPE", "LOC", "NORP"}),
    "subsidiary_of": ({"ORG"}, {"ORG"}),
    "parent_of": ({"ORG"}, {"ORG"}),
    "member_of": ({"ORG", "PERSON", "PRODUCT"}, {"ORG"}),
    "part_of": (
        {"ORG", "PRODUCT", "FAC", "WORK_OF_ART", "LOC", "GPE"},
        {"ORG", "PRODUCT", "FAC", "WORK_OF_ART", "LOC", "GPE"},
    ),
    "uses": (
        {"ORG", "PERSON", "PRODUCT"},
        {"PRODUCT", "WORK_OF_ART", "LAW", "NOUNPHRASE"},
    ),
    "provides": ({"ORG", "PRODUCT"}, {"NOUNPHRASE", "PRODUCT", "WORK_OF_ART"}),
    "requires": ({"LAW", "ORG"}, {"NOUNPHRASE", "WORK_OF_ART"}),
    "prohibits": ({"LAW", "ORG"}, {"NOUNPHRASE", "WORK_OF_ART"}),
    "covered_by": (
        {
            "ORG",
            "PRODUCT",
            "PERSON",
            "FAC",
            "GPE",
            "EVENT",
            "LAW",
            "NORP",
            "LOC",
            "WORK_OF_ART",
            "NOUNPHRASE",
        },
        {"LAW"},
    ),
    "type": (
        {
            "ORG",
            "PRODUCT",
            "PERSON",
            "FAC",
            "GPE",
            "EVENT",
            "LAW",
            "NORP",
            "LOC",
            "WORK_OF_ART",
            "NOUNPHRASE",
        },
        {"NOUNPHRASE", "ORG", "PRODUCT", "WORK_OF_ART"},
    ),
}


def type_compatible(pred, s_label, o_label):
    if pred not in ALLOWED_TYPES:
        return True
    subj_allowed, obj_allowed = ALLOWED_TYPES[pred]
    return (s_label in subj_allowed) and (o_label in obj_allowed)


def mark(text, s, o):
    s0, s1 = s["start"], s["end"]
    o0, o1 = o["start"], o["end"]
    if s0 > o0:
        s0, s1, o0, o1 = o0, o1, s0, s1
        swapped = True
    else:
        swapped = False
    marked = (
        text[:s0]
        + "[E1]"
        + text[s0:s1]
        + "[/E1]"
        + text[s1:o0]
        + "[E2]"
        + text[o0:o1]
        + "[/E2]"
        + text[o1:]
    )
    return marked, swapped


def main():
    cand_path, preds_yaml, onnx_path, thresh_path = sys.argv[1:5]
    if not os.path.exists(onnx_path):
        sys.exit(0)

    thresholds = json.load(open(thresh_path, "r", encoding="utf-8"))
    classes = json.load(open("models/classes.json", "r", encoding="utf-8"))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    with open(cand_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            text = c["text"]
            s, o = c["subject"], c["object"]
            marked, _ = mark(text, s, o)

            # ONNX returns (labels, probabilities)
            outputs = sess.run(None, {inp_name: np.array([marked])})
            pred_label = outputs[0][0]  # predicted class name
            probs = outputs[1][0]  # [C] array of probabilities

            # Get probability for predicted class
            pred_idx = classes.index(pred_label)
            p = float(probs[pred_idx])
            pred = pred_label

            thr = float(thresholds.get(pred, 0.85))
            if pred == "none" or p < thr:
                continue

            # Basic entity-type sanity check per predicate
            s_label = s.get("label", "")
            o_label = o.get("label", "")
            if not type_compatible(pred, s_label, o_label):
                continue

            edge = {
                "subject": s["text"].strip(),
                "predicate": pred,
                "object": o["text"].strip(),
                "evidence": {
                    "doc_id": c["doc_id"],
                    "quote": c["text"],
                    "char_start": c["sent_start"],
                    "char_end": c["sent_start"] + len(c["text"]),
                },
            }
            sys.stdout.write(json.dumps(edge, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
