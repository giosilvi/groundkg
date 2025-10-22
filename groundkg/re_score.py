# groundkg/re_score.py
import sys, json, os
import onnxruntime as ort
import numpy as np

def mark(text, s, o):
    s0, s1 = s["start"], s["end"]; o0, o1 = o["start"], o["end"]
    if s0 > o0: s0, s1, o0, o1 = o0, o1, s0, s1
    return text[:s0]+"[E1]"+text[s0:s1]+"[/E1]"+text[s1:o0]+"[E2]"+text[o0:o1]+"[/E2]"+text[o1:]

def main():
    cand_path, onnx_path, classes_path = sys.argv[1:4]
    if not os.path.exists(onnx_path):
        print("ERROR: models/promoter_v1.onnx missing", file=sys.stderr); sys.exit(2)
    classes = json.load(open(classes_path,"r",encoding="utf-8"))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    with open(cand_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            text = c["text"]; s=c["subject"]; o=c["object"]
            marked = mark(text, s, o)
            outputs = sess.run(None, {inp_name: np.array([marked])})
            probs = outputs[1][0]  # probability array
            i = int(np.argmax(probs)); pred = classes[i]; p = float(probs[i])
            rec = {
                "doc_id": c["doc_id"], "sent_start": c["sent_start"], "text": text,
                "subject": s, "object": o, "pred": pred, "prob": p
            }
            sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

