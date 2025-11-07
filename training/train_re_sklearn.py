# training/train_re_sklearn.py
# DEPRECATED: This file uses TF-IDF feature extraction.
# Use training/train_re_transformers.py instead (sentence transformer embeddings).
# This file is kept for backward compatibility and will be removed in a future version.

import json, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

def load_jsonl(p): return [json.loads(l) for l in open(p, "r", encoding="utf-8") if l.strip()]

def load_data(train_p, dev_p):
    tr = load_jsonl(train_p); dv = load_jsonl(dev_p)
    Xtr = [r["text"] for r in tr]; ytr = [r["label"] for r in tr]
    Xdv = [r["text"] for r in dv]; ydv = [r["label"] for r in dv]
    return (Xtr, ytr), (Xdv, ydv)

def pick_thresholds(clf, X, y, classes, target_prec=0.90):
    # per-class thresholds; default 0.80 if curve too small
    thresholds = {}
    proba = clf.predict_proba(X)  # [N, C]
    for i, cls in enumerate(classes):
        if cls == "none":
            thresholds[cls] = 1.00
            continue
        y_true = np.array([1 if yy==cls else 0 for yy in y])
        y_score = proba[:, i]
        if y_true.sum() < 5:
            thresholds[cls] = 0.80
            continue
        prec, rec, thr = precision_recall_curve(y_true, y_score)
        # pick highest recall subject to precision >= target_prec
        best_t = 0.80
        best_rec = -1
        for p, r, t in zip(prec, rec, np.r_[thr, 1.0]):
            if p >= target_prec and r > best_rec:
                best_rec, best_t = r, float(t)
        thresholds[cls] = round(best_t, 3)
    return thresholds

def main():
    os.makedirs("models", exist_ok=True)
    (Xtr, ytr), (Xdv, ydv) = load_data("training/re_train.jsonl","training/re_dev.jsonl")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,3), min_df=1, max_features=100000)),
        ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear"))
    ])
    pipe.fit(Xtr, ytr)
    classes = list(pipe.named_steps["clf"].classes_)

    # thresholds from dev
    thresholds = pick_thresholds(pipe, Xdv, ydv, classes, target_prec=0.95)
    with open("models/thresholds.json","w",encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    with open("models/classes.json","w",encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False)

    # export ONNX
    onnx_model = convert_sklearn(
        pipe,
        initial_types=[("text", StringTensorType([None]))],
        options={type(pipe.named_steps["clf"]): {"zipmap": False}},
        target_opset=13
    )
    with open("models/promoter_v1.onnx","wb") as f:
        f.write(onnx_model.SerializeToString())
    print("Saved models/promoter_v1.onnx, thresholds.json, classes.json")

if __name__ == "__main__":
    main()

