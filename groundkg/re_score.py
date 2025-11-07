# groundkg/re_score.py
import sys
import json
import os
import onnxruntime as ort
import numpy as np
from sentence_transformers import SentenceTransformer

# Use same model as training
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Global cache for sentence transformer model
_embedder_cache = None


def get_embedder():
    """Get or create sentence transformer model (cached)."""
    global _embedder_cache
    if _embedder_cache is None:
        _embedder_cache = SentenceTransformer(MODEL_NAME)
    return _embedder_cache


def mark(text, s, o):
    s0, s1 = s["start"], s["end"]
    o0, o1 = o["start"], o["end"]
    if s0 > o0:
        s0, s1, o0, o1 = o0, o1, s0, s1
    return (
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


def main():
    cand_path, onnx_path, classes_path = sys.argv[1:4]
    if not os.path.exists(onnx_path):
        print("ERROR: models/promoter_v1.onnx missing", file=sys.stderr)
        sys.exit(2)
    
    # Load sentence transformer model
    embedder = get_embedder()
    
    # Load ONNX model and classes
    classes = json.load(open(classes_path, "r", encoding="utf-8"))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    
    # Check input shape to verify it expects embeddings
    input_shape = sess.get_inputs()[0].shape
    if len(input_shape) != 2 or input_shape[1] != EMBEDDING_DIM:
        print(f"WARNING: ONNX model expects shape {input_shape}, but embedding dim is {EMBEDDING_DIM}", file=sys.stderr)
    
    # Debug: Check output structure
    output_info = []
    for i, out in enumerate(sess.get_outputs()):
        output_info.append(f"outputs[{i}]: name={out.name}, shape={out.shape}, type={out.type}")
    print(f"DEBUG: ONNX output info: {', '.join(output_info)}", file=sys.stderr)
    
    # Process candidates in batches for efficiency
    batch_texts = []
    batch_candidates = []
    batch_size = 32
    
    with open(cand_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            text = c["text"]
            s = c["subject"]
            o = c["object"]
            marked = mark(text, s, o)
            
            batch_texts.append(marked)
            batch_candidates.append(c)
            
            # Process batch when full
            if len(batch_texts) >= batch_size:
                # Get embeddings
                embeddings = embedder.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
                
                # Run ONNX model on embeddings
                for emb, c in zip(embeddings, batch_candidates):
                    outputs = sess.run(None, {inp_name: emb.reshape(1, -1).astype(np.float32)})
                    # ONNX LogisticRegression with zipmap=False outputs:
                    # outputs[0] = label (string) - predicted class name
                    # outputs[1] = probabilities [batch_size, num_classes] - probability array
                    # Use outputs[1] for probabilities
                    if len(outputs) >= 2:
                        # Find the output with probabilities (2D float array)
                        probs = None
                        for out_idx, out in enumerate(outputs):
                            if len(out.shape) == 2 and out.shape[1] == len(classes) and out.dtype in (np.float32, np.float64):
                                probs = out[0]  # Get first batch item [num_classes]
                                break
                        if probs is None:
                            # Fallback: use outputs[1] if it exists and is numeric
                            if len(outputs) > 1 and not isinstance(outputs[1][0], str):
                                probs = outputs[1][0] if len(outputs[1].shape) == 2 else outputs[1]
                    else:
                        probs = outputs[0][0]
                    
                    if probs is None:
                        raise ValueError(f"Could not find probability output. Outputs: {[(i, o.shape, o.dtype) for i, o in enumerate(outputs)]}")
                    
                    # Ensure probs is numpy array of floats
                    probs = np.asarray(probs, dtype=np.float32).flatten()
                    if len(probs) != len(classes):
                        raise ValueError(f"Probability array length {len(probs)} doesn't match classes {len(classes)}")
                    
                    i = int(np.argmax(probs))
                    pred = classes[i]
                    p = float(probs[i])
                    rec = {
                        "doc_id": c["doc_id"],
                        "sent_start": c["sent_start"],
                        "text": c["text"],
                        "subject": c["subject"],
                        "object": c["object"],
                        "pred": pred,
                        "prob": p,
                    }
                    sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
                batch_texts = []
                batch_candidates = []
        
        # Process remaining items
        if batch_texts:
            embeddings = embedder.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
            for emb, c in zip(embeddings, batch_candidates):
                outputs = sess.run(None, {inp_name: emb.reshape(1, -1).astype(np.float32)})
                # ONNX LogisticRegression with zipmap=False outputs:
                # outputs[0] = label (string) - predicted class name
                # outputs[1] = probabilities [batch_size, num_classes] - probability array
                # Use outputs[1] for probabilities
                if len(outputs) >= 2:
                    # Find the output with probabilities (2D float array)
                    probs = None
                    for out_idx, out in enumerate(outputs):
                        if len(out.shape) == 2 and out.shape[1] == len(classes) and out.dtype in (np.float32, np.float64):
                            probs = out[0]  # Get first batch item [num_classes]
                            break
                    if probs is None:
                        # Fallback: use outputs[1] if it exists and is numeric
                        if len(outputs) > 1 and not isinstance(outputs[1][0], str):
                            probs = outputs[1][0] if len(outputs[1].shape) == 2 else outputs[1]
                else:
                    probs = outputs[0][0]
                
                if probs is None:
                    raise ValueError(f"Could not find probability output. Outputs: {[(i, o.shape, o.dtype) for i, o in enumerate(outputs)]}")
                
                # Ensure probs is numpy array of floats
                probs = np.asarray(probs, dtype=np.float32).flatten()
                if len(probs) != len(classes):
                    raise ValueError(f"Probability array length {len(probs)} doesn't match classes {len(classes)}")
                
                i = int(np.argmax(probs))
                pred = classes[i]
                p = float(probs[i])
                rec = {
                    "doc_id": c["doc_id"],
                    "sent_start": c["sent_start"],
                    "text": c["text"],
                    "subject": c["subject"],
                    "object": c["object"],
                    "pred": pred,
                    "prob": p,
                }
                sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
