# GroundKG: Model-Based Relation Extraction Architecture

## Overview

This project uses a **model-based relation extraction** pipeline end-to-end (no rule fallback at runtime).

## Pipeline Architecture

### Model-Based Pipeline

```
crawl (tools/crawl.py) → data/corpus/*.txt
    ↓
manifest (tools/build_manifest.py) → docs.yaml
    ↓
pack_corpus:
  [ner_tag.py] → out/pack.ner.jsonl (entities per sentence)
  [candidates.py] → out/pack.candidates.jsonl (entity pairs)
  [re_score.py + promoter_v1.onnx] → out/pack.scored.jsonl (ML scores)
    ↓
auto_train (tools/select_training_from_scored.py) → training/re_train.jsonl, training/re_dev.jsonl → retrain (training/train_re_sklearn.py)
    ↓
edges_from_pack (tools/promote_from_scored.py + thresholds) → out/edges.jsonl
    ↓
[export_ttl.py] → out/graph.ttl
    ↓
quality (tools/quality_report.py) → summary metrics
```

## New Components

### 1. Configuration Files

**config/predicates.yaml**
- Defines the predicate taxonomy (e.g., type, covered_by, headquartered_in, operates_in, subsidiary_of, member_of, part_of, provides, requires, prohibits, uses, none)
- Used by the ML model to classify relationships

**models/thresholds.json**
- Confidence thresholds per predicate
- Used to filter low-confidence predictions

### 2. NER & Candidate Generation

**groundkg/ner_tag.py**
- Runs spaCy's NER on input text
- **Pipeline components:**
  - `sentencizer`: Rule-based sentence segmentation (runs first)
  - `entity_ruler`: Pattern-based entity boost (loads from `training/ruler_patterns.jsonl`)
  - `ner`: Statistical NER model (spaCy transformer-based)
- Outputs: `{"doc_id", "sent_idx", "text", "entities": [...]}`
- Entities include: ORG, PRODUCT, PERSON, GPE, LAW, ROLE, EVENT, etc.
- Supports custom entity labels via EntityRuler patterns

**groundkg/candidates.py**
- Pairs entities within the same sentence
- Filters by max character distance (150 chars)
- Prioritizes certain entity types for subjects vs objects
- Outputs: candidate (subject, object) pairs per sentence

### 3. Model Inference / Scoring

**groundkg/re_score.py**
- Loads ONNX model (`models/promoter_v1.onnx`) and emits per-pair predictions with probabilities (no thresholding).

**tools/promote_from_scored.py**
- Converts scored predictions to final edges using per-class thresholds and deduplication.

### 4. Training / Self-Training

**tools/select_training_from_scored.py**
- Selects high-confidence positives and robust negatives from `out/pack.scored.jsonl`.
- Supports minimum-per-class backfill and integration with mined patterns.

**training/train_re_sklearn.py**
- Trains TF-IDF + Logistic Regression, exports ONNX, writes thresholds and classes.

## Makefile Targets

### `make pipeline`
End-to-end: crawl → manifest → pack → auto-train → repack → stats → edges → quality

### `make hash`
Checksums output files for reproducibility

### `make verify`
Validates `out/edges.jsonl` JSONL

## Migration Path

### Typical Run
```bash
make pipeline
make pack_stats
make lint
```

## Training (TODO)

The next step is to create `training/train_re_sklearn.py`:
1. Load `training/re_train.jsonl` and `training/re_dev.jsonl`
2. Featurize text (TF-IDF or embeddings)
3. Train a classifier (Logistic Regression, SVM, or small neural network)
4. Export to ONNX using `skl2onnx`
5. Save as `models/promoter_v1.onnx`
6. Tune `models/thresholds.json` based on dev set precision/recall

## Benefits of This Architecture

✅ **Backward Compatible**: Existing rule-based pipeline still works
✅ **Deterministic Bootstrap**: Rules provide high-precision labels for training
✅ **Incremental Migration**: Can switch to ML without rewriting everything
✅ **Fallback Safety**: If model fails or is unavailable, rules still work
✅ **Reproducible**: Pinned dependencies, checksum validation
✅ **Extensible**: Easy to add new predicates to config and retrain

## Dependencies

Core dependencies in `requirement.txt` include:
- `onnxruntime` - Model inference
- `scikit-learn` - Model training
- `skl2onnx` - Model export
- spaCy + spacy-transformers - NER

Install with:
```bash
pip install -r requirement.txt
```

