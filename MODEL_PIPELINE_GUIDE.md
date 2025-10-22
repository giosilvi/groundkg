# Model-Based Relation Extraction Pipeline - Complete Guide

## ✅ Implementation Complete

This guide documents the **model-based relation extraction pipeline** that has been successfully implemented and tested.

## Architecture Overview

### Model-Only Operation

The pipeline runs purely model-based at runtime:
- Uses spaCy NER to extract entities
- Generates candidate entity pairs (+ noun chunks)
- Scores relations with the ONNX model
- Applies per-predicate thresholds to emit final edges

## Pipeline Flow

```
crawl → data/corpus/*.txt
pack_corpus → out/pack.ner.jsonl → out/pack.candidates.jsonl → out/pack.scored.jsonl
auto_train + retrain → models/*
edges_from_pack → out/edges.jsonl → graph.ttl
quality → quality indicators
```

## Key Components

### 1. Configuration Files

**`config/predicates.yaml`**
- Defines 6 predicate types: `type`, `covered_by`, `headquartered_in`, `consists_of`, `performs_task`, `none`

**`models/thresholds.json`**
- Per-predicate confidence thresholds
- Currently set to 0.2 for development (adjust to 0.8+ for production)

**`models/classes.json`**
- List of class labels from training
- Auto-generated during training

### 2. NER & Candidate Generation

**`groundkg/ner_tag.py`**
- Runs spaCy's pre-trained NER on input text
- Extracts entities: ORG, PERSON, GPE, LAW, PRODUCT, etc.
- Outputs sentence-level entity annotations

**`groundkg/candidates.py`**
- Pairs entities within sentences
- Adds noun phrase chunks (regex-based) to catch missed entities
- Filters by distance (max 150 chars between entities)
- Outputs: `(subject, object)` candidate pairs

### 3. Model Training

Deprecated: `training/mk_training_data.py` (rule-based bootstrap) has been removed.

**`training/train_re_sklearn.py`**
- Trains TF-IDF + Logistic Regression model
- Exports to ONNX format for inference
- Auto-tunes thresholds using precision-recall curves
- Outputs: `promoter_v1.onnx`, `thresholds.json`, `classes.json`

### 4. Model Inference

**`groundkg/re_score.py`**
- Loads ONNX model and outputs per-pair predictions with probabilities.
**`tools/promote_from_scored.py`**
- Applies per-class thresholds and dedupes to produce final edges.

## How to Use

### Initial Setup (One-Time)

```bash
# Install dependencies
pip install -r requirement.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Typical Run

```bash
make pipeline
make pack_stats
make lint
```

## Example Output

### Model-Based Edges (`out/edges.jsonl`)
```json
{"subject": "Tesla", "predicate": "headquartered_in", "object": "Austin", "evidence": {...}}
{"subject": "Tesla", "predicate": "headquartered_in", "object": "Texas", "evidence": {...}}
```

### Knowledge Graph (`out/graph.ttl`)
```turtle
ex:node/Tesla ex:headquartered_in ex:node/Austin .
ex:node/Tesla ex:headquartered_in ex:node/Texas .
```

## Training Data Format

**Input**: `training/re_train.jsonl`
```json
{"text": "[E1]Tesla[/E1] is headquartered in [E2]Austin, Texas[/E2].", "label": "headquartered_in"}
{"text": "[E1]ACME[/E1] is [E2]a Digital Service Provider[/E2].", "label": "type"}
{"text": "[E1]System[/E1] consists of [E2]three components[/E2].", "label": "consists_of"}
{"text": "According to [E1]the vendor[/E1], [E2]the model[/E2] may achieve 92% accuracy.", "label": "none"}
```

## Current Performance

See the Quality indicators printed at the end of `make pipeline`.

## Advantages Over Rule-Based

✅ **No Hard-Coded Patterns**: Model learns from examples
✅ **Generalizes Better**: Works on unseen sentence structures
✅ **Extensible**: Add new predicates by adding training examples
✅ **Probabilistic**: Confidence scores for filtering
✅ **Noun Chunk Support**: Catches multi-word entities NER misses

## Configuration Tuning

### Increasing Precision (Fewer False Positives)
Edit `models/thresholds.json`:
```json
{
  "type": 0.85,
  "headquartered_in": 0.90,
  "consists_of": 0.85,
  "performs_task": 0.85,
  "none": 1.00
}
```

### Adding New Predicates
1. Add to `config/predicates.yaml`
2. Add training examples to `training/re_train.jsonl`
3. Retrain: `python training/train_re_sklearn.py`

### Improving Recall (More Extractions)
- Lower thresholds in `models/thresholds.json`
- Add more positive training examples
- Increase `MAX_CHAR_DIST` in `candidates.py`

## Files Created

```
config/
  predicates.yaml          # Predicate taxonomy
models/
  promoter_v1.onnx         # Trained ONNX model
  thresholds.json          # Confidence thresholds
  classes.json             # Class labels
training/
  (deprecated) mk_training_data.py was removed
  train_re_sklearn.py      # Training script
  re_train.jsonl           # Training examples
  re_dev.jsonl             # Development examples
groundkg/
  ner_tag.py               # NER extraction
  candidates.py            # Candidate generation
  re_infer.py              # Model inference
out/
  ner.jsonl                # NER output
  candidates.jsonl         # Candidate pairs
  edges.jsonl              # Final edges
  graph.ttl                # RDF graph
```

## Next Steps

1. **Collect More Training Data**: Run on larger corpora to bootstrap more examples
2. **Refine Thresholds**: Tune per-predicate thresholds on dev set
3. **Experiment with Models**: Try transformers (BERT, RoBERTa) for better accuracy
4. **Active Learning**: Manually label uncertain predictions to improve model
5. **Domain Adaptation**: Fine-tune on domain-specific text

## Reproducibility

```bash
make demo
make hash
# Output hashes:
# 6e25b5c4cae5835cb6057d74043370246cd133a8  out/edges.jsonl
# 776d1b593cd9641b4f3214a9beb04591f454e398  out/statements.scored.jsonl
# 23b02cd22b7cb37b4064eaa74deb56826f3f1a24  out/attributes.jsonl
# be5768f91095360d98b9bd34169a766c231a1d3d  out/graph.ttl
```

## Success! 🎉

The model-based pipeline is fully operational and tested. The system automatically switches between rule-based and ML-based extraction, providing a smooth migration path from deterministic patterns to learned models.

