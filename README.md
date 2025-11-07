# GroundKG (v0, deterministic core)

From raw text to **statement cards** (who–did–what) with quotes and character offsets → **Graph-Readiness (GR)** scoring → **promoted edges** (only when safe) → optional **attributes** (numbers, units, dates) → minimal **RDF/Turtle** export.

**Deterministic by default**: same inputs ⇒ same outputs. No LLMs in v0.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf

make demo        # run the full v0 pipeline on sample text
# GroundKG (model‑only v0.1)

**From raw text to a small, cited knowledge graph — using models only.**  
Pipeline: **crawl → NER → candidate pairs → relation scoring (sentence transformers + ONNX) → thresholded edges → RDF/Turtle export → quality report**.

- **Models‑only at runtime.** No rule fallback in production.
- **Deterministic‑by‑default.** Same inputs + same seeds/config ⇒ same outputs (see *Reproducibility*).
- **Self‑training loop.** High‑confidence predictions from your corpus become new training data.
- **Standards‑ready.** Minimal RDF/Turtle export; JSONL edges with evidence spans.

---

## Quick start

```bash
# 1) Create venv and install deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirement.txt
python -m spacy download en_core_web_trf
# Note: sentence-transformers will download its model automatically on first use

# 2) (Optional) seed a tiny training set the first time
mkdir -p training
cat > training/re_train.jsonl <<'EOF'
{"text":"[E1]Tesla[/E1] is headquartered in [E2]Austin, Texas[/E2].","label":"headquartered_in"}
{"text":"[E1]ACME[/E1] is a [E2]Digital Service Provider[/E2].","label":"type"}
{"text":"[E1]Neural networks[/E1] are used for [E2]pattern recognition[/E2].","label":"performs_task"}
EOF
cp training/re_train.jsonl training/re_dev.jsonl

# 3) One‑button pipeline (crawl → pack → train → edges → graph)
make pipeline

# 4) Inspect outputs
make pack_stats
head -n 10 out/edges.jsonl | jq
cat out/graph.ttl
```

---

## What the pipeline does

```
crawl (tools/crawl.py) → data/corpus/*.txt
   ↓
manifest (tools/build_manifest.py) → docs.yaml
   ↓
pack_corpus:
  groundkg/ner_tag.py           → out/pack.ner.jsonl       # per‑sentence entities (spaCy NER + EntityRuler patterns)
  groundkg/candidates.py        → out/pack.candidates.jsonl# entity pairs (top‑10 / sent)
  groundkg/re_score.py + ONNX   → out/pack.scored.jsonl    # ML scores (sentence transformers + ONNX), no thresholds
   ↓
auto_train (tools/select_training_from_scored.py) → training/re_train.jsonl + re_dev.jsonl
training/train_re_transformers.py → models/promoter_v1.onnx, models/thresholds.json, models/classes.json
   ↓
edges_from_pack (tools/promote_from_scored.py)    → out/edges.jsonl   # apply per‑class thresholds + dedupe
export_ttl (groundkg/export_ttl.py)               → out/graph.ttl     # minimal RDF/Turtle
quality (tools/quality_report.py)                 → quality summary
```

## Automatic seed bootstrap (when no model exists)

If `models/promoter_v1.onnx` is missing, you can auto‑bootstrap a small, high‑precision seed from your corpus (no LLMs, works on any text):

```bash
# 1) Build NER and candidates (no model needed)
make -f Makefile.gk ner cand

# 2) Bootstrap seeds from clear, type‑gated connectors
make -f Makefile.gk bootstrap_seed
# or one‑shot helper: make -f Makefile.gk bootstrap
# Note: Automatically adds negative examples ("none" class) to ensure at least 2 classes

# 3) Train the first ONNX model from the seed
make -f Makefile.gk coldstart

# 4) Run the full pipeline
make -f Makefile.gk crawl manifest ner cand score infer edges ttl report
```

Later, improve via self‑training:

```bash
make -f Makefile.gk autoselect   # mine confident examples + retrain
```

Placeholders for future bootstraps (disabled by default):
- Dependency rules: `tools/bootstrap_seed_dep.py --enable`
- Distant supervision: `tools/bootstrap_seed_kb.py --kb config/kb.json`
- LLM‑assisted: `tools/bootstrap_seed_llm.py --enable` (add gating + audits)

**Events Pipeline (parallel, optional):**
```
events (groundkg/event_extract.py)        → out/events.jsonl       # regex-based event extraction
event_edges (groundkg/events_to_edges.py) → out/edges.events.jsonl # convert events to edges
ttl_events                                → graph.events.ttl       # export events graph
```

Event types: Acquisition, Funding, Appointment, Launch, Founding (extracted via regex patterns).

### Predicates (editable)
`config/predicates.yaml` defines the label space (e.g., `type`, `covered_by`, `headquartered_in`, `operates_in`, `subsidiary_of`, `parent_of`, `member_of`, `part_of`, `provides`, `requires`, `prohibits`, `uses`, `none`).

---

## Core commands

```bash
# End‑to‑end (idempotent)
make -f Makefile.gk all

# Process corpus only (NER → candidates → scores)
make -f Makefile.gk ner cand score

# Promote to edges + export graph
make -f Makefile.gk infer edges ttl

# Self‑train from scored pool and retrain
make -f Makefile.gk autoselect

# Demo on a tiny sample file
make demo DEMO=data/sample/demo_rich.txt

# Sanity checks / stats
make verify
make pack_stats
make metrics
make hash
```

Tuning knobs (CLI‑overridable):
- **`POS_THR` / `NEG_THR`**: selection thresholds for self‑training (defaults exported as `GK_POS_THR`, `GK_NEG_THR`).
- **`MAX_PER_CLASS`**: cap per class for balanced training batches.
- **`MAX_CHAR_DIST` / `MAX_PAIRS_PER_SENT`** in `groundkg/candidates.py` for pairing strictness.

---

## Command Reference

### Core Pipeline Commands

**`make -f Makefile.gk crawl`**
- Downloads documents from URLs in `data/seed.csv`
- Extracts text from HTML/PDF files
- Outputs: `data/raw/*`, `data/corpus/*.txt`, `data/meta.jsonl`

**`make -f Makefile.gk manifest`**
- Builds YAML manifest from metadata
- Outputs: `docs.yaml` (provenance information)

**`make -f Makefile.gk ner`**
- Extracts named entities using spaCy NER + EntityRuler patterns
- Processes all files in `data/corpus/*.txt`
- Outputs: `out/pack.ner.jsonl` (sentences with entities)

**`make -f Makefile.gk cand`**
- Generates candidate entity pairs from NER output
- Filters by distance and entity types
- Outputs: `out/pack.candidates.jsonl` (subject-object pairs)

**`make -f Makefile.gk score`**
- Scores candidate pairs using ONNX model
- Requires: `models/promoter_v1.onnx` and `models/classes.json`
- Outputs: `out/pack.scored.jsonl` (predictions with probabilities)

**`make -f Makefile.gk infer`**
- Promotes high-confidence predictions to edges
- Applies per-class thresholds from `models/thresholds.json`
- Outputs: `out/edges.jsonl` (final edges with evidence)

**`make -f Makefile.gk edges`**
- Deduplicates edges
- Outputs: `out/edges.dedup.jsonl`

**`make -f Makefile.gk ttl`**
- Exports edges to RDF/Turtle format
- Outputs: `out/graph.ttl`

**`make -f Makefile.gk report`**
- Generates quality report with statistics
- Shows prediction counts, edge counts, training distribution

### Seed Bootstrap Commands

**`make -f Makefile.gk bootstrap_seed`**
- Auto-generates training seeds from candidate pairs
- Scans for clear connector phrases (e.g., "is headquartered in", "subsidiary of")
- Type-gated for high precision
- Automatically adds negative examples ("none" class) to ensure at least 2 classes for training
- Outputs: `training/seed.jsonl`

**`make -f Makefile.gk bootstrap`**
- One-shot helper: runs `ner cand bootstrap_seed`
- Builds seeds from scratch (no model needed)

**`make -f Makefile.gk coldstart`**
- One-time bootstrap: creates train/dev split from seed
- Trains first ONNX model
- Outputs: `models/promoter_v1.onnx`, `models/thresholds.json`, `models/classes.json`
- Requires: `training/seed.jsonl`

### Self-Training Commands

**`make -f Makefile.gk autoselect`**
- Selects high-confidence predictions for retraining
- Uses thresholds: `GK_POS_THR=0.95`, `GK_NEG_THR=0.95` (configurable)
- Retrains model with expanded training data
- Outputs: Updated model files

**`make -f Makefile.gk rescore`**
- Rescores candidates after model retraining
- Alias for `score` (re-runs with updated model)

**`make -f Makefile.gk patterns`**
- Mines relation patterns from scored predictions
- Outputs: `out/patterns.jsonl` (for analysis)

### Events Pipeline Commands

**`make -f Makefile.gk events`**
- Extracts events using regex patterns
- Event types: Acquisition, Funding, Appointment, Launch, Founding
- Outputs: `out/events.jsonl`

**`make -f Makefile.gk event_edges`**
- Converts events to edge format
- Outputs: `out/edges.events.jsonl`

**`make -f Makefile.gk ttl_events`**
- Exports events to RDF/Turtle
- Outputs: `graph.events.ttl`

**`make -f Makefile.gk merge_edges`**
- Merges core edges + event edges
- Outputs: `out/edges.merged.jsonl`

**`make -f Makefile.gk ttl_merged`**
- Exports merged edges to RDF/Turtle
- Outputs: `graph.merged.ttl`

### Utility Commands

**`make -f Makefile.gk all`**
- Runs full pipeline: `crawl manifest ner cand score infer edges ttl report`

**`make -f Makefile.gk clean`**
- Removes all output files and models
- Cleans: `out/`, `models/`, `training/re_*.jsonl`

---

## Repo layout

```
config/               # predicates, misc config
data/                 # seed.csv, raw/, corpus/, meta.jsonl
groundkg/             # NER, candidates, scoring, export
models/               # promoter_v1.onnx, thresholds.json, classes.json
tools/                # crawl, build_manifest, select_training_from_scored, quality_report
training/             # re_train.jsonl, re_dev.jsonl, train_re_sklearn.py
out/                  # pack.ner.jsonl, pack.candidates.jsonl, pack.scored.jsonl, edges.jsonl, graph.ttl
```

---

## Reproducibility

We aim for **bit‑stable** results when the inputs and environment are fixed:

- **Seeds & sorting.** We fix RNG seeds and use deterministic tie‑breakers.
- **Pinned deps.** Use the provided `requirements.txt` and record versions in `models/metrics.json`.
- **Stable configs.** Per‑predicate thresholds live in `models/thresholds.json`.
- **Hashing.** `make hash` prints checksums for key artifacts.

⚠️ Notes:
- Crawling the live web can introduce variance (content changes, 404s). For paper‑trail runs, freeze `data/raw/` and `data/corpus/` and commit `docs.yaml`.
- spaCy’s small model is CPU‑friendly but not perfect; better NER = better pairs.

---

## Limitations & roadmap

**Current limitations**
- English‑first (spaCy `en_core_web_trf`); multilingual models are WIP.
- Relation model = Sentence Transformers (`all-MiniLM-L6-v2`) + Logistic Regression (good balance of quality and speed).
- Attributes extraction is regex‑based; not yet exported as RDF with units/datatypes.
- No viewer UI yet (graph is exported to Turtle; JSONL edges include evidence).

**Near‑term roadmap**
- Transformer RE baseline (DistilBERT/RoBERTa) with ONNX export.
- Multilingual NER/RE pack.
- Attributes → RDF with units and provenance.
- SHACL shapes & validation.
- Minimal viewer: Answer → Path → Quotes in ≤3 clicks.

---

## Ethics & licensing

- Respect `robots.txt`, licenses, and fair‑use for quotes/snippets.
- Keep evidence spans short (≤ 300 chars) with character offsets.
- See `docs.yaml` for per‑document provenance.

---

## Why this exists (non‑technical)

GroundKG turns prose into a **small, explainable graph**: you can click from an answer to the **exact quote** that supports it. That makes findings easier to defend in audits, board packs, and research notes — without sending your documents to a third‑party service.

---

Happy hacking! Questions & ideas welcome via issues.