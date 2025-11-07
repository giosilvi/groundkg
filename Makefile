PY=python
# Self-training knobs (overridable):
POS_THR ?= 0.95
NEG_THR ?= 0.95
MAX_PER_CLASS ?= 500
NER_MODEL ?= en_core_web_trf
OUT=out

.PHONY: setup clean model_promote collect pipeline auto_train pack_corpus pack_stats crawl manifest quality lint

setup:
	$(PY) -m spacy download en_core_web_sm

demo:
	@echo "Deprecated: use 'make pipeline'" ; exit 1

demo_legacy:
	@echo "Deprecated: model-only pipeline in use" ; exit 1

model_promote:
	@echo "Deprecated: models-only mode is enforced. Use 'make demo'." ; exit 1

collect:
	$(PY) training/mk_training_data.py

clean:
	rm -rf $(OUT)

hash:
	@shasum out/edges.jsonl out/attributes.jsonl out/graph.ttl || true

verify:
	@$(PY) -c "import json; [json.loads(l) for l in open('out/edges.jsonl')]; print('edges.jsonl OK')"

crawl:
	$(PY) tools/crawl.py

manifest:
	$(PY) tools/build_manifest.py

clean_pack:
	rm -f $(OUT)/pack.ner.jsonl $(OUT)/pack.candidates.jsonl $(OUT)/pack.scored.jsonl

pack_corpus: clean_pack
	@set -e; \
	mkdir -p $(OUT); \
	for f in $$(ls -1 data/corpus/*.txt 2>/dev/null | sort); do \
	  bn=$$(basename $$f .txt); \
	  echo "→ NER $$bn"; \
	  $(PY) -m groundkg.ner_tag $$f --doc-id $$bn --model $(NER_MODEL) >> $(OUT)/pack.ner.jsonl; \
	done; \
	echo "→ Candidates"; \
	$(PY) -m groundkg.candidates $(OUT)/pack.ner.jsonl > $(OUT)/pack.candidates.jsonl; \
	if [ ! -f models/promoter_v1.onnx ]; then echo "ERROR: train model first"; exit 2; fi; \
	echo "→ RE scoring"; \
	$(PY) -m groundkg.re_score $(OUT)/pack.candidates.jsonl models/promoter_v1.onnx models/classes.json > $(OUT)/pack.scored.jsonl; \
	echo "Done pack_corpus."

pack_stats:
	@echo "Pred counts in pack:"; \
	jq -r '.pred' $(OUT)/pack.scored.jsonl | sort | uniq -c | sort -nr | sed 's/^/  /'

auto_train:
	@echo "Selecting training data from out/pack.scored.jsonl (POS_THR=$(POS_THR), NEG_THR=$(NEG_THR), MAX_PER_CLASS=$(MAX_PER_CLASS))";
	@POS_THR=$(POS_THR) NEG_THR=$(NEG_THR) GK_POS_THR=$(POS_THR) GK_NEG_THR=$(NEG_THR) GK_MAX_PER_CLASS=$(MAX_PER_CLASS) \
		$(PY) tools/select_training_from_scored.py out/pack.scored.jsonl;
	@if [ -s training/re_train.jsonl ]; then \
	  echo "Training examples found → retraining model..."; \
	  $(PY) training/train_re_sklearn.py; \
	  echo "Retrained model from corpus pack."; \
	else \
	  echo "No selections (training/re_train.jsonl empty). Skipping retrain."; \
	fi

# End-to-end: crawl → manifest → pack → auto-train → repack → stats → edges
pipeline: clean crawl manifest pack_corpus auto_train pack_corpus pack_stats edges_from_pack verify hash quality
	@echo "Pipeline complete."

quality:
	@echo "Quality indicators:"; \
	$(PY) tools/quality_report.py out/pack.scored.jsonl out/edges.jsonl training/re_train.jsonl models/thresholds.json

edges_from_pack:
	@echo "Promoting pack.scored.jsonl to edges.jsonl using thresholds..."; \
	$(PY) tools/promote_from_scored.py out/pack.scored.jsonl models/thresholds.json | $(PY) -m groundkg.dedupe_edges /dev/stdin > out/edges.jsonl; \
	$(PY) -m groundkg.export_ttl out/edges.jsonl > out/graph.ttl

lint:
	@echo "Running Ruff (unused imports/vars)..."; \
	ruff check groundkg tools training --select F401,F841 --fix || true; \
	echo "\nRunning Vulture (unused defs)..."; \
	vulture groundkg tools training vulture_whitelist.py --min-confidence 80 --exclude out,data,models,.venv,__pycache__ || true

mine_patterns:
	@echo "Top surface patterns from scored (high-conf):"; \
	$(PY) tools/mine_patterns.py --scored $(OUT)/pack.scored.jsonl --min-count 3 --min-prob 0.9 | head -50; \
	echo "\nTop surface patterns from raw candidates:"; \
	$(PY) tools/mine_patterns.py --candidates $(OUT)/pack.candidates.jsonl --min-count 10 | head -50