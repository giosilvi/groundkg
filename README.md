# GroundKG (v0, deterministic core)

From raw text to **statement cards** (who–did–what) with quotes and character offsets → **Graph-Readiness (GR)** scoring → **promoted edges** (only when safe) → optional **attributes** (numbers, units, dates) → minimal **RDF/Turtle** export.

**Deterministic by default**: same inputs ⇒ same outputs. No LLMs in v0.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

make demo        # run the full v0 pipeline on sample text