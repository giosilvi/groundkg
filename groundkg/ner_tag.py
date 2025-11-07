# groundkg/ner_tag.py
import sys
import json
import warnings

# Suppress thinc FutureWarnings about torch.cuda.amp.autocast deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="thinc")

import spacy


def main():
    in_path = sys.argv[1]
    doc_id = "doc"
    model = "en_core_web_trf"
    for i, a in enumerate(sys.argv):
        if a == "--doc-id" and i + 1 < len(sys.argv):
            doc_id = sys.argv[i + 1]
        if a == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]

    # Enable NER + sentence boundaries; parser not needed here
    nlp = spacy.load(model, disable=["textcat"])
    # 1) sentence boundaries first
    pipe_names = list(getattr(nlp, "pipe_names", []))
    if hasattr(nlp, "add_pipe"):
        if "sentencizer" not in pipe_names and "senter" not in pipe_names:
            try:
                nlp.add_pipe("sentencizer", first=True)
            except Exception:
                pass
        # 2) lightweight patterns before statistical NER
        if "entity_ruler" not in pipe_names:
            try:
                ruler = (
                    nlp.add_pipe("entity_ruler", before="ner")
                    if "ner" in pipe_names
                    else nlp.add_pipe("entity_ruler")
                )
                try:
                    ruler.from_disk("training/ruler_patterns.jsonl")
                except Exception:
                    pass  # ok if file missing in some environments
            except Exception:
                pass
    if hasattr(nlp, "enable_pipe"):
        try:
            nlp.enable_pipe("ner")
        except Exception:
            pass
    text = open(in_path, "r", encoding="utf-8").read()
    doc = nlp(text) # main call to the pipeline

    for sent in doc.sents:
        ents = []
        for ent in doc.ents:
            if ent.start_char >= sent.start_char and ent.end_char <= sent.end_char:
                ents.append(
                    {
                        "text": ent.text,
                        "start": ent.start_char
                        - sent.start_char,  # relative to sentence
                        "end": ent.end_char - sent.start_char,
                        "label": ent.label_,
                    }
                )
        rec = {
            "doc_id": doc_id,
            "sent_idx": sent.start,
            "sent_start": sent.start_char,  # document-relative
            "text": sent.text.strip(),
            "entities": ents,
        }
        sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
