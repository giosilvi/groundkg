# groundkg/ner_tag.py
import sys, json, spacy

def main():
    in_path = sys.argv[1]
    doc_id = "doc"; model = "en_core_web_trf"
    for i,a in enumerate(sys.argv):
        if a == "--doc-id" and i+1 < len(sys.argv): doc_id = sys.argv[i+1]
        if a == "--model" and i+1 < len(sys.argv): model = sys.argv[i+1]

    # Enable NER + senter; parser not needed here
    nlp = spacy.load(model, disable=["textcat"])
    nlp.enable_pipe("ner")
    text = open(in_path, "r", encoding="utf-8").read()
    doc = nlp(text)

    for sent in doc.sents:
        ents = []
        for ent in doc.ents:
            if ent.start_char >= sent.start_char and ent.end_char <= sent.end_char:
                ents.append({
                    "text": ent.text,
                    "start": ent.start_char - sent.start_char,  # relative to sentence
                    "end": ent.end_char - sent.start_char,
                    "label": ent.label_
                })
        rec = {
            "doc_id": doc_id,
            "sent_idx": sent.start,
            "sent_start": sent.start_char,     # document-relative
            "text": sent.text.strip(),
            "entities": ents
        }
        sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

