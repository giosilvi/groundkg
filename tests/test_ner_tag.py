import io
import json

import pytest

from groundkg import ner_tag


class FakeEntity:
    def __init__(self, text, start_char, end_char, label):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.label_ = label


class FakeSentence:
    def __init__(self, text, start, start_char):
        self.text = text
        self.start = start
        self.start_char = start_char
        self.end_char = start_char + len(text)


class FakeDoc:
    def __init__(self, sents, ents):
        self.sents = sents
        self.ents = ents


class FakeNLP:
    def __init__(self, doc):
        self._doc = doc
        self.enabled_pipes = []

    def enable_pipe(self, name):
        self.enabled_pipes.append(name)

    def __call__(self, text):
        return self._doc


@pytest.fixture
def fake_doc():
    sent_text = "Alice lives in Paris."
    sentence = FakeSentence(sent_text, start=0, start_char=0)
    ents = [
        FakeEntity("Alice", 0, 5, "PERSON"),
        FakeEntity("Paris", 15, 20, "GPE"),
    ]
    return FakeDoc([sentence], ents)


def test_main_streams_sentence_entities(tmp_path, monkeypatch, fake_doc):
    text_path = tmp_path / "doc.txt"
    text_path.write_text("Alice lives in Paris.", encoding="utf-8")

    fake_nlp = FakeNLP(fake_doc)

    def fake_load(model, disable):
        assert model == "en_core_web_trf"
        assert "textcat" in disable
        return fake_nlp

    monkeypatch.setattr(ner_tag.spacy, "load", fake_load)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ner_tag.py",
            str(text_path),
            "--doc-id",
            "doc-1",
            "--model",
            "en_core_web_trf",
        ],
    )

    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)

    ner_tag.main()

    lines = [json.loads(line) for line in buf.getvalue().splitlines() if line]
    assert len(lines) == 1
    record = lines[0]
    assert record["doc_id"] == "doc-1"
    assert record["sent_idx"] == 0
    assert record["text"] == "Alice lives in Paris."
    assert record["entities"] == [
        {"text": "Alice", "start": 0, "end": 5, "label": "PERSON"},
        {"text": "Paris", "start": 15, "end": 20, "label": "GPE"},
    ]
    assert fake_nlp.enabled_pipes == ["ner"]


def test_main_defaults_doc_id_and_handles_multiple_sentences(tmp_path, monkeypatch):
    text = "Alice meets Bob. Charlie visits Rome."
    text_path = tmp_path / "doc.txt"
    text_path.write_text(text, encoding="utf-8")

    sentences = [
        FakeSentence("Alice meets Bob.", start=0, start_char=0),
        FakeSentence("Charlie visits Rome.", start=3, start_char=17),
    ]
    ents = [
        FakeEntity("Alice", 0, 5, "PERSON"),
        FakeEntity("Charlie", 17, 24, "PERSON"),
        FakeEntity("Rome", 32, 36, "GPE"),
    ]
    fake_doc = FakeDoc(sentences, ents)
    fake_nlp = FakeNLP(fake_doc)

    def fake_load(model, disable):
        assert model == "en_core_web_trf"
        return fake_nlp

    monkeypatch.setattr(ner_tag.spacy, "load", fake_load)
    monkeypatch.setattr("sys.argv", ["ner_tag.py", str(text_path)])

    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)

    ner_tag.main()

    lines = [json.loads(line) for line in buf.getvalue().splitlines() if line]
    assert [rec["doc_id"] for rec in lines] == ["doc", "doc"]
    assert [rec["text"] for rec in lines] == ["Alice meets Bob.", "Charlie visits Rome."]
    # Ensure entity offsets are relative to each sentence
    second_entities = lines[1]["entities"]
    assert any(ent == {"text": "Charlie", "start": 0, "end": 7, "label": "PERSON"} for ent in second_entities)
    assert any(ent == {"text": "Rome", "start": 15, "end": 19, "label": "GPE"} for ent in second_entities)
