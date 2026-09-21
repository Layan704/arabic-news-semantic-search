"""Unit tests for ``arnlp.ambiguity`` (embedding-based WSD).

A fake encoder maps known texts to fixed vectors, so the ranking logic is
tested deterministically and without downloading the E5 model.
"""

import json

import numpy as np

from arnlp.ambiguity import (
    DEFAULT_SENSES,
    EmbeddingDisambiguator,
    Sense,
    SenseScore,
    load_sense_inventory,
)


class FakeEncoder:
    """Maps known strings to fixed vectors; L2-normalizes like E5."""

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def _vec(self, text: str) -> np.ndarray:
        v = np.asarray(self.mapping[text], dtype=np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts):
        return np.stack([self._vec(t) for t in texts])


def test_disambiguate_ranks_closest_sense_first():
    context = "ذهب الرجل إلى السوق"
    went = Sense("ذهب", "went", "فعل الذهاب والانتقال")
    gold = Sense("ذهب", "gold", "المعدن النفيس الأصفر")
    enc = FakeEncoder(
        {
            context: [1.0, 0.0],
            went.text: [1.0, 0.0],   # aligned with the context
            gold.text: [0.0, 1.0],   # orthogonal
        }
    )
    wsd = EmbeddingDisambiguator(encoder=enc)

    ranked = wsd.disambiguate(context, [gold, went])  # deliberately gold-first
    assert [s.sense.sense_id for s in ranked] == ["went", "gold"]
    assert ranked[0].score > ranked[1].score
    assert isinstance(ranked[0], SenseScore)


def test_best_sense_and_top_k():
    a = Sense("w", "a", "ga")
    b = Sense("w", "b", "gb")
    enc = FakeEncoder({"c": [1, 0], "ga": [0.9, 0.1], "gb": [-1, 0]})
    wsd = EmbeddingDisambiguator(encoder=enc)

    assert wsd.best_sense("c", [a, b]).sense.sense_id == "a"
    assert len(wsd.disambiguate("c", [a, b], top_k=1)) == 1


def test_empty_senses_returns_empty():
    wsd = EmbeddingDisambiguator(encoder=FakeEncoder({}))
    assert wsd.disambiguate("anything", []) == []
    assert wsd.best_sense("anything", []) is None


def test_default_inventory_is_well_formed():
    assert "عين" in DEFAULT_SENSES
    for word, senses in DEFAULT_SENSES.items():
        assert len(senses) >= 2                       # genuinely ambiguous
        ids = [s.sense_id for s in senses]
        assert len(ids) == len(set(ids))              # unique sense ids
        for s in senses:
            assert s.word == word and s.gloss


def test_disambiguate_word_uses_inventory():
    word = "عين"
    senses = DEFAULT_SENSES[word]
    context = "spring-context"
    mapping = {context: [1, 0]}
    for s in senses:
        mapping[s.text] = [1, 0] if s.sense_id == "spring" else [0, 1]
    wsd = EmbeddingDisambiguator(encoder=FakeEncoder(mapping))

    ranked = wsd.disambiguate_word(word, context)
    assert ranked[0].sense.sense_id == "spring"


def test_unknown_word_returns_empty():
    wsd = EmbeddingDisambiguator(encoder=FakeEncoder({}))
    assert wsd.disambiguate_word("لا_يوجد_هذا", "ctx") == []


def test_load_sense_inventory(tmp_path):
    p = tmp_path / "inv.json"
    p.write_text(
        json.dumps(
            {
                "بنك": [
                    {
                        "sense_id": "finance",
                        "gloss": "مؤسسة مالية",
                        "examples": ["أودع المال في البنك"],
                    },
                    {"sense_id": "bench", "gloss": "مقعد طويل"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    inv = load_sense_inventory(p)

    assert set(inv) == {"بنك"}
    senses = inv["بنك"]
    assert [s.sense_id for s in senses] == ["finance", "bench"]
    assert senses[0].examples == ("أودع المال في البنك",)
    assert senses[0].text == "مؤسسة مالية أودع المال في البنك"
