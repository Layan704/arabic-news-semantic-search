"""Unit tests for the stopword helpers in ``arnlp.preprocessing``."""

from arnlp.preprocessing import (
    BASE_STOPWORDS,
    CLITIC_PREFIXES,
    LEVANTINE_MONTHS,
    build_stopwords,
)


def test_clitic_expansion():
    sw = build_stopwords()
    for word in ["في", "من", "كان", "قد"]:
        for prefix in CLITIC_PREFIXES:
            assert (prefix + word) in sw, f"{prefix + word} should be a stopword"


def test_levantine_months_included():
    sw = build_stopwords()
    for m in ["نيسان", "شباط", "تشرين", "كانون"]:
        assert m in sw


def test_extra_stopwords():
    sw = build_stopwords(extra=["xyzword"])
    assert "xyzword" in sw


def test_base_set_subset():
    sw = build_stopwords()
    assert BASE_STOPWORDS.issubset(sw)
    assert LEVANTINE_MONTHS.issubset(sw)
