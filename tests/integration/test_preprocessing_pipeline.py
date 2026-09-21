"""Integration tests for the full ArabicTextPipeline.

Marked ``slow`` because they load the CAMeL MLE model. Run with::

    pytest -m slow
"""

import pytest

from arnlp.preprocessing import ArabicTextPipeline, PipelineConfig


@pytest.fixture(scope="module")
def pipe() -> ArabicTextPipeline:
    return ArabicTextPipeline(PipelineConfig(lemmatizer="mle"))


@pytest.mark.slow
def test_handles_empty_input(pipe):
    doc = pipe.process("")
    assert doc.tokens == [] and doc.lemmas == []


@pytest.mark.slow
def test_decimal_preserved_in_embedding_view(pipe):
    doc = pipe.process("ألزمت الشركة بدفع 1.7 مليون يورو")
    assert "1.7" in doc.embedding_view


@pytest.mark.slow
def test_clustering_view_has_placeholders(pipe):
    doc = pipe.process("في 2024 ارتفعت الأسعار 5%")
    assert "YEAR_TOKEN" in doc.clustering_view
    assert "PCT_TOKEN"  in doc.clustering_view


@pytest.mark.slow
def test_filtered_tokens_aligned_with_lemmas(pipe, sample_articles):
    for art in sample_articles:
        doc = pipe.process(f"{art['title']} {art['body']}")
        assert len(doc.filtered_tokens) == len(doc.lemmas)


@pytest.mark.slow
def test_no_diacritics_in_lemmas(pipe, sample_articles):
    import re
    diac = re.compile(r"[ً-ْٰ]")
    for art in sample_articles:
        doc = pipe.process(f"{art['title']} {art['body']}")
        for lem in doc.lemmas:
            assert not diac.search(lem)


@pytest.mark.slow
def test_dates_extracted(pipe, sample_articles):
    dates_article = next(a for a in sample_articles if a["url"] == "test://dates")
    doc = pipe.process(f"{dates_article['title']} {dates_article['body']}")
    iso_set = {d.iso for d in doc.dates}
    assert "2024-02-28" in iso_set
    assert "2023-10" in iso_set
    assert "1948" in iso_set
