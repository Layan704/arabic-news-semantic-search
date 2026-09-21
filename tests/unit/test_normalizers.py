"""Unit tests for the normalization helpers in ``arnlp.preprocessing``."""

from arnlp.preprocessing import (
    PLACEHOLDER_TOKENS,
    normalize_digits,
    normalize_for_clustering,
    normalize_for_embedding,
)


class TestDigitNormalization:
    def test_arabic_indic_to_ascii(self):
        assert normalize_digits("١٢٣٤٥") == "12345"

    def test_eastern_arabic_indic_to_ascii(self):
        assert normalize_digits("۱۲۳") == "123"

    def test_passthrough_for_ascii(self):
        assert normalize_digits("hello 42") == "hello 42"


class TestEmbeddingView:
    def test_decimal_preserved(self):
        out = normalize_for_embedding("بدفع 1.7 مليون")
        assert "1.7" in out

    def test_thousands_separator_preserved(self):
        out = normalize_for_embedding("سعر 1,500 دولار")
        assert "1,500" in out

    def test_trailing_dot_stripped(self):
        out = normalize_for_embedding("جملة.")
        assert "." not in out

    def test_percent_spelled_out(self):
        out = normalize_for_embedding("نمو 5%")
        assert "بالمئة" in out
        assert "%" not in out

    def test_latin_preserved(self):
        out = normalize_for_embedding("Claude Opus نموذج")
        assert "Claude" in out and "Opus" in out

    def test_no_placeholders(self):
        out = normalize_for_embedding("في 2024 و 50%")
        assert not any(p in out for p in PLACEHOLDER_TOKENS)


class TestClusteringView:
    def test_slash_split_to_space(self):
        # Arabic month pair should split, not merge
        out = normalize_for_clustering("أبريل/نيسان 2024")
        assert "ابريل" in out.split()
        assert "نيسان" in out.split()
        assert "ابريلنيسان" not in out

    def test_placeholders_preserved(self):
        out = normalize_for_clustering("في YEAR_TOKEN حصل NUM_TOKEN")
        assert "YEAR_TOKEN" in out.split()
        assert "NUM_TOKEN" in out.split()

    def test_digits_dropped(self):
        out = normalize_for_clustering("في 2024 حصل")
        assert "2024" not in out

    def test_latin_dropped(self):
        out = normalize_for_clustering("نموذج Claude جديد")
        assert "Claude" not in out
