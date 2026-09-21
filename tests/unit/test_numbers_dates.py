"""Unit tests for the number/date helpers in ``arnlp.preprocessing``."""

from arnlp.preprocessing import extract_dates, mask_numbers


class TestExtractDates:
    def test_day_month_year(self):
        hits = extract_dates("في 15 يناير 2024 وقع")
        assert len(hits) == 1
        assert hits[0].iso == "2024-01-15"
        assert hits[0].kind == "dmy"

    def test_month_year(self):
        hits = extract_dates("نيسان 2024 كان")
        assert len(hits) == 1
        assert hits[0].iso == "2024-04"
        assert hits[0].kind == "my"

    def test_bare_year(self):
        hits = extract_dates("منذ عام 1948 قامت")
        assert len(hits) == 1
        assert hits[0].iso == "1948"
        assert hits[0].kind == "year"

    def test_levantine_month_recognized(self):
        hits = extract_dates("في تشرين الأول 2023")
        assert any(h.iso == "2023-10" for h in hits)

    def test_no_overlap_dmy_vs_year(self):
        # "15 يناير 2024" should yield ONE dmy hit, not also a separate year hit
        hits = extract_dates("في 15 يناير 2024 وقع")
        assert len(hits) == 1


class TestMaskNumbers:
    def test_year_masked(self):
        assert "YEAR_TOKEN" in mask_numbers("في 2024 حصل")
        assert "2024" not in mask_numbers("في 2024 حصل")

    def test_percent_masked(self):
        assert "PCT_TOKEN" in mask_numbers("نمو 5%")

    def test_money_masked(self):
        assert "MONEY_TOKEN" in mask_numbers("بقيمة 100 دولار")

    def test_generic_number_masked(self):
        assert "NUM_TOKEN" in mask_numbers("بعد 15 سنة")

    def test_specificity_order(self):
        # year should win over generic NUM
        out = mask_numbers("في 2024")
        assert "YEAR_TOKEN" in out
        assert "NUM_TOKEN" not in out
