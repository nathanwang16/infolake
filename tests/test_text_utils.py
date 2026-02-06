"""Tests for common/text_utils.py — excerpt extraction and text cleaning."""

import pytest

from common.text_utils import extract_excerpt, clean_text_for_display, _is_header_like


class TestExtractExcerpt:
    def test_short_text_returns_none(self):
        assert extract_excerpt("Too short") is None
        assert extract_excerpt("") is None
        assert extract_excerpt(None) is None

    def test_returns_first_paragraph(self):
        text = (
            "HEADER LINE\n\n"
            "This is the first real paragraph with enough content to be considered meaningful "
            "and it should be extracted as the excerpt for this document.\n\n"
            "This is the second paragraph."
        )
        excerpt = extract_excerpt(text)
        assert excerpt is not None
        assert "first real paragraph" in excerpt

    def test_fallback_to_word_limit(self):
        text = " ".join(["word"] * 200)
        excerpt = extract_excerpt(text, max_words=50)
        assert len(excerpt.split()) <= 51  # +1 for possible partial word with ellipsis

    def test_adds_ellipsis_when_truncated(self):
        text = " ".join(["word"] * 200)
        excerpt = extract_excerpt(text, max_words=50)
        assert excerpt.endswith("...")

    def test_no_ellipsis_for_complete_text(self):
        text = "A" * 60  # Just above MIN_EXCERPT_LENGTH
        excerpt = extract_excerpt(text)
        assert excerpt is not None
        assert not excerpt.endswith("...")


class TestIsHeaderLike:
    def test_all_caps(self):
        assert _is_header_like("INTRODUCTION") is True

    def test_ends_with_colon(self):
        assert _is_header_like("Section One:") is True

    def test_short_text(self):
        assert _is_header_like("Hi") is True

    def test_numbered_item(self):
        assert _is_header_like("1. First point in the list") is True

    def test_normal_paragraph(self):
        assert _is_header_like(
            "This is a normal paragraph with enough content to be meaningful."
        ) is False


class TestCleanTextForDisplay:
    def test_removes_control_chars(self):
        text = "hello\x00world\x07test"
        result = clean_text_for_display(text)
        assert "\x00" not in result
        assert "\x07" not in result
        assert result == "helloworldtest"

    def test_normalizes_whitespace(self):
        text = "hello   \t\t  world"
        assert clean_text_for_display(text) == "hello world"

    def test_limits_newlines(self):
        text = "line1\n\n\n\n\nline2"
        assert clean_text_for_display(text) == "line1\n\nline2"

    def test_empty_string(self):
        assert clean_text_for_display("") == ""
        assert clean_text_for_display(None) == ""
