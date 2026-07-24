import pytest

pytest.importorskip("librosa")  # skip this module if requirements/ml.txt isn't installed

from afrimeet.data.preprocess import clean_transcript  # noqa: E402


def test_clean_transcript_lowercases_and_strips_punctuation():
    assert clean_transcript("Hello, World!!") == "hello world"


def test_clean_transcript_collapses_whitespace():
    assert clean_transcript("  hello   there  ") == "hello there"


def test_clean_transcript_keeps_apostrophes_and_hyphens():
    assert clean_transcript("it's a well-known fact.") == "it's a well-known fact"
