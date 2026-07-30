import pytest

pytest.importorskip("librosa")  # skip this module if requirements/ml.txt isn't installed

from afrimeet.data.preprocess import clean_transcript  # noqa: E402


def test_clean_transcript_preserves_casing_and_punctuation():
    # The model should learn to produce punctuation/casing, so training targets must
    # actually contain it -- clean_transcript() must not strip it away.
    assert clean_transcript("Hello, World!!") == "Hello, World!!"


def test_clean_transcript_collapses_whitespace():
    assert clean_transcript("  hello   there  ") == "hello there"


def test_clean_transcript_strips_leading_and_trailing_whitespace():
    assert clean_transcript("  It's a well-known fact.  ") == "It's a well-known fact."
