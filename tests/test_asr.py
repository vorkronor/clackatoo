"""Integration test: transcribe the real private recording with Parakeet."""

from __future__ import annotations

from pathlib import Path

import pytest

from parakeet_transcribe.asr import ParakeetASR

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def asr() -> ParakeetASR:
    return ParakeetASR()


def test_transcribe_produces_text_and_word_timestamps(asr: ParakeetASR, test_audio_wav: Path):
    result = asr.transcribe(test_audio_wav)

    assert result.text.strip()
    assert result.words, "expected word-level timestamps"
    assert all(w.end >= w.start for w in result.words)
    assert result.words == sorted(result.words, key=lambda w: w.start)

    # The recording switches from Czech to English partway through and says
    # "English" explicitly - a cheap sanity check that decoding actually ran
    # rather than e.g. silently transcribing dead air.
    assert "english" in result.text.lower()
