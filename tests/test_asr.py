"""Integration test: transcribe the real private recording with Parakeet."""

from __future__ import annotations

from pathlib import Path

import pytest

from parakeet_transcribe.audio import get_duration_seconds
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


def test_transcribe_long_form_matches_single_shot_when_chunked(
    asr: ParakeetASR, test_audio_wav: Path, tmp_path: Path
):
    # A small chunk_minutes forces even this short sample to split into a couple of
    # chunks, exercising the silence-detection + stitching path end to end.
    duration = get_duration_seconds(test_audio_wav)
    single = asr.transcribe(test_audio_wav)
    chunked = asr.transcribe_long_form(
        test_audio_wav, duration_sec=duration, chunk_minutes=0.15, tmp_dir=tmp_path
    )

    assert chunked.words, "expected word-level timestamps"
    assert chunked.words == sorted(chunked.words, key=lambda w: w.start)
    assert "english" in chunked.text.lower()

    # Decoding right at a chunk boundary can differ slightly from a single pass
    # (less surrounding context), so this isn't an exact match - just a sanity
    # check that stitching hasn't dropped or duplicated a chunk's worth of audio.
    assert 0.7 <= len(chunked.words) / len(single.words) <= 1.3


def test_transcribe_long_form_skips_chunking_below_threshold(
    asr: ParakeetASR, test_audio_wav: Path, tmp_path: Path
):
    result = asr.transcribe_long_form(
        test_audio_wav, duration_sec=5.0, chunk_minutes=2.0, tmp_dir=tmp_path
    )
    assert not (tmp_path / f"{test_audio_wav.stem}_asr_chunks").exists()
    assert result.text.strip()
