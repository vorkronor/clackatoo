"""Integration test: diarize the real private recording with pyannote.audio."""

from __future__ import annotations

from pathlib import Path

import pytest
from huggingface_hub.errors import HfHubHTTPError

from parakeet_transcribe.cli import resolve_hf_token
from parakeet_transcribe.diarize import Diarizer

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def diarizer() -> Diarizer:
    token = resolve_hf_token(None)
    if not token:
        pytest.skip("No Hugging Face token available (run `hf auth login` or set HF_TOKEN)")
    try:
        return Diarizer(hf_token=token)
    except HfHubHTTPError as e:
        pytest.skip(
            "Could not load the diarization pipeline - likely a gated model whose "
            f"terms haven't been accepted on huggingface.co yet: {e}"
        )


def test_diarize_finds_at_least_one_speaker_turn(diarizer: Diarizer, test_audio_wav: Path):
    turns = diarizer.diarize(test_audio_wav)

    assert turns
    assert all(t.end > t.start for t in turns)
