"""Shared fixtures.

The integration tests exercise the real pipeline against a private recording
that is deliberately kept out of version control (see .gitignore). Anyone
without that file gets those tests skipped rather than failed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_AUDIO_PATH = REPO_ROOT / "_test-data" / "my voice.aifc"


@pytest.fixture(scope="session")
def test_audio_path() -> Path:
    if not TEST_AUDIO_PATH.exists():
        pytest.skip(f"Private test recording not found at {TEST_AUDIO_PATH}")
    return TEST_AUDIO_PATH


@pytest.fixture(scope="session")
def test_audio_wav(test_audio_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from parakeet_transcribe import audio

    audio.ensure_ffmpeg()
    out_dir = tmp_path_factory.mktemp("wav")
    return audio.convert_to_wav(test_audio_path, out_dir)
