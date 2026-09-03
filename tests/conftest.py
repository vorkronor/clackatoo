"""Shared fixtures.

The integration tests exercise the real pipeline against a personal audio
recording that each developer keeps out of version control (see .gitignore).
Drop any audio file ffmpeg can read at `_test-data/sample.<ext>` (e.g.
`sample.wav`, `sample.m4a`) to run them; anyone without such a file gets
those tests skipped rather than failed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_AUDIO_DIR = REPO_ROOT / "_test-data"
TEST_AUDIO_GLOB = "sample.*"


@pytest.fixture(scope="session")
def test_audio_path() -> Path:
    matches = sorted(TEST_AUDIO_DIR.glob(TEST_AUDIO_GLOB)) if TEST_AUDIO_DIR.is_dir() else []
    if not matches:
        pytest.skip(
            f"No test recording found: drop an audio file at "
            f"{TEST_AUDIO_DIR / TEST_AUDIO_GLOB} to run this test"
        )
    return matches[0]


@pytest.fixture(scope="session")
def test_audio_wav(test_audio_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from parakeet_transcribe import audio

    audio.ensure_ffmpeg()
    out_dir = tmp_path_factory.mktemp("wav")
    return audio.convert_to_wav(test_audio_path, out_dir)
