"""End-to-end smoke test: invoke the CLI exactly as a user would."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.integration


def test_cli_transcribes_without_diarization(test_audio_path: Path, tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "transcribe.py"),
            str(test_audio_path),
            "--no-diarization",
            "-o", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    stem = test_audio_path.stem
    txt_path = tmp_path / f"{stem}.txt"
    srt_path = tmp_path / f"{stem}.srt"
    json_path = tmp_path / f"{stem}.json"

    assert txt_path.exists() and txt_path.read_text(encoding="utf-8").strip()
    assert srt_path.exists() and "-->" in srt_path.read_text(encoding="utf-8")
    assert json_path.exists()
