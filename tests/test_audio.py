"""Tests for audio.py's ffmpeg conversion and file discovery."""

from __future__ import annotations

from pathlib import Path

from parakeet_transcribe import audio


def test_convert_to_wav_produces_a_wav_file(test_audio_wav: Path):
    assert test_audio_wav.exists()
    assert test_audio_wav.suffix == ".wav"


def test_get_duration_seconds_is_positive_and_reasonable(test_audio_wav: Path):
    duration = audio.get_duration_seconds(test_audio_wav)
    assert duration > 0
    assert duration < 300  # this is a short voice sample, not a multi-hour recording


def test_discover_audio_files_accepts_a_direct_file_regardless_of_extension(test_audio_path: Path):
    found = audio.discover_audio_files([test_audio_path], recursive=False)
    assert found == [test_audio_path]


def test_discover_audio_files_finds_aifc_when_scanning_a_directory(test_audio_path: Path):
    found = audio.discover_audio_files([test_audio_path.parent], recursive=False)
    assert test_audio_path in found
