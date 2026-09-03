"""Unit tests for formats.py - fast, deterministic, no models or audio needed."""

from __future__ import annotations

import json
from pathlib import Path

from parakeet_transcribe.formats import write_json, write_srt, write_txt
from parakeet_transcribe.merge import SpeakerWord, Utterance

UTTERANCES = [
    Utterance(speaker="SPEAKER_00", start=0.0, end=1.5, text="hello there"),
    Utterance(speaker="SPEAKER_01", start=1.5, end=3.0, text="hi"),
]
WORDS = [
    SpeakerWord("hello", 0.0, 0.5, "SPEAKER_00"),
    SpeakerWord("there", 0.6, 1.5, "SPEAKER_00"),
    SpeakerWord("hi", 1.5, 3.0, "SPEAKER_01"),
]


def test_write_txt_includes_speaker_labels(tmp_path: Path):
    path = tmp_path / "out.txt"
    write_txt(UTTERANCES, path, include_speakers=True)
    content = path.read_text(encoding="utf-8")
    assert "SPEAKER_00: hello there" in content
    assert "SPEAKER_01: hi" in content


def test_write_txt_omits_speaker_labels_when_disabled(tmp_path: Path):
    path = tmp_path / "out.txt"
    write_txt(UTTERANCES, path, include_speakers=False)
    content = path.read_text(encoding="utf-8")
    assert "SPEAKER" not in content
    assert "hello there" in content


def test_write_srt_timestamps_and_order(tmp_path: Path):
    path = tmp_path / "out.srt"
    write_srt(UTTERANCES, path, include_speakers=False)
    content = path.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,500" in content
    assert content.index("hello there") < content.index("hi")


def test_write_json_roundtrip(tmp_path: Path):
    path = tmp_path / "out.json"
    write_json(
        path,
        audio_file="sample.wav",
        duration_sec=3.0,
        asr_model="nvidia/parakeet-tdt-0.6b-v3",
        diarization_model="pyannote/speaker-diarization-3.1",
        utterances=UTTERANCES,
        words=WORDS,
        include_speakers=True,
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert len(data["utterances"]) == 2
    assert data["words"][0] == {"word": "hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}


def test_write_json_hides_speakers_when_disabled(tmp_path: Path):
    path = tmp_path / "out.json"
    write_json(
        path,
        audio_file="sample.wav",
        duration_sec=3.0,
        asr_model="nvidia/parakeet-tdt-0.6b-v3",
        diarization_model=None,
        utterances=UTTERANCES,
        words=WORDS,
        include_speakers=False,
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["speakers"] == []
    assert data["diarization_model"] is None
    assert all(u["speaker"] is None for u in data["utterances"])
