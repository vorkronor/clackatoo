"""Writers for the three output formats: .txt, .srt, .json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .merge import SpeakerWord, Utterance


def _srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_txt(utterances: list[Utterance], path: Path, include_speakers: bool) -> None:
    lines = []
    for u in utterances:
        if include_speakers:
            lines.append(f"{u.speaker}: {u.text}")
        else:
            lines.append(u.text)
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def write_srt(utterances: list[Utterance], path: Path, include_speakers: bool) -> None:
    blocks = []
    for i, u in enumerate(utterances, start=1):
        text = f"{u.speaker}: {u.text}" if include_speakers else u.text
        blocks.append(
            f"{i}\n{_srt_timestamp(u.start)} --> {_srt_timestamp(u.end)}\n{text}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def write_json(
    path: Path,
    audio_file: str,
    duration_sec: float,
    asr_model: str,
    diarization_model: Optional[str],
    utterances: list[Utterance],
    words: list[SpeakerWord],
    include_speakers: bool,
) -> None:
    speakers = sorted({u.speaker for u in utterances}) if include_speakers else []
    data = {
        "audio_file": audio_file,
        "duration_sec": round(duration_sec, 3),
        "asr_model": asr_model,
        "diarization_model": diarization_model if include_speakers else None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "speakers": speakers,
        "utterances": [
            {
                "speaker": u.speaker if include_speakers else None,
                "start": round(u.start, 3),
                "end": round(u.end, 3),
                "text": u.text,
            }
            for u in utterances
        ],
        "words": [
            {
                "word": w.text,
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "speaker": w.speaker if include_speakers else None,
            }
            for w in words
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
