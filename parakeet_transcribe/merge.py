"""Merge ASR word timestamps with diarization speaker turns."""

from __future__ import annotations

from dataclasses import dataclass

from .asr import Word
from .diarize import SpeakerTurn

UNKNOWN_SPEAKER = "UNKNOWN"

# When grouping consecutive same-speaker words into one printable utterance,
# start a new one after this much silence or this many words, so a single
# uninterrupted monologue doesn't become one giant subtitle-unfriendly block.
MAX_GAP_SEC = 1.0
MAX_UTTERANCE_WORDS = 40


@dataclass
class SpeakerWord:
    text: str
    start: float
    end: float
    speaker: str


@dataclass
class Utterance:
    speaker: str
    start: float
    end: float
    text: str


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(words: list[Word], turns: list[SpeakerTurn]) -> list[SpeakerWord]:
    """Tag each ASR word with the diarization speaker it overlaps most.

    Words that fall in a small gap between turns (no direct overlap) are
    assigned to the nearest turn by timestamp distance instead of being left
    unlabeled.
    """
    if not turns:
        return [SpeakerWord(w.text, w.start, w.end, UNKNOWN_SPEAKER) for w in words]

    out = []
    for w in words:
        best_overlap, best_turn = max(
            ((_overlap(w.start, w.end, t.start, t.end), t) for t in turns),
            key=lambda pair: pair[0],
        )
        if best_overlap > 0:
            speaker = best_turn.speaker
        else:
            mid = (w.start + w.end) / 2
            nearest = min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end)))
            speaker = nearest.speaker
        out.append(SpeakerWord(w.text, w.start, w.end, speaker))
    return out


def group_utterances(
    words: list[SpeakerWord],
    max_gap: float = MAX_GAP_SEC,
    max_words: int = MAX_UTTERANCE_WORDS,
) -> list[Utterance]:
    """Group consecutive same-speaker words into utterances (speaker turns)."""
    if not words:
        return []

    utterances: list[Utterance] = []
    current_speaker = words[0].speaker
    current_words = [words[0]]

    for w in words[1:]:
        gap = w.start - current_words[-1].end
        same_speaker = w.speaker == current_speaker
        if same_speaker and gap <= max_gap and len(current_words) < max_words:
            current_words.append(w)
        else:
            utterances.append(_make_utterance(current_speaker, current_words))
            current_speaker = w.speaker
            current_words = [w]

    utterances.append(_make_utterance(current_speaker, current_words))
    return utterances


def _make_utterance(speaker: str, words: list[SpeakerWord]) -> Utterance:
    text = " ".join(w.text for w in words).strip()
    return Utterance(speaker=speaker, start=words[0].start, end=words[-1].end, text=text)
