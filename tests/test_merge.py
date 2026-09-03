"""Unit tests for merge.py - fast, deterministic, no models or audio needed."""

from __future__ import annotations

from parakeet_transcribe.asr import Word
from parakeet_transcribe.diarize import SpeakerTurn
from parakeet_transcribe.merge import UNKNOWN_SPEAKER, SpeakerWord, assign_speakers, group_utterances


def test_assign_speakers_by_overlap():
    words = [Word("hello", 0.0, 0.5), Word("world", 1.0, 1.5)]
    turns = [SpeakerTurn("A", 0.0, 0.8), SpeakerTurn("B", 0.8, 2.0)]

    tagged = assign_speakers(words, turns)

    assert [w.speaker for w in tagged] == ["A", "B"]


def test_assign_speakers_falls_back_to_nearest_turn_in_a_gap():
    words = [Word("um", 0.9, 0.95)]  # falls in the gap between the two turns below
    turns = [SpeakerTurn("A", 0.0, 0.5), SpeakerTurn("B", 1.0, 2.0)]

    tagged = assign_speakers(words, turns)

    assert tagged[0].speaker == "B"  # closer to B's start than A's end


def test_assign_speakers_without_turns_is_unknown():
    tagged = assign_speakers([Word("hi", 0.0, 0.2)], turns=[])
    assert tagged[0].speaker == UNKNOWN_SPEAKER


def test_group_utterances_splits_on_gap_and_speaker_change():
    words = [
        SpeakerWord("hi", 0.0, 0.2, "A"),
        SpeakerWord("there", 0.3, 0.6, "A"),
        SpeakerWord("hey", 3.0, 3.2, "A"),  # gap > MAX_GAP_SEC -> new utterance
        SpeakerWord("you", 3.3, 3.6, "B"),  # speaker change -> new utterance
    ]

    utterances = group_utterances(words)

    assert [u.text for u in utterances] == ["hi there", "hey", "you"]
    assert [u.speaker for u in utterances] == ["A", "A", "B"]


def test_group_utterances_splits_long_monologue_by_word_count():
    words = [SpeakerWord(str(i), float(i), float(i) + 0.5, "A") for i in range(45)]

    utterances = group_utterances(words, max_words=40)

    assert len(utterances) == 2
    assert len(utterances[0].text.split()) == 40
    assert len(utterances[1].text.split()) == 5
