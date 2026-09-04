"""Unit tests for audio.py's silence-aligned chunk planning - fast, no models or ffmpeg."""

from __future__ import annotations

from parakeet_transcribe.audio import plan_chunks


def test_short_audio_is_not_split():
    bounds = plan_chunks(duration_sec=90, silences=[], target_sec=120, max_sec=180)
    assert bounds == [(0.0, 90)]


def test_prefers_nearest_silence_midpoint_to_target():
    # Silence at (119.5, 120.5) is right at the 120s target; (245, 246) sits well past it.
    bounds = plan_chunks(
        duration_sec=400,
        silences=[(119.5, 120.5), (245.0, 246.0)],
        target_sec=120,
        max_sec=180,
    )
    assert bounds[0] == (0.0, 120.0)


def test_falls_back_to_hard_cut_when_no_silence_in_range():
    bounds = plan_chunks(duration_sec=500, silences=[], target_sec=120, max_sec=180)
    assert all(end - start <= 180 for start, end in bounds)
    assert bounds[0] == (0.0, 180.0)


def test_chunks_are_contiguous_and_cover_the_full_duration():
    bounds = plan_chunks(
        duration_sec=1300,
        silences=[(118, 119), (119.5, 120.5), (240, 241), (245, 246)],
        target_sec=120,
        max_sec=180,
    )
    assert bounds[0][0] == 0.0
    assert bounds[-1][1] == 1300
    for (_, end_a), (start_b, _) in zip(bounds, bounds[1:]):
        assert end_a == start_b
    assert all(end - start <= 180 + 1e-6 for start, end in bounds)
