"""Audio format handling: ffmpeg conversion, duration probing, and silence-aware chunking."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

TARGET_SAMPLE_RATE = 16000

# ffmpeg's silencedetect filter: noise floor and minimum duration tuned for speech
# pauses (sentence/breath gaps) rather than e.g. only near-total digital silence.
SILENCE_NOISE_DB = "-30dB"
SILENCE_MIN_DURATION_SEC = 0.4

# Extensions we'll pick up automatically when a directory is passed on the CLI.
# ffmpeg supports far more than this; the list just controls directory scanning.
KNOWN_AUDIO_EXTENSIONS = {
    ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma", ".aif", ".aiff", ".aifc",
    ".mp4", ".mov", ".mkv", ".webm",
}


def ensure_ffmpeg() -> None:
    """Raise a clear error if ffmpeg/ffprobe aren't on PATH."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"Required tool(s) not found on PATH: {', '.join(missing)}. "
            "Install ffmpeg first, e.g. `brew install ffmpeg` on macOS."
        )


def convert_to_wav(input_path: Path, out_dir: Path) -> Path:
    """Convert any ffmpeg-readable audio/video file to 16kHz mono PCM WAV.

    Returns the path to the converted file inside out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (input_path.stem + ".wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
        "-vn",  # drop video stream if present (e.g. .mp4/.mov)
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to convert {input_path.name}:\n{result.stderr.strip()[-2000:]}"
        )
    return out_path


def get_duration_seconds(path: Path) -> float:
    """Probe media duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path.name}:\n{result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not parse duration for {path.name}: {result.stdout!r}")


def detect_silences(
    wav_path: Path,
    noise_db: str = SILENCE_NOISE_DB,
    min_duration: float = SILENCE_MIN_DURATION_SEC,
) -> list[tuple[float, float]]:
    """Return (start, end) seconds for each silent span, via ffmpeg's silencedetect filter."""
    cmd = [
        "ffmpeg", "-i", str(wav_path),
        "-af", f"silencedetect=noise={noise_db}:d={min_duration}",
        "-f", "null", "-",
    ]
    # silencedetect reports to stderr regardless of exit status; a nonzero return
    # here means the input itself is broken, which convert_to_wav would already
    # have caught, so we don't re-check result.returncode.
    result = subprocess.run(cmd, capture_output=True, text=True)
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", result.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", result.stderr)]
    # A silence that runs to end-of-file logs a start with no matching end; drop it,
    # there's nothing after it to split before anyway.
    return list(zip(starts, ends))


def plan_chunks(
    duration_sec: float,
    silences: list[tuple[float, float]],
    target_sec: float,
    max_sec: float,
) -> list[tuple[float, float]]:
    """Pick (start, end) cut points roughly every target_sec.

    Each cut prefers the midpoint of the nearest detected silence to the target, so
    a chunk boundary falls in a pause rather than mid-word. If no silence is found
    within range, falls back to a hard cut at max_sec (accepting a mid-sentence cut)
    so a long silence-free stretch can't grow a chunk without bound.
    """
    if duration_sec <= max_sec:
        return [(0.0, duration_sec)]

    midpoints = sorted((s + e) / 2 for s, e in silences)
    bounds: list[tuple[float, float]] = []
    pos = 0.0
    while duration_sec - pos > max_sec:
        target = pos + target_sec
        window_lo, window_hi = pos + target_sec * 0.5, min(pos + max_sec, duration_sec)
        candidates = [m for m in midpoints if window_lo <= m <= window_hi]
        cut = min(candidates, key=lambda m: abs(m - target)) if candidates else pos + max_sec
        bounds.append((pos, cut))
        pos = cut
    bounds.append((pos, duration_sec))
    return bounds


def split_wav_into_chunks(
    wav_path: Path, chunks_dir: Path, bounds: list[tuple[float, float]]
) -> list[Path]:
    """Slice a PCM WAV into one file per (start, end) bound in bounds."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []
    for i, (start, end) in enumerate(bounds):
        out_path = chunks_dir / f"{wav_path.stem}_chunk{i:03d}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-c", "copy",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to slice chunk {i} of {wav_path.name}:\n{result.stderr.strip()[-2000:]}"
            )
        chunk_paths.append(out_path)
    return chunk_paths


def discover_audio_files(paths: list[Path], recursive: bool) -> list[Path]:
    """Expand a mix of files and directories into a sorted list of audio files."""
    found: list[Path] = []
    for p in paths:
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            for candidate in sorted(p.glob(pattern)):
                if candidate.is_file() and candidate.suffix.lower() in KNOWN_AUDIO_EXTENSIONS:
                    found.append(candidate)
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"No such file or directory: {p}")
    return found
