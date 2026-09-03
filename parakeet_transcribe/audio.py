"""Audio format handling: ffmpeg conversion and duration probing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

TARGET_SAMPLE_RATE = 16000

# Extensions we'll pick up automatically when a directory is passed on the CLI.
# ffmpeg supports far more than this; the list just controls directory scanning.
KNOWN_AUDIO_EXTENSIONS = {
    ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma",
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
