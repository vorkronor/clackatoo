"""Command-line entry point: batch-transcribe interview recordings with Parakeet."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from . import audio
from .asr import DEFAULT_MODEL_ID, ParakeetASR
from .diarize import DEFAULT_DIARIZATION_MODEL_ID, Diarizer
from .formats import write_json, write_srt, write_txt
from .merge import assign_speakers, group_utterances
from .util import Stopwatch, log


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe interview recordings (Czech/English, auto-detected) with "
        "NVIDIA Parakeet, with optional speaker diarization.",
    )
    p.add_argument("paths", nargs="*", type=Path, help="Audio file(s) or director(y/ies) to transcribe")
    p.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    p.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Write outputs here instead of next to each source file",
    )
    p.add_argument(
        "--formats", nargs="+", choices=["txt", "srt", "json"], default=["txt", "srt", "json"],
        help="Which output formats to write (default: all three)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL_ID, help="ASR model id")
    p.add_argument(
        "--chunk-minutes", type=float, default=2.0,
        help="Split recordings longer than this into silence-aligned chunks before ASR "
        "(reduces memory use on long recordings); pass 0 to disable and transcribe in one shot",
    )
    p.add_argument("--no-diarization", action="store_true", help="Skip speaker diarization")
    p.add_argument("--diarization-model", default=DEFAULT_DIARIZATION_MODEL_ID)
    p.add_argument("--num-speakers", type=int, default=None, help="Exact speaker count, if known")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--hf-token", default=None, help="Hugging Face token (else HF_TOKEN env var / cached login)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--overwrite", action="store_true", help="Re-transcribe even if outputs already exist")
    p.add_argument("--check", action="store_true", help="Verify environment and exit (no transcription)")
    return p.parse_args(argv)


def resolve_hf_token(cli_token: str | None) -> str | None:
    from huggingface_hub import get_token

    return (
        cli_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or get_token()
    )


def output_paths(src: Path, output_dir: Path | None, formats: list[str]) -> dict[str, Path]:
    base_dir = output_dir if output_dir else src.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    return {fmt: base_dir / f"{stem}.{fmt}" for fmt in formats}


def already_done(paths: dict[str, Path]) -> bool:
    return all(p.exists() for p in paths.values())


def run_check(args: argparse.Namespace) -> int:
    ok = True
    try:
        audio.ensure_ffmpeg()
        log("ffmpeg/ffprobe: OK")
    except RuntimeError as e:
        log(f"ffmpeg/ffprobe: FAILED - {e}")
        ok = False

    try:
        import nemo.collections.asr  # noqa: F401
        log("nemo_toolkit[asr]: OK (importable)")
    except ImportError as e:
        log(f"nemo_toolkit[asr]: FAILED - {e}")
        ok = False

    if not args.no_diarization:
        try:
            import pyannote.audio  # noqa: F401
            log("pyannote.audio: OK (importable)")
        except ImportError as e:
            log(f"pyannote.audio: FAILED - {e}")
            ok = False

        token = resolve_hf_token(args.hf_token)
        if token:
            log("Hugging Face token: found")
        else:
            log("Hugging Face token: NOT found (set --hf-token or HF_TOKEN, or run `hf auth login`)")
            ok = False

    try:
        import torch
        log(
            f"torch: {torch.__version__} | cuda={torch.cuda.is_available()} | "
            f"mps={torch.backends.mps.is_available()}"
        )
    except ImportError as e:
        log(f"torch: FAILED - {e}")
        ok = False

    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.check:
        return run_check(args)

    if not args.paths:
        log("No input paths given. Pass one or more audio files or directories, or use --check.")
        return 2

    audio.ensure_ffmpeg()
    files = audio.discover_audio_files(args.paths, args.recursive)
    if not files:
        log("No audio files found.")
        return 1
    log(f"Found {len(files)} audio file(s) to process.")

    diarize_enabled = not args.no_diarization
    hf_token = resolve_hf_token(args.hf_token)

    asr = ParakeetASR(model_id=args.model, device=args.device)
    diarizer = None
    if diarize_enabled:
        diarizer = Diarizer(model_id=args.diarization_model, hf_token=hf_token, device=args.device)

    failures: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="parakeet_transcribe_") as tmp:
        tmp_dir = Path(tmp)
        for i, src in enumerate(files, start=1):
            log(f"[{i}/{len(files)}] {src.name}")
            out_paths = output_paths(src, args.output_dir, args.formats)
            if not args.overwrite and already_done(out_paths):
                log("  already transcribed, skipping (use --overwrite to redo)")
                continue

            sw = Stopwatch()
            try:
                wav_path = audio.convert_to_wav(src, tmp_dir)
                duration = audio.get_duration_seconds(wav_path)

                asr_result = asr.transcribe_long_form(
                    wav_path, duration_sec=duration, chunk_minutes=args.chunk_minutes, tmp_dir=tmp_dir
                )

                if diarize_enabled:
                    turns = diarizer.diarize(
                        wav_path,
                        num_speakers=args.num_speakers,
                        min_speakers=args.min_speakers,
                        max_speakers=args.max_speakers,
                    )
                else:
                    turns = []

                speaker_words = assign_speakers(asr_result.words, turns)
                utterances = group_utterances(speaker_words)

                if "txt" in out_paths:
                    write_txt(utterances, out_paths["txt"], include_speakers=diarize_enabled)
                if "srt" in out_paths:
                    write_srt(utterances, out_paths["srt"], include_speakers=diarize_enabled)
                if "json" in out_paths:
                    write_json(
                        out_paths["json"],
                        audio_file=src.name,
                        duration_sec=duration,
                        asr_model=args.model,
                        diarization_model=args.diarization_model if diarize_enabled else None,
                        utterances=utterances,
                        words=speaker_words,
                        include_speakers=diarize_enabled,
                    )

                wav_path.unlink(missing_ok=True)
                log(f"  done in {sw.elapsed_str()}")
            except Exception as e:
                log(f"  FAILED: {e}")
                failures.append(src)

    if failures:
        log(f"{len(failures)} file(s) failed: {', '.join(f.name for f in failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
