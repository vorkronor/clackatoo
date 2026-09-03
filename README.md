# parakeet-transcription

CLI for transcribing Czech/English (and 23 other EU languages, auto-detected) interview
recordings using [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
with optional speaker diarization ([pyannote.audio](https://github.com/pyannote/pyannote-audio)).
Any input format ffmpeg can read (.m4a, .mp3, .wav, .mp4, ...) is supported.

## Setup

1. **ffmpeg**: `brew install ffmpeg`
2. **Python env** (3.10 or 3.11 recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   This pulls in PyTorch + NeMo + pyannote.audio — a multi-GB install, expect it to take a while.
3. **Hugging Face token** (only needed for diarization):
   - Create a token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is enough).
   - Accept the user conditions for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
     and [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
   - Either run `hf auth login`, set `HF_TOKEN` in your environment, or pass `--hf-token`.
4. Sanity check everything is wired up before running on real audio:
   ```
   python transcribe.py --check
   ```

The ASR model (~2.4GB) and diarization models are downloaded once on first use and cached under
`~/.cache/huggingface`.

## Usage

```
python transcribe.py interview1.m4a
python transcribe.py recordings/                  # every audio file in the folder
python transcribe.py recordings/ -r                # + subfolders
python transcribe.py interview1.m4a --no-diarization   # faster, plain transcript
python transcribe.py recordings/ -o transcripts/    # write outputs to a separate folder
python transcribe.py interview1.m4a --num-speakers 2  # hint speaker count if known
```

By default every audio file gets three outputs next to it: `.txt`, `.srt`, `.json`. Restrict with
`--formats`, e.g. `--formats json`. Files that already have all their expected outputs are skipped
on a re-run unless you pass `--overwrite`.

Run `python transcribe.py --help` for the full flag list (device selection, custom model ids, etc).

## Output

- **`.txt`** — plain transcript, one paragraph per speaker turn (`SPEAKER_00: ...`).
- **`.srt`** — the same turns as subtitle cues, for scrubbing to a moment in the recording.
- **`.json`** — everything, meant for building a front-end on top of later:
  ```jsonc
  {
    "audio_file": "interview1.m4a",
    "duration_sec": 1834.2,
    "asr_model": "nvidia/parakeet-tdt-0.6b-v3",
    "diarization_model": "pyannote/speaker-diarization-3.1",
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "utterances": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 4.32, "text": "..."}],
    "words":      [{"word": "Dobrý", "start": 0.0, "end": 0.32, "speaker": "SPEAKER_00"}]
  }
  ```

## Notes / known limitations

- Speaker labels are `SPEAKER_00`, `SPEAKER_01`, ... (pyannote's generic labels, not names) —
  consistent within one file, not across files.
- Diarization and ASR are two separate models; speaker boundaries are lined up with word
  timestamps by overlap, so a mislabeled word near a speaker change is possible on crosstalk.
- Long recordings (>20 min) automatically switch the ASR encoder to windowed ("local")
  attention to keep memory bounded, per NVIDIA's guidance — a small potential accuracy trade-off
  versus short clips.
- Device is auto-detected (CUDA > Apple Silicon MPS > CPU); override with `--device`. Parakeet is
  built/tested by NVIDIA on CUDA GPUs — CPU/MPS work but are slower, and if an op isn't supported
  on MPS the script automatically falls back to CPU with a warning.
- With `-o`/`--output-dir` and `-r`/`--recursive` together, two input files with the same name in
  different subfolders will overwrite each other's output — keep filenames unique or run per
  subfolder in that case.
