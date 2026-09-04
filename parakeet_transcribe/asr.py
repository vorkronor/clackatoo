"""ASR via NVIDIA Parakeet (nvidia/parakeet-tdt-0.6b-v3) through the NeMo toolkit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import audio
from .util import log, resolve_device

DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

# Audio longer than this switches the encoder to local (windowed) attention, per
# NVIDIA's guidance for long-form transcription: full attention is quadratic in
# audio length and can blow up memory well before a typical interview ends.
# Below the threshold we keep full attention for the best accuracy.
LONG_AUDIO_THRESHOLD_SEC = 20 * 60
LOCAL_ATTENTION_CONTEXT = [256, 256]

# Hard ceiling for a single --chunk-minutes chunk, regardless of how large a value
# is requested: this keeps every chunk safely below LONG_AUDIO_THRESHOLD_SEC, since
# the local-attention path above has been observed to both hit unimplemented ops on
# some MPS backends and use far more memory than its "bounded" design intends.
ABSOLUTE_MAX_CHUNK_SEC = 15 * 60


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    text: str
    start: float
    end: float


@dataclass
class AsrResult:
    text: str
    words: list[Word]
    segments: list[Segment]


class ParakeetASR:
    """Loads the model once; call transcribe() per file to reuse it across a batch."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "auto"):
        import nemo.collections.asr as nemo_asr  # heavy import, deferred to first use

        self.device = resolve_device(device)
        log(f"Loading ASR model {model_id} on {self.device} ...")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)

        try:
            self.model = self.model.to(self.device)
        except RuntimeError as e:
            if self.device != "cpu":
                log(f"Could not move ASR model to {self.device} ({e}); falling back to cpu.")
                self.device = "cpu"
                self.model = self.model.to("cpu")
            else:
                raise

        self.model.eval()
        self._local_attention_active = False

    def _use_local_attention(self) -> None:
        if not self._local_attention_active:
            log("Long audio detected: switching encoder to local attention.")
            self.model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=LOCAL_ATTENTION_CONTEXT,
            )
            self._local_attention_active = True

    def transcribe(self, wav_path: Path, duration_sec: Optional[float] = None) -> AsrResult:
        if duration_sec is not None and duration_sec > LONG_AUDIO_THRESHOLD_SEC:
            self._use_local_attention()

        import torch

        with torch.inference_mode():
            output = self.model.transcribe([str(wav_path)], timestamps=True)

        hyp = output[0]
        ts = hyp.timestamp or {}
        word_ts = ts.get("word", [])
        segment_ts = ts.get("segment", [])

        words = [
            Word(text=w["word"], start=float(w["start"]), end=float(w["end"]))
            for w in word_ts
        ]
        segments = [
            Segment(text=s["segment"], start=float(s["start"]), end=float(s["end"]))
            for s in segment_ts
        ]
        return AsrResult(text=hyp.text, words=words, segments=segments)

    def transcribe_long_form(
        self,
        wav_path: Path,
        duration_sec: float,
        chunk_minutes: float,
        tmp_dir: Path,
    ) -> AsrResult:
        """Like transcribe(), but splits audio above chunk_minutes into silence-aligned
        chunks first, transcribing each independently and merging the results.

        Keeping chunks short avoids LONG_AUDIO_THRESHOLD_SEC's local-attention path
        entirely for typical recordings, which sidesteps both an MPS backend op gap
        and much higher memory use than that path's "bounded" design suggests.
        Pass chunk_minutes <= 0 to disable and always transcribe in one shot.
        """
        max_chunk_sec = min(chunk_minutes * 60 * 1.5, ABSOLUTE_MAX_CHUNK_SEC) if chunk_minutes > 0 else None
        if max_chunk_sec is None or duration_sec <= max_chunk_sec:
            return self.transcribe(wav_path, duration_sec=duration_sec)

        log(
            f"Recording is {duration_sec / 60:.1f} min; splitting into ~{chunk_minutes:.1f} "
            "min chunks at silence for ASR."
        )
        silences = audio.detect_silences(wav_path)
        bounds = audio.plan_chunks(
            duration_sec, silences, target_sec=chunk_minutes * 60, max_sec=max_chunk_sec
        )
        chunks_dir = tmp_dir / f"{wav_path.stem}_asr_chunks"
        chunk_paths = audio.split_wav_into_chunks(wav_path, chunks_dir, bounds)

        texts: list[str] = []
        all_words: list[Word] = []
        all_segments: list[Segment] = []
        for (start, end), chunk_path in zip(bounds, chunk_paths):
            log(f"  chunk {start / 60:.1f}-{end / 60:.1f} min")
            result = self.transcribe(chunk_path, duration_sec=end - start)
            if result.text:
                texts.append(result.text)
            all_words.extend(Word(w.text, w.start + start, w.end + start) for w in result.words)
            all_segments.extend(Segment(s.text, s.start + start, s.end + start) for s in result.segments)

        return AsrResult(text=" ".join(texts), words=all_words, segments=all_segments)
