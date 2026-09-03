"""ASR via NVIDIA Parakeet (nvidia/parakeet-tdt-0.6b-v3) through the NeMo toolkit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .util import log, resolve_device

DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

# Audio longer than this switches the encoder to local (windowed) attention, per
# NVIDIA's guidance for long-form transcription: full attention is quadratic in
# audio length and can blow up memory well before a typical interview ends.
# Below the threshold we keep full attention for the best accuracy.
LONG_AUDIO_THRESHOLD_SEC = 20 * 60
LOCAL_ATTENTION_CONTEXT = [256, 256]


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
