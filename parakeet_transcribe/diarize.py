"""Speaker diarization via pyannote.audio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .util import log, resolve_device

DEFAULT_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"


@dataclass
class SpeakerTurn:
    speaker: str
    start: float
    end: float


class Diarizer:
    def __init__(
        self,
        model_id: str = DEFAULT_DIARIZATION_MODEL_ID,
        hf_token: Optional[str] = None,
        device: str = "auto",
    ):
        from pyannote.audio import Pipeline
        import torch

        if not hf_token:
            raise RuntimeError(
                "A Hugging Face access token is required for speaker diarization. "
                "Pass --hf-token, set the HF_TOKEN environment variable, or run "
                "`hf auth login`. You must also accept the user conditions for "
                "pyannote/segmentation-3.0 and pyannote/speaker-diarization-3.1 on huggingface.co."
            )

        self.device = resolve_device(device)
        log(f"Loading diarization model {model_id} on {self.device} ...")
        self.pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
        if self.pipeline is None:
            raise RuntimeError(
                f"Failed to load {model_id}. Make sure you've accepted its user "
                "conditions (and pyannote/segmentation-3.0's) on huggingface.co, and "
                "that your token has read access."
            )
        try:
            self.pipeline.to(torch.device(self.device))
        except RuntimeError as e:
            if self.device != "cpu":
                log(f"Could not move diarization pipeline to {self.device} ({e}); falling back to cpu.")
                self.device = "cpu"
                self.pipeline.to(torch.device("cpu"))
            else:
                raise

    def diarize(
        self,
        wav_path: Path,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> list[SpeakerTurn]:
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

        annotation = self.pipeline(str(wav_path), **kwargs)
        turns = [
            SpeakerTurn(speaker=speaker, start=float(turn.start), end=float(turn.end))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda t: t.start)
        return turns
