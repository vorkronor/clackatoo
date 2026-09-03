"""Small shared helpers: device resolution and timestamped logging."""

from __future__ import annotations

import sys
import time


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device string: 'cuda', 'mps', or 'cpu'."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


class Stopwatch:
    def __init__(self):
        self._start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def elapsed_str(self) -> str:
        s = self.elapsed()
        if s < 60:
            return f"{s:.1f}s"
        m, s = divmod(s, 60)
        return f"{int(m)}m{s:.0f}s"
