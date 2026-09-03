#!/usr/bin/env python3
"""Entry point. See `python transcribe.py --help` or README.md for usage."""

import sys

from parakeet_transcribe.cli import main

if __name__ == "__main__":
    sys.exit(main())
