"""Allow the roadmap's direct script invocation without installing the spikes package."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("spikes.provider.cli", run_name="__main__")
