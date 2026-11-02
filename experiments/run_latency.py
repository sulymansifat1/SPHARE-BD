"""SPHARE-BD Computational Latency Measurement."""
from __future__ import annotations
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from run_latency_benchmark import run_latency_benchmark

# Direct alias
run_latency = run_latency_benchmark

if __name__ == "__main__":
    df = run_latency()
    print(df[["component", "per_recording_latency_median_ms", "real_time_factor"]])
