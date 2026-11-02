"""SPHARE-BD Experiment A runner (aliases run_proposed_model.py)."""
from __future__ import annotations
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from run_proposed_model import run_proposed_model, generate_verification_report

# Backward compatible alias
run_experiment_a = run_proposed_model

if __name__ == "__main__":
    seeds_arg = [int(s) for s in sys.argv[1:] if s.isdigit()] or [42, 123, 2024]
    df_res = run_proposed_model(seeds=seeds_arg)
    generate_verification_report(df_res)
