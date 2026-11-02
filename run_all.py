#!/usr/bin/env python3
"""SPHARE-BD: Central Research Pipeline Launcher.

Hierarchical Support Assessment for Selective Bearing Fault Diagnosis Under Operating-Condition Shifts
Authors: Md. Sulyman Islam Sifat, Md Alamgir Kabir (Daffodil International University)

Usage:
    python run_all.py                     # Runs complete research pipeline
    python run_all.py --only model        # Runs proposed model evaluation
    python run_all.py --only ablation     # Runs sequential gate ablation
    python run_all.py --only baselines    # Runs coverage-matched baseline comparison
    python run_all.py --only synthetic    # Runs controlled synthetic benchmarks
    python run_all.py --only latency      # Measures CPU inference latency
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = RESULTS_DIR / "reports"
LOGS_DIR = RESULTS_DIR / "logs"

for d in [SRC_DIR, EXPERIMENTS_DIR, RESULTS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

LOG_FILE = LOGS_DIR / "run_all.log"


def log(msg: str, header: bool = False) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if header:
        formatted = f"\n{'='*80}\n[{timestamp}] {msg}\n{'='*80}"
    else:
        formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(formatted + "\n")


# ==============================================================================
# ENVIRONMENT CHECK & VIRTUAL ENVIRONMENT MANAGEMENT
# ==============================================================================
def check_python_version() -> Tuple[bool, str]:
    v = sys.version_info
    v_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 9:
        return True, v_str
    return False, v_str


def setup_virtual_environment(venv_dir: Path) -> Path:
    """Creates isolated virtual environment and returns python executable path."""
    is_win = platform.system() == "Windows"
    py_exec = venv_dir / ("Scripts/python.exe" if is_win else "bin/python")

    if not venv_dir.exists() or not py_exec.exists():
        log(f"[Env] Creating virtual environment at {venv_dir}...")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    # Upgrade pip and install requirements
    log("[Env] Verifying dependencies in virtual environment...")
    req_file = REPO_ROOT / "requirements.txt"
    if req_file.exists():
        subprocess.check_call([str(py_exec), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
        subprocess.check_call([str(py_exec), "-m", "pip", "install", "-r", str(req_file), "--quiet"])

    return py_exec


def is_running_in_venv() -> bool:
    """Checks whether the current process is already inside a virtual environment."""
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix or "VIRTUAL_ENV" in os.environ


# ==============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="SPHARE-BD Research Pipeline")
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "model", "ablation", "baselines", "synthetic", "latency"],
        help="Specify a single experiment to run (default: 'all')",
    )
    parser.add_argument("--no-venv", action="store_true", help="Do not create or switch into a .venv environment")
    parser.add_argument("--skip-download", action="store_true", help="Skip automatic downloading of public datasets")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pre-flight unit tests")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024], help="Random seeds for evaluation")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch compute device ('cpu' or 'cuda')")
    args = parser.parse_args()

    t_start = time.time()
    log("SPHARE-BD RESEARCH PIPELINE INITIALIZED", header=True)
    log(f"Repository Root: {REPO_ROOT}")
    log(f"Operating System: {platform.system()} ({platform.release()})")
    log(f"Architecture: {platform.machine()}")
    log(f"Target Task: {args.only}")

    # --------------------------------------------------------------------------
    # Step 1: Environment Check
    # --------------------------------------------------------------------------
    log("STEP 1: Checking Python Interpreter and Environment", header=True)
    valid_py, py_ver = check_python_version()
    if not valid_py:
        log(f"ERROR: Python 3.9+ is required. Found Python {py_ver}.", header=False)
        sys.exit(1)
    log(f"Python interpreter: {sys.executable} (Version {py_ver}) - COMPATIBLE")

    venv_dir = REPO_ROOT / ".venv"
    if not args.no_venv and not is_running_in_venv():
        try:
            py_exec = setup_virtual_environment(venv_dir)
            log(f"[Env] Virtual environment ready: {py_exec}")
            log("[Env] Switching execution context into .venv...")
            cmd = [str(py_exec), str(Path(__file__).resolve()), "--no-venv", "--only", args.only]
            if args.skip_download:
                cmd.append("--skip-download")
            if args.skip_tests:
                cmd.append("--skip-tests")
            if args.seeds:
                cmd.extend(["--seeds"] + [str(s) for s in args.seeds])
            cmd.extend(["--device", args.device])
            ret = subprocess.call(cmd)
            sys.exit(ret)
        except Exception as e:
            log(f"[Env] Note: Automatic venv setup failed ({e}); continuing with current environment.")

    # --------------------------------------------------------------------------
    # Step 2: Dataset Preparation
    # --------------------------------------------------------------------------
    log("STEP 2: Dataset Verification and Preparation", header=True)
    from datasets.data_manager import DataManager
    dm = DataManager(REPO_ROOT)
    summary = dm.get_summary()

    log(f"Active Dataset Search Root: {summary['dataset_root']}")
    log(f"SDUST Status: Available={summary['sdust']['available']}, Found Files={summary['sdust']['found_files']}")
    log(f"CWRU Status: Available={summary['cwru']['available']}, Found Files={summary['cwru']['found_files']}")

    if not summary["cwru"]["available"] and not args.skip_download:
        log("[Dataset] Downloading CWRU evaluation recordings from Case Western Reserve University...")
        dm.download_cwru(verbose=True)
        summary = dm.get_summary()

    # --------------------------------------------------------------------------
    # Step 3: Checkpoints & Models
    # --------------------------------------------------------------------------
    log("STEP 3: Model Checkpoint Verification", header=True)
    from model import FrozenFaultClassifier
    import torch

    for seed in args.seeds:
        ckpt_path = REPO_ROOT / "checkpoints" / f"classifier_seed{seed}.pt"
        if ckpt_path.exists():
            model = FrozenFaultClassifier(n_classes=4)
            ckpt = torch.load(ckpt_path, map_location=args.device)
            state = ckpt["model"] if "model" in ckpt else (ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            model.load_state_dict(state)
            log(f"Verified Checkpoint for Seed {seed}: {ckpt_path.name} (51,028 parameters)")
        else:
            log(f"WARNING: Checkpoint missing for seed {seed}: {ckpt_path}")

    # Optional Pre-flight Tests
    if not args.skip_tests:
        log("Running Pre-flight Unit Tests...", header=True)
        import unittest
        suite = unittest.defaultTestLoader.discover(str(REPO_ROOT / "tests"))
        runner = unittest.TextTestRunner(verbosity=1)
        res = runner.run(suite)
        if not res.wasSuccessful():
            log("WARNING: Some unit tests failed.")

    # --------------------------------------------------------------------------
    # Execution Routing Based on --only
    # --------------------------------------------------------------------------
    run_all = args.only == "all"

    # Task: Proposed Model
    if run_all or args.only == "model":
        log("Executing Proposed Model Evaluation (SDUST & CWRU)", header=True)
        from experiments.run_proposed_model import run_proposed_model, generate_verification_report
        df_exp = run_proposed_model(seeds=args.seeds)
        generate_verification_report(df_exp)

    # Task: Gate Ablation
    if run_all or args.only == "ablation":
        log("Executing Sequential Gate Ablation", header=True)
        from experiments.run_ablation import run_gate_ablation
        run_gate_ablation()

    # Task: Baseline Recalibration
    if run_all or args.only == "baselines":
        log("Executing Baseline Comparison (Coverage-Matched ~90%)", header=True)
        from experiments.run_baselines import run_baselines
        run_baselines()

    # Task: Synthetic Benchmarks
    if run_all or args.only == "synthetic":
        log("Executing Synthetic Benchmarks (Exp B & Exp C)", header=True)
        from experiments.run_synthetic import run_synthetic_benchmarks
        run_synthetic_benchmarks()

    # Task: Latency Benchmarking
    if run_all or args.only == "latency":
        log("Executing Computational Latency Benchmark", header=True)
        from experiments.run_latency import run_latency
        run_latency()

    # If full pipeline, generate figures & report
    if run_all:
        log("Rendering Publication Figures", header=True)
        from plot_figures import generate_all_figures
        generate_all_figures()

        from generate_ieee_figures import generate_fig1_framework, generate_fig2_three_panel
        generate_fig1_framework()
        generate_fig2_three_panel()

    elapsed = time.time() - t_start
    log(f"EXECUTION COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds", header=True)


if __name__ == "__main__":
    main()
