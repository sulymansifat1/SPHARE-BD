"""Unit test: Verify programmatic synthetic signal synthesis without external datasets."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic.bearing_physics import sample_bearing_population
from synthetic.signal_models import synthesize


def test_synthetic_synthesis():
    cfg_path = REPO_ROOT / "configs" / "synthetic_signal_config.json"
    if not cfg_path.exists():
        print("Config synthetic_signal_config.json not found, skipping.")
        return

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    bearings = sample_bearing_population(42, 2, cfg["bearing_population"])
    assert len(bearings) == 2

    # Synthesize 1 Healthy signal
    sig, info = synthesize(
        bearing=bearings[0],
        fault_class="HEALTHY",
        fault_code=0,
        speed_rpm=1500.0,
        load_N=1000.0,
        recording_index=0,
        variant=0,
        master_seed=42,
        cfg=cfg,
        load_min=800.0,
        load_max=1200.0,
        gen_shift=False,
    )

    assert len(sig) > 0
    assert info["fault_class"] == "HEALTHY"
    assert np.isclose(info["nominal_load"], 1000.0)
    assert np.isclose(info["mean_inst_speed_rpm"], 1500.0, atol=30.0)
    print(f"Synthetic signal generation verified: generated {len(sig)} samples for HEALTHY.")


if __name__ == "__main__":
    test_synthetic_synthesis()
