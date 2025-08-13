# config.py
"""
Custom configuration for the inverted pendulum RL project.

- Runs a test suite for g in {0.0, 9.8, 1.6}
- Uses custom MASS and LENGTH here 
- Controls training timesteps, seeds, output layout, and optional GIF rendering
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# ========= Physics Parameters =========
TEST_GRAVITIES = [0.0, 9.8, 1.6]  # Required: g = 0, 9.8, 1.6
MASS: float = 1.2                  # <- your custom mass (kg)
LENGTH: float = 0.75               # <- your custom length (meters)

# ========= Training/Logging =========
TOTAL_TIMESTEPS: int = 150_000  # Total number of training timesteps for the RL agent.
SEED: int | None = 42           # Random seed for reproducibility (set to None for non-deterministic results).
EVAL_FREQ: int = 10_000         # How often (in timesteps) to run evaluations during training.
N_EVAL_EPISODES: int = 5        # Number of episodes to run per evaluation cycle.

SAVE_GIF: bool = True              
RENDER_WINDOW: bool = True       

# ========= Paths =========
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class RunPaths:
    run_dir: Path
    model_path: Path
    csv_path: Path
    meta_path: Path
    gif_path: Path

def make_paths_for_gravity(g: float) -> RunPaths:
    """Create a fresh output directory for a specific gravity value."""
    safe_g = str(g).replace(".", "_")
    run_dir = RESULTS_DIR / f"{RUN_STAMP}_g{safe_g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        model_path=run_dir / "ppo_pendulum_best_model.zip",
        csv_path=run_dir / "rollout.csv",
        meta_path=run_dir / "meta.txt",
        gif_path=run_dir / "simulation.gif",
    )
