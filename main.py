# main.py
from __future__ import annotations
from pathlib import Path
from stable_baselines3 import PPO

from config import (
    TEST_GRAVITIES, MASS, LENGTH, TOTAL_TIMESTEPS, SEED,
    EVAL_FREQ, N_EVAL_EPISODES, SAVE_GIF, RENDER_WINDOW,
    make_paths_for_gravity,
)
from env_factory import create_env
from train_agent import train_ppo
from logs_and_plots import rollout_and_log, plots_from_rollout
from simulate import run_pygame_simulation

def run_for_gravity(g: float) -> Path:
    print(f"\n=== Training for g = {g} | mass = {MASS} | length = {LENGTH} ===")
    paths = make_paths_for_gravity(g)

    # Create training & eval environments with custom physics
    env = create_env(g)
    eval_env = create_env(g)

    # Train PPO and save best model
    model = train_ppo(
        env=env,
        eval_env=eval_env,
        total_timesteps=TOTAL_TIMESTEPS,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        seed=SEED,
        best_model_dir=str(paths.run_dir),
        gravity=g
    )

    # Load best model if saved; else keep the trained instance
    try:
        model = PPO.load(paths.model_path)
    except Exception:
        pass

    rows = rollout_and_log(model, gravity=g, steps=1000, csv_path=paths.csv_path)
    plots_from_rollout(rows, paths.run_dir)

    with open(paths.meta_path, "w") as f:
        f.write(f"gravity={g}\nmass={MASS}\nlength={LENGTH}\nsteps={TOTAL_TIMESTEPS}\n")

    if SAVE_GIF or RENDER_WINDOW:
        saved_gif = run_pygame_simulation(
            model, gravity=g, steps=600,
            save_gif=SAVE_GIF, gif_path=str(paths.gif_path),
            show_window=RENDER_WINDOW
        )
        if saved_gif:
            print(f"[INFO] GIF saved: {saved_gif}")

    print(f"[DONE] Outputs saved to: {paths.run_dir.resolve()}")
    return paths.run_dir

def main():
    for g in TEST_GRAVITIES:
        run_for_gravity(g)

if __name__ == "__main__":
    main()
