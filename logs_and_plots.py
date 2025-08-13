# logs_and_plots.py
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from config import MASS, LENGTH

def rollout_and_log(model, gravity: float, steps: int, csv_path: Path) -> np.ndarray:
    """
    Run a deterministic rollout with the trained policy and log to CSV.
    Returns a numpy array with columns: [t, theta, theta_dot, torque, reward]
    """
    env = gym.make("Pendulum-v1", g=gravity)
    obs, _ = env.reset()

    header = ["t", "theta", "theta_dot", "torque", "reward"]
    rows = []
    for t in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        theta = float(np.arctan2(obs[1], obs[0]))  # recover angle from sin/cos
        theta_dot = float(obs[2])
        torque = float(action[0])
        rows.append([t, theta, theta_dot, torque, float(reward)])
        if terminated or truncated:
            obs, _ = env.reset()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    env.close()
    return np.array(rows, dtype=float)

def plots_from_rollout(rows: np.ndarray, outdir: Path) -> None:
    """
    Save basic QA plots: angle, angular velocity, torque histogram, reward.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Angle
    fig, ax = plt.subplots()
    ax.plot(rows[:, 0], rows[:, 1])
    ax.set(title="Angle (rad) over time", xlabel="timestep", ylabel="theta (rad)")
    ax.grid(True)
    fig.savefig(outdir / "angle_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Angular velocity
    fig, ax = plt.subplots()
    ax.plot(rows[:, 0], rows[:, 2])
    ax.set(title="Angular velocity over time", xlabel="timestep", ylabel="theta_dot (rad/s)")
    ax.grid(True)
    fig.savefig(outdir / "angular_velocity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Torque histogram
    fig, ax = plt.subplots()
    ax.hist(rows[:, 3], bins=30)
    ax.set(title="Torque distribution", xlabel="torque", ylabel="frequency")
    ax.grid(True)
    fig.savefig(outdir / "torque_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Reward
    fig, ax = plt.subplots()
    ax.plot(rows[:, 0], rows[:, 4])
    ax.set(title="Reward over time", xlabel="timestep", ylabel="reward")
    ax.grid(True)
    fig.savefig(outdir / "reward_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
