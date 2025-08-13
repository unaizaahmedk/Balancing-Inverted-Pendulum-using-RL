# train_agent.py
from __future__ import annotations
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import os
from pathlib import Path

class LiveRewardPlotCallback(BaseCallback):
    def __init__(self, run_dir: str, gravity: float, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.gravity = gravity
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label="Episode Reward")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.set_title(f"Live Training Reward (g={gravity})")
        self.ax.grid(True)
        plt.ion()
        plt.show()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not isinstance(infos, list):  # Non-vector env case
            infos = [infos]
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.line.set_xdata(range(len(self.episode_rewards)))
                self.line.set_ydata(self.episode_rewards)
                self.ax.relim()
                self.ax.autoscale_view()
                self.fig.canvas.draw()
                plt.pause(0.01)
        return True

    def _on_training_end(self) -> None:
        plt.ioff()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.run_dir / f"training_rewards_g{self.gravity}_{timestamp}.png"
        self.fig.savefig(save_path)
        print(f"Training reward plot saved: {save_path}")
        plt.show()


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, reward_threshold=-200, n_episodes=20, verbose=0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.n_episodes = n_episodes
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if len(self.locals["infos"]) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    if len(self.episode_rewards) > self.n_episodes:
                        self.episode_rewards.pop(0)
                    if len(self.episode_rewards) == self.n_episodes:
                        avg_reward = sum(self.episode_rewards) / self.n_episodes
                        if avg_reward >= self.reward_threshold:
                            print(f"\nEarly stopping: Average reward {avg_reward:.2f} >= {self.reward_threshold}")
                            return False
        return True


def train_ppo(env, eval_env, total_timesteps: int, eval_freq: int,
              n_eval_episodes: int, seed: int | None,
              best_model_dir: str, gravity: float) -> PPO:
    """
    Train PPO with EvalCallback, live reward plot saved in run_dir,
    and early stopping if avg reward >= threshold.
    """
    from pathlib import Path
    eval_env = Monitor(eval_env.envs[0])

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=best_model_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    policy_kwargs = dict(net_arch=[64, 64])
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=seed,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        gamma=0.98,
        batch_size=256,
        n_steps=2048,
        ent_coef=0.0,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            eval_cb,
            LiveRewardPlotCallback(run_dir=best_model_dir, gravity=gravity),
            EarlyStoppingCallback()
        ],
        progress_bar=True
    )
    return model
