# env_factory.py
from __future__ import annotations
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from config import MASS, LENGTH, SEED
from stable_baselines3.common.monitor import Monitor

def create_env(gravity: float) -> DummyVecEnv:
    """
    Create a vectorized Pendulum-v1 environment with custom gravity/mass/length.
    """
    def make_env():
        env = gym.make("Pendulum-v1", g=gravity)
        env.unwrapped.m = float(MASS)
        env.unwrapped.l = float(LENGTH)
        if SEED is not None:
            env.reset(seed=SEED)
        env = Monitor(env)
        return env
    return DummyVecEnv([make_env])
