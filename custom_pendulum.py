# custom_pendulum.py
from __future__ import annotations
import gymnasium as gym
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from config import MASS, LENGTH

class CustomPendulumEnv(PendulumEnv):
    """
    Thin wrapper over Gymnasium's PendulumEnv that enforces custom mass/length.
    Gravity 'g' is passed when constructing via gym.make("Pendulum-v1", g=...).
    """
    def __init__(self, g: float = 9.8):
        super().__init__(g=g)
        # Set custom physical parameters
        self.m = float(MASS)
        self.l = float(LENGTH)

def make_custom_env(gravity: float) -> gym.Env:
    """
    Construct a Pendulum-v1 env with given gravity, then set MASS/LENGTH.
    """
    env = gym.make("Pendulum-v1", g=gravity)
    env.unwrapped.m = float(MASS)
    env.unwrapped.l = float(LENGTH)
    return env
