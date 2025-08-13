# simulate.py
from __future__ import annotations
import numpy as np
import pygame
import gymnasium as gym

def run_pygame_simulation(model, gravity: float, steps: int = 600,
                          save_gif: bool = False, gif_path: str = "pendulum_simulation.gif",
                          show_window: bool = False):
    """
    Simple Pygame visualizer for Pendulum-v1. If show_window=False, renders off-screen;
    if save_gif=True and imageio is available, writes a GIF at gif_path.
    """
    try:
        import imageio
    except Exception:
        imageio = None

    env = gym.make("Pendulum-v1", g=gravity)
    obs, _ = env.reset()

    pygame.init()
    width = height = 600
    if show_window:
        screen = pygame.display.set_mode((width, height))
    else:
        screen = pygame.Surface((width, height))
    clock = pygame.time.Clock()

    origin = (width // 2, height // 2 + 100)
    # The visual scale is illustrative; the task is judged by reward during training
    px_per_m = 200

    frames = []
    for t in range(steps):
        if show_window:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    steps = t
                    break

        screen.fill((255, 255, 255))
        theta = np.arctan2(obs[1], obs[0])
        x = origin[0] + int(px_per_m * np.sin(theta))
        y = origin[1] - int(px_per_m * np.cos(theta))
        pygame.draw.line(screen, (0, 0, 0), origin, (x, y), 4)
        pygame.draw.circle(screen, (50, 100, 200), origin, 6)
        pygame.draw.circle(screen, (200, 50, 50), (x, y), 10)

        if show_window:
            pygame.display.flip()
            clock.tick(60)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

        if save_gif and imageio is not None:
            surf = pygame.surfarray.array3d(screen).swapaxes(0, 1)
            frames.append(surf)

    pygame.quit()
    env.close()

    if save_gif and frames and imageio is not None:
        imageio.mimsave(gif_path, frames, fps=30)
        return gif_path
    return None
