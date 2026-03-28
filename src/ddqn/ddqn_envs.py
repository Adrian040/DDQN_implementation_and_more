"""
Entornos y wrappers para DDQN en:
1) MinAtar Breakout
2) Atari Breakout (ALE/Breakout-v5)

Diseño:
- Se transforma la observación a formato channel-first (C, H, W).
- Se registran estadísticas por episodio.
- Atari usa preprocesamiento estándar (grayscale, resize, frame skip y frame stack).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

# Compatibilidad con distintas versiones de Gymnasium
try:
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    def apply_frame_stack(env: gym.Env, num_stack: int) -> gym.Env:
        return FrameStackObservation(env, num_stack)
except ImportError:
    from gymnasium.wrappers import AtariPreprocessing, FrameStack

    def apply_frame_stack(env: gym.Env, num_stack: int) -> gym.Env:
        return FrameStack(env, num_stack)


class ChannelFirstObsWrapper(gym.ObservationWrapper):
    """
    Convierte observaciones tipo HWC -> CHW.
    Si la observación ya viene en CHW, la deja igual.
    También convierte el dtype a uint8 para almacenamiento eficiente.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space

        if not isinstance(old_space, gym.spaces.Box):
            raise TypeError("ChannelFirstObsWrapper requiere observation_space tipo Box.")

        old_shape = old_space.shape
        if len(old_shape) == 3 and old_shape[0] not in (1, 4) and old_shape[-1] <= 32:
            # HWC -> CHW
            new_shape = (old_shape[2], old_shape[0], old_shape[1])
        else:
            new_shape = old_shape

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs = np.array(obs)
        if obs.ndim == 3 and obs.shape[0] not in (1, 4) and obs.shape[-1] <= 32:
            obs = np.transpose(obs, (2, 0, 1))
        return obs.astype(np.uint8)


class MinAtarObsWrapper(gym.ObservationWrapper):
    """
    MinAtar típicamente entrega observaciones binarias en forma HWC.
    Aquí las convertimos a CHW y uint8.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space

        if not isinstance(old_space, gym.spaces.Box):
            raise TypeError("MinAtarObsWrapper requiere observation_space tipo Box.")

        h, w, c = old_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(c, h, w),
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs = np.array(obs, dtype=np.uint8)
        obs = np.transpose(obs, (2, 0, 1))
        return obs


def make_minatar_breakout(seed: int | None = None) -> gym.Env:
    """
    Crea MinAtar Breakout con Gymnasium.
    """
    env = gym.make("MinAtar/Breakout-v1")
    env = MinAtarObsWrapper(env)
    env = RecordEpisodeStatistics(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def make_atari_breakout(seed: int | None = None) -> gym.Env:
    """
    Crea Atari Breakout usando ALE/Gymnasium.
    Se registra ale_py explícitamente para compatibilidad moderna.
    """
    import ale_py

    gym.register_envs(ale_py)

    env = gym.make(
        "ALE/Breakout-v5",
        obs_type="grayscale",
        frameskip=1,
        full_action_space=False,
    )

    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )

    env = apply_frame_stack(env, 4)
    env = ChannelFirstObsWrapper(env)
    env = RecordEpisodeStatistics(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env