from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

try:
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    def apply_frame_stack(env: gym.Env, num_stack: int) -> gym.Env:
        return FrameStackObservation(env, num_stack)
except ImportError:
    from gymnasium.wrappers import AtariPreprocessing, FrameStack

    def apply_frame_stack(env: gym.Env, num_stack: int) -> gym.Env:
        return FrameStack(env, num_stack)


class ChannelFirstObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space
        old_shape = old_space.shape

        if len(old_shape) == 3 and old_shape[0] not in (1, 4) and old_shape[-1] <= 32:
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
    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(c, h, w),
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs = np.array(obs, dtype=np.uint8)
        return np.transpose(obs, (2, 0, 1))


def make_minatar_breakout(seed: int | None = None) -> gym.Env:
    env = gym.make("MinAtar/Breakout-v1")
    env = MinAtarObsWrapper(env)
    env = RecordEpisodeStatistics(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def make_atari_breakout(seed: int | None = None) -> gym.Env:
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