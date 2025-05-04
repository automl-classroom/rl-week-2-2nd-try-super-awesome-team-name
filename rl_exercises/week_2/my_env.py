from __future__ import annotations

import gymnasium as gym
import numpy as np


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
    ):
        """Initializes the observation and action space for the environment."""
        self.position = 0
        self.current_steps = 0

        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

        self.states = np.arange(2)

    def reset(
        self,
        seed: int | None = None,
    ):
        self.position = 0
        self.current_steps = 0
        return self.position, {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")
        self.current_steps += 1
        if action == 1:
            self.position = 1
        elif action == 0:
            self.position = 0
        return self.position, float(action), False, False, {}

    def get_reward_per_action(self) -> np.ndarray:
        nS, nA = self.observation_space.n, self.action_space.n
        R = np.zeros((nS, nA), dtype=float)
        for s in range(nS):
            for a in range(nA):
                R[s, a] = float(a)
        return R

    def get_transition_matrix(
        self,
    ) -> np.ndarray:
        nS, nA = len(self.states), self.action_space.n
        T = np.zeros((nS, nA, nS), dtype=float)
        for s in self.states:
            for a in range(self.action_space.n):
                s_next = 1 if a == 1 else 0
                T[s, a, s_next] = float(1)
        return T


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        super().__init__(env)
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed: int | None = None,
    ):
        true_obs, info = self.env.reset(seed=seed)
        return self._noisy_obs(true_obs), info

    def step(self, action: int):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._noisy_obs(next_obs), reward, terminated, truncated, info

    def _noisy_obs(self, true_obs: int) -> int:
        if self.rng.random() < self.noise:
            n = self.observation_space.n
            others = [s for s in range(n) if s != true_obs]
            return int(self.rng.choice(others))
        else:
            return int(true_obs)
