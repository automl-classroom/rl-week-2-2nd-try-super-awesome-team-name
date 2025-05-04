from __future__ import annotations

from typing import Any

import warnings

import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.environments import MarsRover


class PolicyIteration(AbstractAgent):
    """
    Policy Iteration Agent.

    This agent performs standard tabular policy iteration on an environment
    with known transition dynamics and rewards. The policy is evaluated and
    improved until convergence.

    Parameters
    ----------
    env : MarsRover
        Environment instance. This class is designed specifically for the MarsRover env.
    gamma : float, optional
        Discount factor for future rewards, by default 0.9.
    seed : int, optional
        Random seed for policy initialization, by default 333.
    filename : str, optional
        Path to save/load the policy, by default "policy.npy".
    """

    def __init__(
        self,
        env: MarsRover,
        gamma: float = 0.9,
        seed: int = 333,
        filename: str = "policy.npy",
        **kwargs: dict,
    ) -> None:
        if hasattr(env, "unwrapped"):
            env = env.unwrapped  # type: ignore[assignment]
        self.env = env
        self.seed = seed
        self.filename = filename
        # rng = np.random.default_rng(
        #    seed=self.seed
        # )  # Uncomment and use this line if you need a random seed for reproducibility

        super().__init__(**kwargs)

        self.n_obs = self.env.observation_space.n  # type: ignore[attr-defined]
        self.n_actions = self.env.action_space.n  # type: ignore[attr-defined]
        self.n_states = len(self.env.states)
        # TODO: Get the MDP components (states, actions, transitions, rewards)
        self.S = self.env.states
        self.A = self.env.actions
        self.T = self.env.T
        self.R = self.env.rewards
        self.gamma = gamma
        self.R_sa = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for s_prime in range(self.n_states):
                    self.R_sa[s, a] += self.T[s, a, s_prime] * self.env.rewards[s_prime]

        # TODO: Initialize policy and Q-values
        np.random.seed(self.seed)
        self.pi = np.random.randint(low=0, high=2, size=self.n_states, dtype=int)
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=float)

        self.policy_fitted: bool = False
        self.steps: int = 0

    def predict_action(  # type: ignore[override]
        self, observation: int, info: dict | None = None, evaluate: bool = False
    ) -> tuple[int, dict]:
        """
        Predict an action using the current policy.

        Parameters
        ----------
        observation : int
            The current observation/state.
        info : dict or None, optional
            Additional info passed during prediction (unused).
        evaluate : bool, optional
            Evaluation mode toggle (unused here), by default False.

        Returns
        -------
        tuple[int, dict]
            The selected action and an empty info dictionary.
        """
        # TODO: Return the action according to current policy
        return (self.pi[observation], {})

    def update_agent(self, *args: tuple, **kwargs: dict) -> None:
        """Run policy iteration to compute the optimal policy and state-action values."""
        if not self.policy_fitted:
            self.Q, self.pi, steps = policy_iteration(
                self.Q, self.pi, (self.S, self.A, self.T, self.R_sa, self.gamma)
            )

            self.policy_fitted = True

    def save(self, *args: tuple[Any], **kwargs: dict) -> None:
        """
        Save the learned policy to a `.npy` file.

        Raises
        ------
        Warning
            If the policy has not yet been fitted.
        """
        if self.policy_fitted:
            np.save(self.filename, np.array(self.pi))
        else:
            warnings.warn("Tried to save policy but policy is not fitted yet.")

    def load(self, *args: tuple[Any], **kwargs: dict) -> np.ndarray:
        """
        Load the policy from file.

        Returns
        -------
        np.ndarray
            The loaded policy array.
        """
        self.pi = np.load(self.filename)
        self.policy_fitted = True
        return self.pi


def policy_evaluation(
    pi: np.ndarray,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Perform policy evaluation for a fixed policy.

    Parameters
    ----------
    pi : np.ndarray
        The current policy (array of actions).
    T : np.ndarray
        Transition probabilities T[s, a, s'].
    R_sa : np.ndarray
        Reward matrix R[s, a].
    gamma : float
        Discount factor.
    epsilon : float, optional
        Convergence threshold, by default 1e-8.

    Returns
    -------
    np.ndarray
        The evaluated value function V[s] for all states.
    """
    n_states = R_sa.shape[0]
    V = np.zeros(n_states)

    while True:
        delta = 0
        V_prev = V.copy()
        for s in range(n_states):
            a = pi[s]
            V[s] = sum(
                T[s, a, s_prime] * (R_sa[s, a] + gamma * V_prev[s_prime])
                for s_prime in range(n_states)
            )
            delta = max(delta, abs(V[s] - V_prev[s]))
        if delta < epsilon:
            break
    return V


def policy_improvement(
    V: np.ndarray,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Improve the current policy based on the value function.

    Parameters
    ----------
    V : np.ndarray
        Current value function.
    T : np.ndarray
        Transition probabilities T[s, a, s'].
    R_sa : np.ndarray
        Reward matrix R[s, a].
    gamma : float
        Discount factor.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Q-function and the improved policy.
    """
    n_states, n_actions = R_sa.shape
    Q = np.zeros((n_states, n_actions))
    pi_new = np.zeros(n_states, dtype=int)

    for state in range(n_states):
        for action in range(n_actions):
            Q[state, action] = sum(
                T[state, action, next_state]
                * (R_sa[state, action] + gamma * V[next_state])
                for next_state in range(n_states)
            )
        pi_new[state] = np.argmax(Q[state])

    return Q, pi_new


def policy_iteration(
    Q: np.ndarray,
    pi: np.ndarray,
    MDP: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Full policy iteration loop until convergence.

    Parameters
    ----------
    Q : np.ndarray
        Initial Q-table (can be zeros).
    pi : np.ndarray
        Initial policy.
    MDP : tuple
        A tuple (S, A, T, R_sa, gamma) representing the MDP.
    epsilon : float, optional
        Convergence threshold for value updates, by default 1e-8.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        Final Q-table, final policy, and number of improvement steps.
    """

    S, A, T, R_sa, gamma = MDP
    pi_prev = pi.copy()
    steps = 0

    while True:
        V = policy_evaluation(pi_prev, T, R_sa, gamma, epsilon)
        Q_new, pi_after = policy_improvement(V, T, R_sa, gamma)

        if np.array_equal(pi_after, pi_prev):
            break
        else:
            pi_prev = pi_after
            steps += 1

    return Q_new, pi_after, steps


if __name__ == "__main__":
    algo = PolicyIteration(env=MarsRover())
    algo.update_agent()
