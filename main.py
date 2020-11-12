import pdb
import numpy as np


class ParticleNumberLine:
    def __init__(self, x_lim=5, v_lim=1, goal_eps=(0.1, 0.1), seed=None, field=False):
        self.seed = seed
        self.x_lim = x_lim
        self.v_lim = v_lim
        self.goal_eps = goal_eps

        self._A = np.array([[1, 1], [0, 1]])
        self._B = np.array([[1 / 2], [1]])

        self.state = np.zeros(2)
        np.random.seed(seed)
        self.restart()

    def restart(self):
        state = self.sample_state()
        state[-1, 0] = 0
        self.state = state

    def step(self, u):
        s = self.state
        state = self._A @ s + self._B.dot(u)
        # Wobble
        state[-1] = np.random.normal(0, np.abs(state[-1]))
        if np.random.sample() < np.clip((np.abs(state[-1]) - 10)/10, 0, 1)[0]:
            state[-1] = 0.
        self.state = state
        return state[0] + np.random.normal(0, 0 if np.abs(state[-1]) <= 0 else np.abs(state[-1]))

    def sample_state(self):
        return np.random.rand(2, 1) * np.array([self.x_lim, self.v_lim]).reshape(-1, 1)

def main():
    env = ParticleNumberLine()
    # TODO: forward-backward bayesian filtering


if __name__ == "__main__":
    main()