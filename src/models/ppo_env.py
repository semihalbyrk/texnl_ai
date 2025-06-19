import gymnasium as gym
import numpy as np

class AssetBalancingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, util_array, capacity, target=0.8, cost=0.01):
        super().__init__()
        self.util = util_array.astype(float)
        self.capacity = capacity
        self.target = target
        self.cost = cost
        n = len(util_array)
        self.action_space = gym.spaces.Discrete(n*n)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(n,))
        self.state = self.util.copy()

    def _move(self, i, j):
        # capacity 0 ise minimal epsilon kullan
        eps = 1e-6
        cap_i = max(self.capacity[i], eps)
        delta = 1 / cap_i          # artık ∞ olmaz
        self.state[i] = max(0, self.state[i] - delta)
        self.state[j] = min(1, self.state[j] + delta)

    def step(self, action):
        i, j = divmod(action, len(self.state))
        self._move(i, j)
        reward = -np.abs(self.state - self.target).sum() - self.cost
        return self.state, reward, False, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.util.copy()
        return self.state, {}
