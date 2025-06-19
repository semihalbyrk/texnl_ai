import gymnasium as gym
import numpy as np

class AssetBalancingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, util_array, capacity, names=None, target=0.8, cost=0.01):
        super().__init__()
        self.util = util_array.astype(float)
        self.capacity = capacity
        self.names = names if names is not None else [f"SP{i}" for i in range(len(util_array))]
        self.target = target
        self.cost = cost
        n = len(util_array)
        self.action_space = gym.spaces.Discrete(n * n)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(n,))
        self.state = self.util.copy()
        self.last_i = None
        self.last_j = None

    def _move(self, i, j):
        eps = 1e-6
        cap_i = max(self.capacity[i], eps)
        delta = 1 / cap_i
        self.state[i] = max(0, self.state[i] - delta)
        self.state[j] = min(1, self.state[j] + delta)
        self.last_i = i
        self.last_j = j

    def step(self, action):
        i, j = divmod(action, len(self.state))
        self._move(i, j)
        reward = -np.abs(self.state - self.target).sum() - self.cost
        return self.state, reward, False, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.util.copy()
        self.last_i = None
        self.last_j = None
        return self.state, {}

    def get_last_move_str(self):
        try:
            return f"{self.names[self.last_i]} → {self.names[self.last_j]}"
        except:
            return None
