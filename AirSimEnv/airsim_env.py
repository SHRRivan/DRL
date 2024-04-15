import numpy as np
import airsim

import gym
from gym import spaces


class AirSimEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
            self,
            image_shape
    ):
        self.shape = image_shape
        self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def __del__(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _compute_reward(self):
        raise NotImplementedError()

    # def close(self):
    #     raise NotImplementedError()

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError()

    def render(self, mode='human'):
        return self._get_obs()
