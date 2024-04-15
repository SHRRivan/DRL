import gym
import torch
import torch.nn as nn
import numpy as np

from common.utils import *
from common.preprocessing import *
from gym import spaces


class Mlp(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=64):
        super(Mlp, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        input_dim, out_dim = get_input_dim(self.observation_space, self.action_space)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out = nn.Linear(hidden_size, out_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return self.out(x)


if __name__ == '__main__':
    # env = MySim_D()
    # env.set_mode(True)
    env = gym.make('CartPole-v0')
    # env = gym.make('PongNoFrameskip-v4')
    obs = env.reset()
    action = env.action_space.sample()
    action = np.array(action)
    print(action)
    s_, _, _, _ = env.step(action)
