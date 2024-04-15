import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import *
from common.preprocessing import *
from gym import spaces


class MlpDuel(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=64):
        super(MlpDuel, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        input_dim, out_dim = get_input_dim(self.observation_space, self.action_space)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.a = nn.Linear(hidden_size, out_dim, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        a = self.a(x)
        v = self.v(x)
        return v + a - torch.mean(a, dim=-1, keepdim=True), v
