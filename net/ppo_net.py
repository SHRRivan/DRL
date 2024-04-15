import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from common.utils import *
from gym import spaces


class ActorMlp(nn.Module):
    def __init__(self, observation_space, action_space, log_std_init=0):
        super(ActorMlp, self).__init__()

        state_dim, action_dim = get_input_dim(observation_space, action_space)
        self.action_dim = action_dim

        # Initial action std
        self.log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)

        self.linear1 = nn.Linear(state_dim, 400, bias=True)
        self.linear2 = nn.Linear(400, 300, bias=True)
        self.actor = nn.Linear(300, action_dim, bias=True)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))

        # action actually is the mean value of "action"
        action = self.actor(x)
        return action


class CriticMlp(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CriticMlp, self).__init__()

        state_dim, _ = get_input_dim(observation_space, action_space)

        self.linear1 = nn.Linear(state_dim, 400, bias=True)
        self.linear2 = nn.Linear(400, 300, bias=True)
        self.critic = nn.Linear(300, 1, bias=True)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))

        critic = self.critic(x)
        return critic


class ActorCNN(nn.Module):
    def __init__(self, in_channels, out_channels, log_std_init=0):
        super(ActorCNN, self).__init__()
        # Initial action std
        self.log_std = nn.Parameter(th.ones(out_channels) * log_std_init, requires_grad=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.fc4 = nn.Linear(64 * 7 * 7, 512, bias=True)
        self.fc5 = nn.Linear(512, out_channels, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.fc4(x.view(x.size(0), -1)))
        x = self.fc5(x)
        return x


class CriticCNN(nn.Module):
    def __init__(self, in_channels):
        super(CriticCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.fc4 = nn.Linear(64 * 7 * 7, 512, bias=True)
        self.fc5 = nn.Linear(512, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.fc4(x.view(x.size(0), -1)))
        x = self.fc5(x)
        return x


if __name__ == '__main__':
    env = gym.make('BipedalWalkerHardcore-v3')
    # env = gym.make('CartPole-v1')
    print(env.action_space)
    # env = gym.make('PongNoFrameskip-v4')
    obs = env.reset()
    net = ActorMlp(env.observation_space, env.action_space)
    a, v = net(torch.from_numpy(obs))

    print(net.action_dim)
    print(a)
    print(v)
