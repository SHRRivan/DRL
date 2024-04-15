import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import *


class ActorNetwork(nn.Module):
    def __init__(self, observation_space, action_space, init_w=3e-3):
        super(ActorNetwork, self).__init__()

        state_dim, action_dim = get_input_dim(observation_space, action_space)
        self.action_dim = action_dim

        self.linear1 = nn.Linear(state_dim, 400, bias=True)
        self.linear2 = nn.Linear(400, 300, bias=True)
        self.linear3 = nn.Linear(300, action_dim, bias=True)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        # make action to [-1, 1] avoiding action bound error
        x = self.tanh(self.linear3(x))
        return x


if __name__ == '__main__':
    env = gym.make('BipedalWalkerHardcore-v3')
    # env = gym.make('PongNoFrameskip-v4')
    obs = env.reset()
    net = ActorNetwork(env.observation_space, env.action_space)
    action = net(torch.from_numpy(obs)).detach().numpy()
    print(net.action_dim)
    print(action)

