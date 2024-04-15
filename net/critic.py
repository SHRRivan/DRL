import gym
import torch
import torch as th
import torch.nn as nn
import numpy as np

from typing import Type, List
from common.utils import get_input_dim
from common.torch_layers import create_mlp
from common.preprocessing import preprocess_obs


# class CriticNetwork(nn.Module):
#     def __init__(self, observation_space, action_space, n_critics=2, init_w=3e-3):
#         super(CriticNetwork, self).__init__()
#
#         state_dim, action_dim = get_input_dim(observation_space, action_space)
#
#         self.n_critics = n_critics
#         self.q_networks = []
#
#         self.linear1 = nn.Linear(state_dim + action_dim, 400, bias=True)
#         self.linear2 = nn.Linear(400, 300, bias=True)
#         self.linear3 = nn.Linear(300, 1, bias=True)
#
#         self.linear3.weight.data.uniform_(-init_w, init_w)
#         self.linear3.bias.data.uniform_(-init_w, init_w)
#
#         self.relu = nn.ReLU()
#
#     def forward(self, state, action):
#         x = torch.cat([state, action], 1)
#         x = self.relu(self.linear1(x))
#         x = self.relu(self.linear2(x))
#         x = self.linear3(x)
#         return x,


class CriticNetwork(nn.Module):
    def __init__(
            self,
            observation_space,
            action_space,
            net_arch: List[int] = [400, 300],
            n_critics: int = 2,
            activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super(CriticNetwork, self).__init__()
        self.observation_space = observation_space

        state_dim, action_dim = get_input_dim(observation_space, action_space)

        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_mlp(state_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, state: th.Tensor, action: th.Tensor):
        value_input = torch.cat([state, action], 1)
        return tuple(q_net(value_input) for q_net in self.q_networks)

    def q1_forward(self, state: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            state = preprocess_obs(state, self.observation_space)
        return self.q_networks[0](th.cat([state, actions], dim=1))


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    # env = gym.make('PongNoFrameskip-v4')
    obs = env.reset()
    a = env.action_space.sample()
    net = CriticNetwork(env.observation_space, env.action_space, net_arch=[400, 300], n_critics=2)

    obs = torch.from_numpy(np.array([obs]))
    a = torch.from_numpy(np.array([a]))

    value = net(obs, a)
    value_1 = net.q1_forward(obs, a)
    print(value_1)
