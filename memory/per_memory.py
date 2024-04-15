import random

import torch
import torch as th
import numpy as np

from common.sumTree import SumTree


class PerReplayBuffer(object):
    # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01          # small amount to avoid zero priority
    alpha = 0.6             # [0~1] convert the importance of TD error to priority
    beta = 0.4              # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.      # clipped abs error

    def __init__(self, buffer_size=1_000_000):
        self.tree = SumTree(buffer_size)
        self.memory_size = buffer_size

    def push(self, state, action, reward, next_state, done):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper

        data = (state, action, reward, next_state, done)
        self.tree.add(max_p, data)      # set max priority for new

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        index = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])          # max->1
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()     # for later calculate ISweight
        min_prob = min_prob if min_prob != 0 else 0.00001

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            state, action, reward, next_state, done = data
            states.append(th.from_numpy(state).unsqueeze(0))
            actions.append(action)
            rewards.append(reward)
            next_states.append(th.from_numpy(next_state).unsqueeze(0))
            dones.append(done)
            priorities.append(p)
            index.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(sampling_probabilities / min_prob, -self.beta)
        is_weight /= is_weight.max()

        return index, is_weight, th.cat(states), actions, rewards, th.cat(next_states), dones

    def update(self, index, errors):
        for i, idx in enumerate(index):
            p = (np.abs(errors[i]) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5]
    b = 2
    c = np.power(np.array(a), 2).max()
    print(c)
