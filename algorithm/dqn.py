import gym
import torch
import torch as th
import numpy as np
from torch.nn import functional as F

from gym import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from samples import ReplayBufferSamples
from common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from common.off_policy_algorithm import OffPolicyAlgorithm
from common.utils import *
from common.preprocessing import *


class DQN(OffPolicyAlgorithm):
    def __init__(
            self,
            env: gym.Env,
            policy: str,
            per: bool = False,
            dc: bool = False,
            dueling: bool = False,
            double: bool = False,
            learning_rate: float = 1e-4,
            buffer_size: int = 1_000_000,
            learning_starts: int = 5000,
            max_step: int = 1000,
            batch_size: int = 32,
            tau: float = 1.0,
            gamma: float = 0.99,
            train_freq: int = 4,
            target_update_interval: int = 10000,
            exploration_fraction: float = 0.1,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.05,
            max_grad_norm: float = 0.5,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            features_extractor: Optional[nn.Module] = None,
            tensorboard_log: Optional[str] = "run",
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            train_or_eval: bool = True,
    ):
        super(DQN, self).__init__(
            env,
            policy,
            per,
            dc,
            dueling,
            double,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            max_step=max_step,
            tau=tau,
            train_freq=train_freq,
            target_update_interval=target_update_interval,
            tensorboard_log=tensorboard_log,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            features_extractor=features_extractor,
            device=device,
            train_or_eval=train_or_eval,
        )
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm

        self.features_extractor = None

        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 1.
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None

        # initial ReplayMemory, step, logger, optimizer
        self.buffer = None
        self.logger = None
        self.optimizer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        super()._setup_model()

        # exploration schedule
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def act(self, observation) -> np.ndarray:      # no ability for DQN to put constant action
        if self.exploration_rate > np.random.uniform():
            action = np.array(self.env.action_space.sample())
        else:
            with th.no_grad():
                observation = preprocess_obs(obs_as_tensor(observation, self.device), self.env.observation_space)
                actions = self.q_net(observation)
                action = torch.argmax(actions).cpu().numpy()

        self.step += 1
        self.exploration_rate = self.exploration_schedule(self.process_remaining)
        self.logger.add_scalar('rate/exploration', self.exploration_rate, self.step)

        return action

    def target_update(self) -> None:
        polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

    def train(self) -> None:
        self.call_times += 1

        losses = []
        replay_data = self.buffer.sample(batch_size=self.batch_size)
        # If env.observation_space belongs to gym.discrete, transform it to one-hot vector
        # Actually it hardly occurs
        # observations, next_observations = self._is_discrete_observation(replay_data)

        with th.no_grad():
            # Compute the next Q-values using the target network
            next_q_values = self.q_net_target(self.extract_features(replay_data.next_observations))
            # Follow greedy policy: use the one with the highest value
            next_q_values, _ = next_q_values.max(dim=1)
            # Avoid potential broadcast issue
            next_q_values = next_q_values.reshape(-1, 1)
            # 1-step TD target
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates
        current_q_values = self.q_net(self.extract_features(replay_data.observations))

        # Retrieve the q-values for the actions from the replay buffer
        current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

        # Compute Huber loss (less sensitive to outliers)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        losses.append(loss.item())
        self.logger.add_scalar('train/loss', np.mean(losses), self.step)

        # Optimize the policy
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm
        th.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)

        self.optimizer.step()

    def per_train(self):
        index, is_weights, states, actions, rewards, next_states, dones = self.buffer.sample(
            batch_size=self.batch_size)

        is_weights = torch.FloatTensor(is_weights).to(self.device)
        loss = (is_weights * self.compute_td_loss(index, states, actions, rewards, next_states, dones)).mean()

        self.logger.add_scalar('train/loss', loss.item(), self.step)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)

        self.optimizer.step()

    def compute_td_loss(self, idxs, states, actions, rewards, next_states, is_done):
        """ Compute td loss using torch operations only. Use the formula above. """
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = torch.tensor(actions).long().unsqueeze(1).to(self.device)              # shape: [batch_size]
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(self.device)  # shape: [batch_size]
        is_done = torch.tensor(is_done).long().unsqueeze(1).to(self.device)              # shape: [batch_size]

        # get q-values for all actions in current states
        predicted_values_for_actions = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            # compute q-values for all actions in next states
            predicted_next_values = self.q_net_target(next_states)

            # compute V*(next_states) using predicted next q-values
            next_state_values = predicted_next_values.max(-1)[0].unsqueeze(1)

            # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
            target_values_for_actions = rewards + (1 - is_done) * self.gamma * next_state_values

        # mean squared error loss to minimize
        errors = (predicted_values_for_actions - target_values_for_actions).cpu().squeeze().tolist()

        self.buffer.update(idxs, errors)

        loss = F.smooth_l1_loss(predicted_values_for_actions, target_values_for_actions)

        return loss

    def learn(
            self,
            total_timesteps,
            tb_log_name='DQN'
    ):
        self.initial_log(tb_log_name)
        return super().learn(
            total_timesteps,
            tb_log_name)

    # TODO: better method to save and load model
    def save(self, algorithm_name):
        torch.save(self.q_net.state_dict(), './eval_' + algorithm_name + '.pth')
        torch.save(self.q_net_target.state_dict(), './target_' + algorithm_name + '.pth')

    def load(self, algorithm_name, path=None):
        eval_path = '/eval_' + algorithm_name + '.pth'
        target_path = '/target_' + algorithm_name + '.pth'
        try:
            self.q_net.load_state_dict(torch.load('.' + eval_path))
            self.q_net_target.load_state_dict(torch.load('.' + target_path))
        except FileNotFoundError:
            if not path:
                print('Please check your model-saved path!')
                raise
            else:
                self.q_net.load_state_dict(torch.load(path + eval_path))
                self.q_net_target.load_state_dict(torch.load(path + target_path))

    def predict(self, observation: np.ndarray):
        with th.no_grad():
            observation = preprocess_obs(obs_as_tensor(observation, self.device), self.env.observation_space)
            actions = self.q_net(observation)
            action = torch.argmax(actions).item()
            return action

    # def _is_discrete_observation(self, data: ReplayBufferSamples):
    #     if isinstance(self.env.observation_space, spaces.Discrete):
    #         observations = preprocess_obs(
    #             data.observations.cpu().squeeze().numpy(),
    #             self.env.observation_space).to(self.device)
    #         next_observations = preprocess_obs(
    #             data.observations.cpu().squeeze().numpy(),
    #             self.env.observation_space).to(self.device)
    #     else:
    #         observations, next_observations = data.observations, data.next_observations
    #
    #     return observations, next_observations


if __name__ == '__main__':
    # e = MySim_D()
    # e.set_mode(True)
    e = gym.make('CartPole-v0')
    model = DQN(env=e, policy="MLP", learning_starts=500)
    # o = e.reset()
    # a = model.act(o)
