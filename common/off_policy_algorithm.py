import gym
import torch
import numpy as np
import torch as th

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from torch.utils.tensorboard import SummaryWriter
from common.utils import *
from common.empty_function import EmptyWriter

from net.mlp import Mlp
from net.duel import MlpDuel
from net.actor import ActorNetwork
from net.critic import CriticNetwork

from common.preprocessing import preprocess_obs
from common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from memory.memory import ReplayBuffer
from memory.per_memory import PerReplayBuffer


class OffPolicyAlgorithm(object):
    def __init__(
            self,
            env: gym.Env,
            policy: str,
            per: bool = False,
            dc: bool = False,
            dueling: bool = False,
            n_critics: int = 2,
            learning_rate: float = 1e-4,
            buffer_size: int = 1_000_000,
            learning_starts: int = 50000,
            max_step: int = 500,
            tau: float = 1.0,
            train_freq: int = 4,
            target_update_interval: int = 1000,
            tensorboard_log: Optional[str] = None,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            features_extractor: Optional[nn.Module] = None,
            device: Union[th.device, str] = "auto",
            train_or_eval: bool = True,
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = dict()

        self.env = env
        self.policy = policy
        self.per = per
        self.dc = dc
        self.dueling = dueling

        self.n_critics = n_critics
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.max_step = max_step
        self.tau = tau
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.tensorboard_log = tensorboard_log
        self.device = get_device(device)
        self.mode = train_or_eval

        self.features_extractor = features_extractor
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

        self.step = 0
        self.call_times = 0
        self.process_remaining = 1.
        self.logger = None
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps = deque(maxlen=100)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _setup_model(self) -> None:
        # initial feature_extractor
        if self.features_extractor is None:
            self.features_extractor = self.features_extractor_class(self.env.observation_space, **self.features_extractor_kwargs)

        # initial the type of policy
        if self.class_name() == 'DQN':
            if self.policy == "MLP":
                if isinstance(self.env.action_space, gym.spaces.Box):
                    raise TypeError('The dqn algorithm only support Discrete action_space, please change it!')
                if self.dueling:
                    self.q_net = MlpDuel(self.env.observation_space, self.env.action_space).to(self.device)
                    self.q_net_target = MlpDuel(self.env.observation_space, self.env.action_space).to(self.device)
                else:
                    self.q_net = Mlp(self.env.observation_space, self.env.action_space).to(self.device)
                    self.q_net_target = Mlp(self.env.observation_space, self.env.action_space).to(self.device)

                # initial target network's parameter
                polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

                # initial optimizer
                self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

            elif self.policy == "CNN":
                pass

            else:
                print('Now my library only support MLP and CNN policy!')

        elif self.class_name() == 'DDPG' or self.class_name() == 'TD3':
            if self.policy == 'MLP':
                self.actor = ActorNetwork(self.env.observation_space, self.env.action_space).to(self.device)
                self.actor_target = ActorNetwork(self.env.observation_space, self.env.action_space).to(self.device)

                self.critic = CriticNetwork(self.env.observation_space, self.env.action_space,
                                            n_critics=self.n_critics).to(self.device)
                self.critic_target = CriticNetwork(self.env.observation_space, self.env.action_space,
                                                   n_critics=self.n_critics).to(self.device)

                polyak_update(self.actor.parameters(), self.actor_target.parameters(), 1)
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), 1)

                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

            elif self.policy == 'CNN':
                pass

            else:
                print('Now my library only support MLP and CNN policy!')

        else:
            print("The have off policy algorithms are DQN, DDPG, TD3, please check!")

        # initial the type of replay memory
        if self.per:
            self.buffer = PerReplayBuffer(buffer_size=self.buffer_size)
        else:
            self.buffer = ReplayBuffer(self.env.observation_space, self.env.action_space, device=self.device,
                                       buffer_size=self.buffer_size, dc=self.dc)

        # initial the model's method
        self.train_or_eval()

    def act(self, observation) -> None:
        """
        select action according to the greedy policy
        :param:observation
        :return:
        """
        raise NotImplementedError()

    def target_update(self) -> None:
        """
        update the target network at a fixed interval
        :return:
        """
        raise NotImplementedError

    def train(self) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        raise NotImplementedError()

    def learn(self, total_timesteps, tb_log_name='off_policy'):
        episode = 0
        while self.step <= total_timesteps:
            episode_reward = 0
            episode_step = 0
            observation = self.env.reset()
            while True:
                action = self.act(observation)
                next_observation, reward, done, info = self.env.step(action)
                self.process_remaining = 1 - self.step / total_timesteps

                episode_step += 1
                episode_reward += reward

                self.buffer.push(observation, action, reward, next_observation, done)

                observation = next_observation

                if self.step % self.target_update_interval == 0:
                    self.target_update()

                if self.step >= self.learning_starts and self.step % self.train_freq == 0:
                    self.train()

                if done or episode_step >= self.max_step:
                    self.episode_rewards.append(episode_reward)
                    self.episode_steps.append(episode_step)
                    episode += 1
                    self.logger.add_scalar('train/ep_mean_reward', np.mean(self.episode_rewards), self.step)
                    self.logger.add_scalar('train/ep_mean_step', np.mean(self.episode_steps), self.step)
                    print(f'Episode is {episode}.     Episode reward is {episode_reward}')
                    print(f'Training session has finished {self.step  * 100 / total_timesteps:.4f}%.')
                    break

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.env.observation_space, normalize_images=True)
        return self.features_extractor(preprocessed_obs)

    def initial_log(self, tb_log_name):
        if self.tensorboard_log:
            self.logger = SummaryWriter(self.tensorboard_log + '/' + tb_log_name)
        else:
            self.logger = EmptyWriter()

    def train_or_eval(self):
        if self.policy == "MLP":
            if self.class_name() == 'DQN':
                if self.mode:
                    self.q_net.train()
                else:
                    self.q_net.eval()
                self.q_net_target.eval()
            elif self.class_name() == 'DDPG' or self.class_name() == 'TD3':
                if self.mode:
                    self.actor.train()
                    self.critic.train()
                else:
                    self.actor.eval()
                    self.critic.eval()
                self.actor_target.eval()
                self.critic_target.eval()
        else:
            print('Now that these off-policy algorithms do only support MLP.')

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.env.action_space.low, self.env.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.env.action_space.low, self.env.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))
