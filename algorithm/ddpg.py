import gym
import torch
import torch as th
import numpy as np

from torch.nn import functional as F
from typing import Optional
from gym import spaces
from common.utils import *
from common.preprocessing import *
from common.off_policy_algorithm import OffPolicyAlgorithm
from common.empty_function import EmptyNoise
from common.normalized_actions import NormalizedActions
from common.noise import OrnsteinUhlenbeckActionNoise, GaussianActionNoise


class DDPG(OffPolicyAlgorithm):
    def __init__(
            self,
            env: gym.Env,
            policy: str = "AC",
            per: bool = False,
            dc: bool = False,
            n_critics: int = 1,
            learning_rate: float = 1e-3,
            buffer_size: int = 1_000_000,
            warm_up: int = 500,
            learning_starts: int = 1000,
            max_step: int = 1000,
            batch_size: int = 32,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: int = 4,
            exploration_initial_eps: float = 1.,
            exploration_final_eps: float = 0.2,
            exploration_fraction: float = 0.8,
            action_noise: str = 'OrnsteinUhlenbeck',
            target_policy_noise: float = 0.0,
            target_noise_clip: float = 0.1,
            policy_delay: int = 1,
            tensorboard_log: Optional[str] = "./run",
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            train_or_eval: bool = True,
    ):
        super(DDPG, self).__init__(
            env,
            policy,
            per,
            dc,
            n_critics=n_critics,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            max_step=max_step,
            tau=tau,
            train_freq=train_freq,
            tensorboard_log=tensorboard_log,
            device=device,
            train_or_eval=train_or_eval,
        )
        self.warm_up = warm_up
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.action_noise = action_noise
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay

        # initial noise and noise decay strategy
        self.noise = None
        self.noise_decay = self.exploration_initial_eps

        # initial the actor, critic and their target networks
        self.actor, self.critic = None, None
        self.actor_target, self.critic_target = None, None

        # initial ReplayMemory, step, logger, optimizer
        self.buffer = None
        self.logger = None
        self.optimizer = None

        # initial noise type and normalized action class
        self.get_noise()
        self.action_wrapper = NormalizedActions(self.env.action_space)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        super()._setup_model()

        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        if isinstance(self.env.action_space, spaces.Discrete):
            raise TypeError('The action_space should be gym.BOX in Actor-Critic.')

    def act(self, observation) -> np.ndarray:
        if self.step <= self.warm_up:
            unscaled_action = np.array(self.env.action_space.sample())
        else:
            with th.no_grad():
                observation = preprocess_obs(obs_as_tensor(observation, self.device), self.env.observation_space)
                scaled_action = self.actor(observation).cpu().numpy()
                unscaled_action = self.action_wrapper.unscale_action(scaled_action + self.noise() * self.noise_decay)

        self.step += 1
        self.noise_decay = self.exploration_schedule(self.process_remaining)

        return unscaled_action

    def target_update(self) -> None:
        return

    def train(self) -> None:
        self.call_times += 1

        actor_losses, critic_losses = [], []
        if self.dc and self.call_times % self.buffer.interval == 0:
            index, pre_observations, pre_actions = self.buffer.pre_sample(batch_size=2 * self.batch_size)
            advantage = self.critic(pre_observations, pre_actions.squeeze())[0]
            dc_data = self.buffer.get_dc_data(index, advantage)
            replay_data = self.buffer.dc_process(dc_data)
        else:
            replay_data = self.buffer.sample(batch_size=self.batch_size)

        with th.no_grad():
            # Select action according to policy and add clipped noise
            noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.actor_target(self.extract_features(replay_data.next_observations)) + noise).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(self.critic_target(self.extract_features(replay_data.next_observations), next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        current_q_values = self.critic(self.extract_features(replay_data.observations), replay_data.actions)

        # Compute critic loss
        critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        critic_losses.append(critic_loss.item())

        # Optimize the critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.call_times % self.policy_delay == 0:
            # Compute actor loss
            # ‘-’ means the direction opposite to the gradient decrease
            f_observation = self.extract_features(replay_data.observations)

            actor_loss = -self.critic.q1_forward(f_observation, self.actor(f_observation)).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        if len(actor_losses) > 0:
            self.logger.add_scalar("train/actor_loss", np.mean(actor_losses), self.step)
        self.logger.add_scalar("train/critic_loss", np.mean(critic_losses), self.step)

    def learn(
            self,
            total_timesteps,
            tb_log_name='DDPG'
    ):
        self.initial_log(tb_log_name)
        return super().learn(
            total_timesteps,
            tb_log_name)

    def save(self, algorithm_name):
        torch.save(self.actor.state_dict(), './actor_' + algorithm_name + '.pth')
        torch.save(self.critic.state_dict(), './critic_' + algorithm_name + '.pth')

    def load(self, algorithm_name, path=None):
        actor_path = '/actor_' + algorithm_name + '.pth'
        critic_path = '/critic_' + algorithm_name + '.pth'
        try:
            self.actor.load_state_dict(torch.load('.' + actor_path))
            self.critic.load_state_dict(torch.load('.' + critic_path))
        except FileNotFoundError:
            if not path:
                print('Please check your model-saved path!')
                raise
            else:
                self.q_net.load_state_dict(torch.load(path + actor_path))
                self.q_net_target.load_state_dict(torch.load(path + critic_path))

    def predict(self, observation):
        with th.no_grad():
            observation = preprocess_obs(observation, self.env.observation_space).to(self.device)
            scaled_action = self.actor(observation).cpu().numpy()
            unscaled_action = self.action_wrapper.unscale_action(scaled_action)

            return unscaled_action

    def get_noise(self):
        action_dim = get_action_dim(self.env.action_space)

        assert isinstance(self.action_noise, str), 'The Input noise type should be str.'

        if self.action_noise == 'OrnsteinUhlenbeck':
            self.noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(action_dim), sigma=np.ones(action_dim))

        elif self.action_noise == 'Gaussian':
            self.noise = GaussianActionNoise(mean=np.zeros(action_dim))

        elif not self.action_noise:
            self.noise = EmptyNoise()

        else:
            raise ValueError(f'{self.action_noise} for action noise is not support.')
