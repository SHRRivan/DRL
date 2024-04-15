import time
import gym
import torch
import warnings
import numpy as np
import torch as th
import torch.nn as nn

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from torch.utils.tensorboard import SummaryWriter

from net.ppo_net import ActorMlp, CriticMlp, ActorCNN, CriticCNN
from common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from common.utils import get_device, get_schedule_fn, set_random_seed, obs_as_tensor, safe_mean
from common.preprocessing import get_action_dim, preprocess_obs, maybe_transpose
from common.empty_function import EmptyWriter
from common.atari_wrappers import FrameStack
from memory.memory import DictRolloutBuffer, RolloutBuffer
from common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    make_proba_distribution,
)


class OnPolicyAlgorithm(object):
    def __init__(
            self,
            env: gym.Env,
            policy: str,
            learning_rate: float,
            n_steps: int,
            gamma: float,
            gae_lambda: float,
            ent_coef: float,
            vf_coef: float,
            max_grad_norm: float,
            use_rms_prop: bool = False,
            tensorboard_log: Optional[str] = None,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            features_extractor: Optional[nn.Module] = None,
            verbose: int = 0,
            n_envs: int = 1,
            stack_frames: bool = False,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            _init_setup_model: bool = True,
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = dict()

        self.env = env
        self.policy = policy
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_rms_prop = use_rms_prop
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.n_envs = n_envs
        self.stack_frames = stack_frames
        self.seed = seed
        self.device = get_device(device)

        self.features_extractor = features_extractor
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

        self.action_dist = None
        self.ep_info_buffer, self.dones = None, None
        self.rollout_buffer = None
        self.optimizer = None
        self.logger, self.action_noise = None, None
        self.lr_schedule, self.process_remaining = None, 1
        self.observation_space, self.action_space, self.action_dim = None, None, None
        self._last_obs, self._last_episode_starts, self.num_timesteps = None, None, None

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = dict(eps=1e-5) if optimizer_kwargs is None else optimizer_kwargs

        if _init_setup_model:
            self._setup_model()

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _setup_model(self):
        self.space()
        self.setup_lr_schedule()
        self.set_random_seed(self.seed)

        # initial feature_extractor
        if self.features_extractor is None:
            self.features_extractor = self.features_extractor_class(self.env.observation_space, **self.features_extractor_kwargs)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        if self.class_name() == 'PPO' or self.class_name() == 'A2C':
            if self.policy == 'MLP':
                self.actor = ActorMlp(self.observation_space, self.action_space).to(self.device)
                self.critic = CriticMlp(self.observation_space, self.action_space).to(self.device)

            elif self.policy == 'CNN':
                if not self.stack_frames:
                    warnings.warn('Now you use CNN policy but no StackFrames, please ensure this!')
                in_channels = 1 if not self.stack_frames else 4
                out_channels = self.action_dim
                self.actor = ActorCNN(in_channels, out_channels).to(self.device)
                self.critic = CriticCNN(in_channels).to(self.device)

            else:
                print('Please check your policy input. Now support MLP and CNN')
        else:
            print('Now the on policy algorithms only have PPO and A2C!')

        self.actor.apply(self.init_weights)
        self.critic.apply(self.init_weights)

        # eps=1e-5 to avoid Nan in Adam optimizer
        parameters = [{'params': self.actor.parameters()}, {'params': self.critic.parameters()}]
        self.optimizer = self.optimizer_class(parameters, lr=self.lr_schedule(1), **self.optimizer_kwargs)

    def get_action(self, obs_tensor: torch.Tensor):
        """
        take action from the policy net
        :return:
        """
        raise NotImplementedError()

    def train(self) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        raise NotImplementedError()

    def trans_state(self, obs):
        obs = np.array(obs)
        obs = np.transpose(obs, (2, 0, 1))  # 将2轴放在0轴之前
        return obs.reshape((1, 4, 84, 84))

    def collect_rollouts(
            self,
            env: gym.Env,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        n_steps = 0
        new_obs = None
        # rollout_buffer reset for on-policy algorithm
        rollout_buffer.reset()

        while n_steps < n_rollout_steps:
            if self.dones:
                self._last_obs = self.env.reset()

            # TODO change this probability to 'may_transpose'
            if self.policy == 'CNN':
                self._last_obs = self.trans_state(self._last_obs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.get_action(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.dones = dones
            self.num_timesteps += self.n_envs

            self._update_info_buffer(infos)
            n_steps += 1
            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate([dones]):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = infos[idx]["terminal_observation"]
                    if self.policy == 'CNN':
                        terminal_obs = self.trans_state(terminal_obs)

                    terminal_obs = obs_as_tensor(terminal_obs, self.device)
                    with th.no_grad():
                        terminal_value = self.critic(preprocess_obs(terminal_obs, self.observation_space))[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        # this is a one-computation
        with th.no_grad():
            if self.policy == 'CNN':
                new_obs = self.trans_state(new_obs)

            # Compute value for the last timestep
            values = self.critic(preprocess_obs(obs_as_tensor(new_obs, self.device), self.observation_space))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        return True

    def learn(
            self,
            total_timesteps: int,
            log_interval: int = 5,
            eval_env: Optional = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyAlgorithm",
            reset_num_timesteps: bool = True,

    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps = self._setup_learn(total_timesteps, eval_env, tb_log_name=tb_log_name)

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self.process_remaining = 1.0 - float(self.num_timesteps) / float(self._total_timesteps)

            # display training infos
            if log_interval is not None and iteration % log_interval == 0:
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
                    ep_len_mean = safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])

                    self.logger.add_scalar("rollout/ep_rew_mean", ep_rew_mean, self.num_timesteps)
                    self.logger.add_scalar("rollout/ep_len_mean", ep_len_mean, self.num_timesteps)
                    print(f'ep_rew_mean = {ep_rew_mean},        ep_len_mean = {ep_len_mean}')
                    print(f'Training session has finished {self.num_timesteps  * 100 / total_timesteps:.4f}%.')

            self.train()
        return self

    def _setup_learn(
            self,
            total_timesteps: int,
            eval_env: Optional,
            reset_num_timesteps: bool = True,
            tb_log_name: str = "run",
    ) -> int:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """
        self.start_time = time.time()

        self.dones = True

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
            self._last_episode_starts = np.ones((self.n_envs,), dtype=bool)

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        # Configure logger's outputs if no logger was passed
        self.initial_log(tb_log_name)

        return total_timesteps

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def _get_action_dist_from_latent(self, features: th.Tensor):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.actor(features)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.actor.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        #     return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        distribution = self._get_action_dist_from_latent(features)
        log_prob = distribution.log_prob(actions)
        values = self.critic(features)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, features: th.Tensor):
        """
        Get the current policy distribution given the observations.

        :param features:
        :return: the action distribution.
        """
        return self._get_action_dist_from_latent(features)

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

    def space(self):
        if self.stack_frames:
            self.env = FrameStack(self.env, 4)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.action_dim = get_action_dim(self.action_space)

        # action distribution initialization
        self.action_dist = make_proba_distribution(self.action_space, use_sde=False, dist_kwargs=None)

    def setup_lr_schedule(self):
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
        self.action_space.seed(seed)
        if self.env is not None:
            self.env.seed(seed)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return np.clip(low + (0.5 * (scaled_action + 1.0) * (high - low)), low, high)

    def set_training_mode(self, state: bool):
        if state:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def save(self, algorithm_name):
        torch.save(self.actor.state_dict(), './actor_' + algorithm_name + '.pth')
        torch.save(self.critic.state_dict(), './critic_' + algorithm_name + '.pth')
