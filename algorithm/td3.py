import gym
import torch
import torch as th
import numpy as np

from typing import Optional
from gym import spaces
from common.utils import *
from common.preprocessing import *

from algorithm.ddpg import DDPG


class TD3(DDPG):
    def __init__(
            self,
            env: gym.Env,
            policy: str = "AC",
            per: bool = False,
            dc: bool = False,
            n_critics: int = 2,
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
            exploration_final_eps: float = 0.001,
            exploration_fraction: float = 0.9,
            action_noise: str = 'OrnsteinUhlenbeck',
            # Remove all tricks from TD3 to obtain DDPG:
            # we still need to specify target_policy_noise > 0 to avoid errors
            target_policy_noise: float = 0.2,
            target_noise_clip: float = 0.5,
            policy_delay: int = 2,
            tensorboard_log: Optional[str] = "./run",
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            train_or_eval: bool = True,
    ):
        super(TD3, self).__init__(
            env,
            policy,
            per,
            dc,
            n_critics=n_critics,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            warm_up=warm_up,
            learning_starts=learning_starts,
            max_step=max_step,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            exploration_fraction=exploration_fraction,
            action_noise=action_noise,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            policy_delay=policy_delay,
            tensorboard_log=tensorboard_log,
            device=device,
            _init_setup_model=False,
            train_or_eval=train_or_eval,
        )

        if _init_setup_model:
            self._setup_model()

    def learn(
            self,
            total_steps,
            tb_log_name='TD3'
    ):
        self.initial_log(tb_log_name)
        return super().learn(
            total_steps,
            tb_log_name)
