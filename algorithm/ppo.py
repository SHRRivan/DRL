import gym
import torch
import warnings

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gym import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from samples.ReplayBufferSamples import Schedule
from common.on_policy_algorithm import OnPolicyAlgorithm
from common.utils import get_schedule_fn, explained_variance, obs_as_tensor
from common.preprocessing import preprocess_obs


class PPO(OnPolicyAlgorithm):
    def __init__(
            self,
            env,
            policy: str,
            learning_rate: float = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            normalize_images: bool = False,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            verbose: int = 0,
            n_envs: int = 1,
            stack_frames: bool = False,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        super().__init__(
            env,
            policy,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            n_envs=n_envs,
            stack_frames=stack_frames,
            device=device,
            seed=seed,
            _init_setup_model=False
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.n_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.n_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.n_envs})"
                )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.normalize_images = normalize_images
        self.target_kl = target_kl
        self._n_updates = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        # TODO figure these below out
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " \
                                               "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def get_action(self, obs_tensor: torch.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = preprocess_obs(obs_tensor, self.observation_space)
        values = self.critic(features)
        distribution = self._get_action_dist_from_latent(features)
        actions = distribution.get_actions(deterministic=False)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def train(self):
        # Switch to train mode (this affects batch norm / dropout)
        self.set_training_mode(True)
        # Compute current clip range
        clip_range = self.clip_range(self.process_remaining)
        # Optional: clip range for the value function
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self.process_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # if self.use_sde:
                #     self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.add_scalar("train/entropy_loss", np.mean(entropy_losses), self.num_timesteps)
        self.logger.add_scalar("train/policy_gradient_loss", np.mean(pg_losses), self.num_timesteps)
        self.logger.add_scalar("train/value_loss", np.mean(value_losses), self.num_timesteps)
        self.logger.add_scalar("train/approx_kl", np.mean(approx_kl_divs), self.num_timesteps)
        self.logger.add_scalar("train/clip_fraction", np.mean(clip_fractions), self.num_timesteps)
        self.logger.add_scalar("train/loss", loss.item(), self.num_timesteps)
        self.logger.add_scalar("train/explained_variance", explained_var, self.num_timesteps)
        if hasattr(self.actor, "log_std"):
            self.logger.add_scalar("train/std", th.exp(self.actor.log_std).mean().item(), self.num_timesteps)

        # self.logger.add_scalar("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.add_scalar("train/clip_range", clip_range, self.num_timesteps)
        if self.clip_range_vf is not None:
            self.logger.add_scalar("train/clip_range_vf", clip_range_vf, self.num_timesteps)

    def learn(
            self,
            total_timesteps: int,
            tb_log_name: str = "PPO",
            reset_num_timesteps: bool = True,
            *args,
            **kwargs
    ) -> "OnPolicyAlgorithm":
        return super().learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
        )

    def load(self, algorithm_name, path=None):
        actor_path = '/actor_' + algorithm_name + '.pth'
        critic_path = '/critic_' + algorithm_name + '.pth'
        try:
            self.actor.load_state_dict(torch.load('.' + actor_path))
            self.critic.load_state_dict(torch.load('.' + critic_path))
        except FileNotFoundError:
            pass

    def predict(self, observation):
        with th.no_grad():
            features = preprocess_obs(obs_as_tensor(observation, self.device), self.observation_space)
            scaled_action = self.get_distribution(features).get_actions().cpu().numpy()

            if isinstance(self.action_space, spaces.Box):
                unscaled_action = np.clip(scaled_action, self.action_space.low, self.action_space.high)

            else:
                unscaled_action = scaled_action

        return unscaled_action
