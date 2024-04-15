import torch

import numpy as np
import torch as th
import torch.nn.functional as F

from gym import spaces
from typing import Any, Dict, Optional, Union, Tuple

from common.on_policy_algorithm import OnPolicyAlgorithm
from common.utils import explained_variance, obs_as_tensor
from common.preprocessing import preprocess_obs


class A2C(OnPolicyAlgorithm):
    def __init__(
            self,
            env,
            policy: str,
            learning_rate: float = 7e-4,
            n_steps: int = 5,
            gamma: float = 0.99,
            gae_lambda: float = 1.0,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_rms_prop: bool = True,
            rms_prop_eps: float = 1e-5,
            normalize_advantage: bool = False,
            tensorboard_log: Optional[str] = None,
            verbose: int = 0,
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
            use_rms_prop=use_rms_prop,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
        )

        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop:
            self.optimizer_class = th.optim.RMSprop
            self.optimizer_kwargs = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        self._n_updates = 0

        if _init_setup_model:
            self._setup_model()

    def get_action(self, obs_tensor: torch.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = preprocess_obs(obs_tensor, self.observation_space)
        values = self.critic(features)
        distribution = self._get_action_dist_from_latent(features)
        actions = distribution.get_actions(deterministic=False)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.set_training_mode(True)

        # TODO easy to implement
        # Update optimizer learning rate
        # self._update_learning_rate(self.optimizer)
        entropy_loss, policy_loss, value_loss = None, None, None

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            values, log_prob, entropy = self.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            self.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.add_scalar("train/explained_variance", explained_var, self.num_timesteps)
        self.logger.add_scalar("train/entropy_loss", entropy_loss.item(), self.num_timesteps)
        self.logger.add_scalar("train/policy_loss", policy_loss.item(), self.num_timesteps)
        self.logger.add_scalar("train/value_loss", value_loss.item(), self.num_timesteps)
        if hasattr(self.policy, "log_std"):
            self.logger.add_scalar("train/std", th.exp(self.policy.log_std).mean().item(), self.num_timesteps)

    def learn(
            self,
            total_timesteps: int,
            tb_log_name: str = "A2C",
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

