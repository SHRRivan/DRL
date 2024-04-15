import copy
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import numpy as np
import matplotlib.pyplot as plt


class ActionNoise(ABC):
    """
    The action noise base class
    """

    def __init__(self):
        super().__init__()

    def reset(self) -> None:
        """
        call end of episode reset for the noise
        """
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()


class GaussianActionNoise(ActionNoise):
    """
    A Gaussian action noise

    :param mean: the mean value of the noise
    :param sigma: the scale of the noise (std here)
    """

    def __init__(self, mean: np.ndarray, sigma: np.ndarray = np.array(0.2)):
        self._mu = mean
        self._sigma = sigma
        super().__init__()

    def __call__(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma)

    def __repr__(self) -> str:
        return f"NormalActionNoise(mu={self._mu}, sigma={self._sigma})"


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.

    Based on https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param mean: the mean of the noise
    :param sigma: the scale of the noise
    :param theta: the rate of mean reversion
    :param dt: the timestep for the noise
    :param initial_noise: the initial value for the noise output, (if None: 0)
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: Optional[np.ndarray] = None,
    ):
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros_like(self._mu)
        self.reset()
        super().__init__()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._mu)

    def __repr__(self) -> str:
        return f"OrnsteinUhlenbeckActionNoise(mu={self._mu}, sigma={self._sigma})"


if __name__ == '__main__':
    ou_noise = OrnsteinUhlenbeckActionNoise(np.zeros(2), np.ones(2))
    ou_noise._sigma = 1
    print(ou_noise._sigma)

