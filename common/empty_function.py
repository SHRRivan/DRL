import numpy as np


class EmptyWriter(object):
    def __init__(self):
        pass

    def add_scalar(self, title, value, period):
        pass


class EmptyNoise(object):
    def __init__(self):
        pass

    def __call__(self) -> np.ndarray:
        return np.array(0)
