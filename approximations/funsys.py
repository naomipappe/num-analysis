
from typing import Callable

import numpy as np


class FunctionSystem:
    def __init__(self):
        self.descr = ' '
        pass

    def generate_system(self, k: int):
        pass

    def get_function(self, k: int):
        pass

    @property
    def description(self):
        return self.descr


class ExponentialSystem(FunctionSystem):
    def __init__(self):
        FunctionSystem.__init__(self)
        self.descr = "{exp(kx)} system of functions"

    def __str__(self):
        return "Exponential system"

    def generate_system(self, k: int):
        for i in range(k+1):
            yield self.get_function(i)

    def get_function(self, k: int):
        return lambda x: np.exp((((k if (k & 1) else -k) + 1) >> 1) * x)


class TrigonometricSystem(FunctionSystem):
    def __init__(self):
        FunctionSystem.__init__(self)
        self.descr = "{1, cos(kx), sin(kx)} system of functions"

    def __str__(self):
        return "Trigonometric system"

    def generate_system(self, k: int):
        for i in range(k+1):
            yield self.get_function(i)

    def get_function(self, k: int):
        return lambda x: (np.sin if (k & 1) else np.cos)(((k + 1) >> 1) * x)
        # if k != 0:
        #     return lambda x: (np.sin if k % 2 == 0 else np.cos)(k * x)/np.sqrt(np.pi)
        # return lambda x: 1/np.sqrt(np.pi*2)

def function_rescale(f: Callable[[float], float], old_a: float, old_b: float,
                     new_a: float, new_b: float) -> Callable[[float], float]:
    return lambda x: f((old_b - old_a) / (new_b - new_a) * (x - new_a) + old_a)
