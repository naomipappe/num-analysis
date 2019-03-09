import math
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

    def generate_system(self, k: int):
        for i in range(k+1):
            yield self.get_function(k)

    def get_function(self, k: int):
        return lambda x: np.e**(k*x)


class TrigonometricSystem(FunctionSystem):
    def __init__(self):
        FunctionSystem.__init__(self)
        self.descr = "{1, cos(kx), sin(kx)} system of functions"

    def generate_system(self, k: int):
        for i in range(k+1):
            yield self.get_function(k)

    def get_function(self, k: int):
        if k != 0:
            return lambda x: (np.sin if k % 2 == 0 else np.cos)(k * x)/np.sqrt(np.pi)
        return lambda x: 1/np.sqrt(np.pi*2)


if __name__ == "__main__":
    test = FunctionSystem()
    print(test.description)
    test = TrigonometricSystem()
    a =test.get_function(5)
