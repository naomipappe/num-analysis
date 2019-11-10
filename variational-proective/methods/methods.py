class VariationalProective:
    def __init__(self):
        super().__init__()

    @classmethod
    def solve(cls):
        raise NotImplementedError


class Ritz(VariationalProective):
    def __init__(self):
        super().__init__()

    @classmethod
    def solve(cls):
        raise NotImplementedError("Ritz")


class Collocation(VariationalProective):
    def __init__(self):
        super().__init__()

    @classmethod
    def solve(cls):
        raise NotImplementedError("Collocation")


class LeastSquares(VariationalProective):
    def __init__(self):
        super().__init__()

    @classmethod
    def solve(cls):
        raise NotImplementedError("Least Squares")


class BubnovGalerkin(VariationalProective):
    def __init__(self):
        super().__init__()

    @classmethod
    def solve(cls):
        raise NotImplementedError("BubnovGalerkin")

