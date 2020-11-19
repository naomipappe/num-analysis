from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class NonLinearEquationResult:
    """
    This class represents the non - linear equation solution, holding the
    solution itself as well as calculation logs for pretty printing
    """
    calculation_result: float or None = 0
    log: list = field(default_factory=list)

    def __str__(self):
        return '\n'.join(self.log)

    def __repr__(self):
        return f'res = {self.calculation_result}'


@dataclass
class NonLinearEquation:
    """
    Data holder for non - linear equation
    """
    lhs: Callable[[float or np.ndarray], float]
    lborder: float
    rborder: float
    init_root: float or None = None
    lhs_derivative: Callable[[float or np.ndarray], float] or None = None
    lhs_derivative2: Callable[[float or np.ndarray], float] or None = None
    delta: float = 1e-6

    def __post_init__(self):
        self.__input_check()

    def __input_check(self):
        """
        Checking that boundaries are valid, root can be found on the interval,
        and that initial root is in the said boundaries
        """
        if self.lborder >= self.rborder:
            raise ValueError(
                f'Invalid boundaries, {self.lborder}>={self.rborder}'
            )
        if self.lhs(self.lborder) * self.lhs(self.rborder) > 0:
            raise ValueError(
                f'No root on [{self.lborder}, {self.rborder}]'
            )
        if self.init_root is not None:
            if not (self.lborder <= self.init_root <= self.rborder):
                raise ValueError(
                    f'Invalid init,{self.init_root}'
                    f'out of [{self.lborder}, {self.rborder}] '
                )

    def initial_approximation_fault(self) -> float:
        """
        Returns
        -------
        Initial root approximation error, as a distance to the closest border
        of the feasiable region of the problem
        """
        return abs(self.closest_border() - self.init_root)

    def closest_border(self) -> float:
        """
        Returns
        -------
        A border closest to the solution of the equation

        """
        if self.lhs(self.init_root) * self.lhs(self.rborder) < 0:
            return self.rborder
        return self.lborder


class NonLinearEquationMethod:
    """
    This class encapsulates the solution strategy for non - linear equation
    """
    @classmethod
    def solve(cls, equation: NonLinearEquation) -> NonLinearEquationResult:
        """
        Solve a non-linear equation

        Parameters
        ----------
        equation : NonLinearEquation

        Obj, with lhs derivative initialised

        Returns
        -------
        result : NonLinearEquationResult

        Obj, containing the resulting approximation
        and calculation logs
        """
        raise NotImplementedError()

    @classmethod
    def __initial_root(cls, equation: NonLinearEquation):
        raise NotImplementedError()


class Newton(NonLinearEquationMethod):
    """
    Newton's method solution strategy
    """
    @classmethod
    def solve(cls, equation: NonLinearEquation) -> NonLinearEquationResult:
        """
        Solve a non-linear equation using the Newton's method

        Parameters
        ----------
        equation : NonLinearEquation
        Obj, with lhs derivative initialised

        Returns
        -------
        result : NonLinearEquationResult

        Obj, containing the resulting approximation
        and calculation logs

        Examples
        --------
        Solve the equation sin(x+2)-x**2+2*x-1 = 0

        from numanalysis.nonlineareq.non_linear_equations import Newton, NonLinearEquation, NonLinearEquationResult

        def lhs(x: float) -> float:
            return sin(x+2)-x**2+2*x-1

        def lhs_der(x: float) -> float:
            return -2 * x + cos(x + 2) + 2

        def lhs_der_2(x: float) -> float:
            return -sin(x + 2) - 2

        lborder, rborder = 1, 2
        init_root = 1

        equation = NonLinearEquation(
            lhs, lborder, rborder, init_root, lhs_der, lhs_der_2)

        result = Newton.solve(equation)
        """
        cls.__initial_root(equation)

        def step(approx):
            return approx - \
                equation.lhs(approx) / equation.lhs_derivative(approx)

        root = equation.init_root
        i = 0
        result = NonLinearEquationResult()

        while abs(step(root) - root) > equation.delta or equation.lhs(step(root)) > equation.delta:
            result.log.append(
                f'Iteration number = {i}, current approximation = {root}, error = {equation.lhs(root)}'
            )
            i += 1
            root = step(root)

        result.calculation_result = root

        return result

    @classmethod
    def __initial_root(cls, equation: NonLinearEquation):
        if equation.init_root is None:
            if equation.lhs(equation.lborder) * equation.lhs_derivative2(equation.lborder) > 0:
                equation.init_root = equation.lborder
            elif equation.lhs(equation.rborder) * equation.lhs_derivative2(equation.rborder) > 0:
                equation.init_root = equation.rborder
            else:
                equation.init_root = (
                    equation.lborder + equation.rborder) / 2


class Secant(NonLinearEquationMethod):
    """
    This class encapsulates the solution strategy for non - linear equation using Secant method
    """
    @classmethod
    def solve(cls, equation: NonLinearEquation) -> NonLinearEquationResult:
        """
        Solve a non-linear equation using the secant method

        Parameters
        ----------
        equation : NonLinearEquation
        NonLinearEquation object, with equation left-hand side derivative initialised

        Returns
        -------
        result : NonLinearEquationResult

        NonLinearEquationResult object, containing the resulting root approximation
        as well as calculation logs

        Examples
        --------
        Solve the equation sin(x+2)-x**2+2*x-1 = 0

        from numanalysis.nonlineareq.non_linear_equations import Secant, NonLinearEquation, NonLinearEquationResult

        def lhs(x: float) -> float:
            return sin(x+2)-x**2+2*x-1

        def lhs_derr(x: float) -> float:
            return -2*x+cos(x+2)+2

        left_border, right_border = 1, 2
        initial_root_candidate = 1

        equation = NonLinearEquation(lhs, left_border, right_border, initial_root_candidate, lhs_derr)

        result = Secant.solve(equation)
        """

        x_values = np.linspace(equation.lborder, equation.rborder, min(
            int(1. / equation.delta), 10 ** 5))
        min_val, max_val = round(np.min(np.abs(equation.lhs_derivative(x_values))), 6), round(
            np.max(np.abs(equation.lhs_derivative(x_values))), 6)
        tau = 2. / (min_val + max_val)

        def step(root_candidate):
            return root_candidate - np.sign(equation.lhs_derivative(root_candidate)) * tau * equation.lhs(
                root_candidate)

        i = 0
        cls.__initial_root(equation)
        root = equation.init_root
        result = NonLinearEquationResult()
        while abs(step(root) - root) > equation.delta or equation.lhs(step(root)) > equation.delta:
            result.log.append(
                f'Iteration number = {i}, current approximation = {root}, error = {equation.lhs(root)}'
            )
            i += 1
            root = step(root)
        result.calculation_result = root

        return result

    @classmethod
    def __initial_root(cls, equation: NonLinearEquation):
        if equation.init_root is None:
            equation.init_root = (
                equation.lborder + equation.rborder) / 2
            equation.init_root = (
                equation.closest_border() + equation.init_root) / 2


class Relaxation(NonLinearEquationMethod):
    """
    This class encapsulates the solution strategy for non - linear equation using Relaxation method
    """
    @classmethod
    def solve(cls, equation: NonLinearEquation) -> NonLinearEquationResult:
        """
        Solve a non-linear equation using the Relaxation method

        Parameters
        ----------
        equation : NonLinearEquation
        Obj, with lhs derivative initialised

        Returns
        -------
        result : NonLinearEquationResult

        Obj, containing the resulting approximation
        and calculation logs

        Examples
        --------
        Solve the equation sin(x+2)-x**2+2*x-1 = 0

        from numanalysis.nonlineareq.non_linear_equations import Relaxation, NonLinearEquation, NonLinearEquationResult

        def lhs(x: float) -> float:
            return sin(x+2)-x**2+2*x-1

        left_border, right_border = 1, 2
        initial_root_candidate = 1

        equation = NonLinearEquation(lhs, left_border, right_border, initial_root_candidate)

        result = Relaxation.solve(equation)
        """
        root = (equation.closest_border() +
                equation.init_root) / 2
        previous_root = equation.init_root

        def step(cur, prev):
            cur, prev = cur - \
                equation.lhs(cur) * (cur - prev) / \
                (equation.lhs(cur) - equation.lhs(prev)), cur
            return cur, prev

        i = 0
        result = NonLinearEquationResult()
        while abs(root - previous_root) > equation.delta or abs(equation.lhs(root)) > equation.delta:
            result.log.append(
                f'Iteration number = {i}, current approximation = {root}, error = {equation.lhs(root)}'
            )
            i += 1
            root, previous_root = step(root, previous_root)
        result.calculation_result = root
        return result

    @classmethod
    def __initial_root(cls, equation: NonLinearEquation):
        if equation.init_root is None:
            middle_point = (equation.lborder + equation.rborder) / 2
            equation.init_root = (
                equation.closest_border() + middle_point) / 2
