# Numerical analysis and it's applications to mathematical physics : practical assignments
## Usage

## Non-linear equations
You can solve non - linear equations using this "library".
<<<<<<< HEAD
For instance, consider an equation

`sin(x+2)-x**2+2*x-1 = 0`. 

This equation has roots at 
`x = 0.0606335` and at `x = 1.12577`. 

Let us find the latter root.
=======
For instance, consider and equation `sin(x+2)-x**2+2*x-1 = 0`. This equation has roots at 
`x = 0.0606335` and at `x = 1.12577`. Let us find the latter root.
>>>>>>> 2c34e8e... Editing README.md file

```python
from numpy.testing import assert_almost_equal
from numanalysis.nonlineareq.non_linear_equations import *
from numpy import sin, cos


def lhs(x: float) -> float:
    return sin(x+2)-x**2+2*x-1


def lhs_der(x: float) -> float:
    return -2*x+cos(x+2)+2


left_border, right_border = 1, 2
initial_root_candidate = 1

equation = NonLinearEquation(
    lhs, left_border, right_border, initial_root_candidate, lhs_der
    )
result = Secant.solve(equation)
assert_almost_equal(result.calculation_result, 1.1257726213960026)
```
Other methods are used in a similar fashion.
<<<<<<< HEAD


=======
>>>>>>> 2c34e8e... Editing README.md file
## Linear equations system
## Eigenvalues problem
