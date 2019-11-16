from matplotlib import pyplot as plt
from typing import Callable


def plotter(
    x: list,
    precise_solution: Callable[[float], float],
    approximation: Callable[[float], float],
    save: bool = False,
    name: str = "result",
):
    plt.plot(x, precise_solution(x), "r")
    plt.plot(x, approximation(x), "b")
    plt.fill_between(
        x, precise_solution(x), approximation(x), color="yellow", alpha="0.5"
    )
    if save:
        plt.savefig(f"results\\figures\\{name}.png")
    else:
        plt.show()
