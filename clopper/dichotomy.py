from typing import Tuple, Callable

from statsmodels.stats.proportion import proportion_confint

from clopper.calculate_coefficients import calculate_coefficients
from clopper.hyperopt_optimization import hyperopt_optimization
from clopper.objective_function import objective_function


def dichotomy(x: Tuple[int, int, int], alpha: float, left: float, right: float,
              coefficients: Tuple[Tuple[int, int, int, int], ...], h: Callable[[float], float],
              tolerance: float = 1e-1) -> float:
    n = sum(x)
    observation = h(x[0] / n) - h(x[1] / n)

    while right - left > tolerance:
        print(f"left = {left}, right = {right}")
        mid = (left + right) / 2
        objective = lambda p0, p1: objective_function((p0, p1, 1 - p0 - p1), mid, coefficients, observation, h)
        prob = hyperopt_optimization(objective, alpha)
        print(f"mid = {mid}, prob = {prob}")

        if abs(prob - alpha) < tolerance:
            return mid
        elif prob < 1 - alpha:
            right = mid
        else:
            left = mid

    return left


def main() -> None:
    # Example usage
    x = (25, 3, 0)
    n = sum(x)
    coefficients = calculate_coefficients(n)
    alpha = 0.001
    p0 = proportion_confint(x[0], n, alpha=alpha, method="beta")[0]
    p1 = proportion_confint(x[1], n, alpha=alpha, method="beta")[1]
    ref_L = p0 - p1
    print(f"Reference L: {ref_L}")
    observation = (x[0] - x[1]) / n
    print(f"Observation: {observation}")
    left = 0
    right = 3
    from time import time
    start_time = time()
    result = dichotomy(x, alpha, left, right, coefficients, lambda x: x, tolerance=1e-2)
    print(f"L = {result}")
    print(f"Time: {time() - start_time}")


if __name__ == '__main__':
    main()
