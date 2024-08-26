from typing import Tuple, Callable

from numba import jit
from numpy import inf, log

from clopper.calculate_coefficients import calculate_coefficients
from clopper.margin_proba import margin_proba


@jit(nopython=True)
def g(x):
    return -log(1 - x) if x < 1 else inf


@jit(nopython=True)
def jit_margin_proba(observation: float, p: Tuple[float, float, float],
                     coefficients: Tuple[Tuple[int, int, int, int], ...], h: Callable[[float], float]) -> float:
    """
    JIT-compiled function to calculate the probability that h(x1 / n) - h(x2 / n) <= theta.

    Args:
        observation (float): The threshold.
        p (np.ndarray): The probability vector.
        coefficients (np.ndarray): Pre-calculated coefficients.
        h (Callable[[float], float]): The function to calculate the probability.

    Returns:
        float: Calculated probability.
    """
    prob = 0.
    n = coefficients[0][0] + coefficients[0][1] + coefficients[0][3]  # Extract n from the first coefficient tuple
    for x1, x2, coefficient, n_minus_x1_x2 in coefficients:
        if h(x1 / n) - h(x2 / n) <= observation:
            term = coefficient * (p[0] ** x1) * (p[1] ** x2) * (p[2] ** n_minus_x1_x2)
            prob += term
    return prob


def main():
    # Example usage
    n = 10
    observation = 0.2
    p = (0.3, 0.3, 0.4)

    coefficients = calculate_coefficients(n)

    # First run (compilation)
    result = jit_margin_proba(observation, p, coefficients, g)
    print(f"First run result (includes compilation time): {result:.6f}")

    # Second run (JIT-compiled)
    result = jit_margin_proba(observation, p, coefficients, g)
    print(f"Second run result (JIT-compiled): {result:.6f}")

    # Compare with non-JIT version
    original_result = margin_proba(observation, p, coefficients, g)
    print(f"Original function result: {original_result:.6f}")

    # Timing comparison
    import time

    start = time.time()
    for _ in range(100000):
        jit_margin_proba(observation, p, coefficients, g)
    jit_time = time.time() - start
    print(f"JIT version time for 1000 runs: {jit_time:.6f} seconds")

    start = time.time()
    for _ in range(100000):
        margin_proba(observation, tuple(p), coefficients, g)
    original_time = time.time() - start
    print(f"Original version time for 1000 runs: {original_time:.6f} seconds")

    print(f"Speedup: {original_time / jit_time:.2f}x")


if __name__ == "__main__":
    main()
