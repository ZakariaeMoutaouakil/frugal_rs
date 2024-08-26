from math import log
from time import time
from typing import Callable, Tuple

from scipy.stats import multinomial

from clopper.calculate_coefficients import calculate_coefficients
from clopper.margin_proba import margin_proba


def multinomial_margin_proba(observation: float, p: Tuple[float, float, float], n: int,
                             h: Callable[[float], float]) -> float:
    """
    Calculates the exact probability that h(x1 / n) - h(x2 / n) <= theta using the multinomial distribution.

    Args:
        observation (float): The threshold.
        p (Tuple[float, float, float]): The probability vector.
        n (int): Number of trials.
        h (Callable[[float], float]): The inverse CDF function.

    Returns:
        float: Exact probability.
    """
    prob = 0.0
    for x1 in range(n + 1):
        for x2 in range(n - x1 + 1):
            x3 = n - x1 - x2
            if h(x1 / n) - h(x2 / n) <= observation:
                prob += multinomial.pmf((x1, x2, x3), n, p)
    return prob


def main():
    # Parameters
    observation = 0.2
    p = (0.3, 0.3, 0.4)
    n = 10

    # Define the inverse CDF function (example: logarithm function)
    def h(x: float) -> float:
        return -log(1 - x) if x < 1 else float('inf')

    # Calculate probability using multinomial distribution
    start_time = time()
    result = multinomial_margin_proba(observation, p, n, h)
    print(f"Multinomial margin probability calculation time: {time() - start_time:.6f} seconds")
    print(f"Number of trials (n): {n}")
    print(f"Observation threshold: {observation}")
    print(f"Probability vector (p1, p2, p3): {p}")
    print(f"Exact margin probability (Multinomial method): {result:.6f}\n")

    # Calculate probability using original method
    start_time = time()
    coeffs = calculate_coefficients(n)
    original_result = margin_proba(observation, p, coeffs, h)
    print(f"Original margin probability calculation time: {time() - start_time:.6f} seconds")
    print(f"Exact margin probability (Original method): {original_result:.6f}")
    print(f"Absolute difference: {abs(result - original_result):.6f}")


if __name__ == "__main__":
    main()
