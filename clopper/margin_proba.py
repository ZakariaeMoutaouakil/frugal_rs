from math import log
from typing import Tuple, Callable

from clopper.calculate_coefficients import calculate_coefficients


def margin_proba(observation: float, p: Tuple[float, float, float],
                 coefficients: Tuple[Tuple[int, int, int, int], ...], h: Callable[[float], float]) -> float:
    """
    Calculates the probability that h(x1 / n) - h(x2 / n) <= theta using the exponential
    distribution.

    Args:
        observation (float): The threshold.
        p (Tuple): The probability vector.
        coefficients (Tuple[Tuple[int, int, int, int], ...]): Pre-calculated coefficients.
        h (Callable): The inverse CDF function.
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
    n = 5  # Number of trials
    observation = 0.2  # Threshold
    p = (0.3, 0.3, 0.4)  # Probability vector

    # Define the inverse CDF function (example: logarithm function)
    def h(x: float) -> float:
        return -log(1 - x) if x < 1 else float('inf')

    # Calculate coefficients
    coefficients = calculate_coefficients(n)

    # Calculate margin probability
    result = margin_proba(observation, p, coefficients, h)

    print(f"Number of trials (n): {n}")
    print(f"Observation threshold: {observation}")
    print(f"Probability vector (p1, p2, p3): {p}")
    print(f"\nCalculated margin probability: {result:.6f}")

    # Additional analysis
    print("\nBreakdown of some calculations:")
    for x1, x2, coeff, remainder in coefficients[:5]:  # Show first 5 coefficients
        h_diff = h(x1 / n) - h(x2 / n)
        condition_met = h_diff <= observation
        print(f"x1={x1}, x2={x2}: h({x1}/{n}) - h({x2}/{n}) = {h_diff:.4f} <= {observation}: {condition_met}")

    print("\nNote: The inverse CDF function used here is h(x) = -log(1-x), which is just an example.")
    print("In practice, you would use the appropriate inverse CDF for your specific distribution.")


if __name__ == "__main__":
    main()
