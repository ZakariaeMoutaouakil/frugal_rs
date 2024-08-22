from math import sqrt
from typing import Callable

from scipy.special import erfinv

from testing.inverse_erf_taylor_series import inverse_erf_taylor_series


def gaussian_quantile_approximation(order: int) -> Callable[[float], float]:
    inv_erf_approx = inverse_erf_taylor_series(order)

    def approximation(p: float) -> float:
        # Ensure p is within (0, 1)
        p = max(1e-15, min(p, 1 - 1e-15))
        return sqrt(2) * inv_erf_approx(2 * p - 1)

    return approximation


def main():
    # Example usage
    order = 5  # Approximation order
    gaussian_quantile_approx = gaussian_quantile_approximation(order)

    # Test the approximation for some probabilities
    test_values = [-1, 0.1, 0.25, 0.5, 0.75, 0.9]

    print(f"Gaussian Quantile Function Approximation (Order {order}):")
    print("p\tApproximation\tExact Value\tAbsolute Error")
    print("---------------------------------------------------------")

    for p in test_values:
        approx_result = gaussian_quantile_approx(p)

        # For comparison, we'll use the inverse of the error function
        # from the math module to calculate the "exact" value
        exact_result = sqrt(2) * erfinv(2 * p - 1)

        abs_error = abs(approx_result - exact_result)

        print(f"{p:.2f}\t{approx_result:.6f}\t\t{exact_result:.6f}\t\t{abs_error:.6f}")


if __name__ == "__main__":
    main()
