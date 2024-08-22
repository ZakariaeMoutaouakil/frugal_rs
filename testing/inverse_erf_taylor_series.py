from math import sqrt, pi
from typing import Callable

from scipy import special

from testing.generate_catalan_sequence import generate_catalan_sequence


def inverse_erf_taylor_series(order: int) -> Callable[[float], float]:
    c_k = generate_catalan_sequence(order + 1)
    sqrt_pi_over_2 = sqrt(pi) / 2

    def approximation(z: float) -> float:
        return sum(
            (c_k[k] / (2 * k + 1)) * ((sqrt_pi_over_2 * z) ** (2 * k + 1))
            for k in range(order + 1)
        )

    return approximation


def main():
    # Example usage
    order = 5  # Approximation order
    inverse_erf_approx = inverse_erf_taylor_series(order)

    # Test the approximation for some values
    test_values = [0.1, 0.5, 0.9]
    for z in test_values:
        approx_result = inverse_erf_approx(z)
        print(f"erf^(-1)({z}) ≈ {approx_result}")

    print("\nComparison with scipy.special.erfinv:")
    for z in test_values:
        approx_result = inverse_erf_approx(z)
        exact_result = special.erfinv(z)
        print(f"z = {z}:")
        print(f"  Approximation: {approx_result}")
        print(f"  Exact value:   {exact_result}")
        print(f"  Absolute error: {abs(approx_result - exact_result)}")
        print(f"  Relative error: {abs((approx_result - exact_result) / exact_result) * 100:.6f}%")


if __name__ == "__main__":
    main()
