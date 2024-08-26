from typing import Tuple

from scipy.special import comb


def calculate_coefficients(n: int) -> Tuple[Tuple[int, int, int, int], ...]:
    """
    Calculates the coefficients for the second margin probability calculation.

    Args:
        n (int): The number of trials.

    Returns:
        Tuple[Tuple[int, int, int, int], ...]: A tuple of tuples containing (x1, x2, coefficient, n-x1-x2).
    """
    coefficients = []
    for x2 in range(n + 1):
        for x1 in range(n + 1):
            if x1 + x2 <= n:
                coefficient = comb(n, x1, exact=True) * comb(n - x1, x2, exact=True)
                coefficients.append((x1, x2, coefficient, n - x1 - x2))
    return tuple(coefficients)


def main():
    # Example usage
    n = 3  # Number of trials

    print(f"Calculating coefficients for n = {n}")
    result = calculate_coefficients(n)

    print("\nResults:")
    print("(x1, x2, coefficient, n-x1-x2)")
    for coeff in result:
        print(coeff)

    print(f"\nTotal number of coefficient combinations: {len(result)}")

    # Additional analysis
    total_sum = sum(coeff[2] for coeff in result)
    print(f"\nSum of all coefficients: {total_sum}")
    print(f"This should equal 3^n (3^{n} = {3 ** n})")

    # Example of how to use a specific coefficient
    example_x1, example_x2 = 1, 1
    for x1, x2, coeff, remainder in result:
        if x1 == example_x1 and x2 == example_x2:
            print(f"\nCoefficient for x1={x1}, x2={x2}: {coeff}")
            print(f"This represents the number of ways to have {x1} successes in the first category,")
            print(f"{x2} successes in the second category, and {remainder} failures in {n} trials.")
            break


if __name__ == "__main__":
    main()
