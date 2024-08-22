from typing import Tuple


def generate_catalan_sequence(n: int) -> Tuple[float, ...]:
    c = [1.0]  # Initialize with c_0 = 1

    for k in range(1, n):
        c_k = sum(c[m] * c[k - 1 - m] / ((m + 1) * (2 * m + 1)) for m in range(k))
        c.append(c_k)

    return tuple(c)


def main() -> None:
    # Test the function with the first few terms given in the image
    result = generate_catalan_sequence(7)
    expected = (1, 1, 7 / 6, 127 / 90, 4369 / 2520, 34807 / 16200)

    print("Generated sequence:", result)
    print("Expected sequence:", expected)

    # Check if the generated sequence matches the expected values
    for gen, exp in zip(result, expected):
        if abs(gen - exp) > 1e-10:  # Using a small tolerance for floating-point comparison
            print(f"Mismatch: generated {gen}, expected {exp}")
            break
    else:
        print("The generated sequence matches the expected values.")


if __name__ == "__main__":
    main()
