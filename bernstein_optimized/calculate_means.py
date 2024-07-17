from random import random
from time import time

import torch
from numpy import isclose
from torch import cuda, Tensor, tensor

from bernstein.calculate_means import calculate_means as calculate_means_ref


def calculate_means(x: torch.Tensor, debug: bool = False) -> torch.Tensor:
    t = x.size(0)

    # Create a range tensor for indexing
    indices = torch.arange(2, t + 1, device=x.device)

    # Calculate cumulative sum of x[1:]
    cumsum = torch.cumsum(x[1:], dim=0)

    # Calculate numerator for all indices at once
    numerator = 0.5 + cumsum

    # Calculate means using vectorized operations
    means = torch.zeros(t, device=x.device)
    means[1:] = numerator / indices

    if debug:
        print(f"indices: {indices}")
        print(f"cumsum: {cumsum}")
        print(f"numerator: {numerator}")
        print(f"means: {means}")

    return means


def main() -> None:
    # Example usage
    x: Tensor = tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device='cuda' if cuda.is_available() else 'cpu')
    result: Tensor = calculate_means(x, debug=True)

    print("Input tensor:", x)
    print("Input tensor device:", x.device)
    print("Calculated means tensor:", result)
    print("Result tensor device:", result.device)
    print("\nCalculated means:")
    for i, mean in enumerate(result):
        print(f"μ{i} = {mean:.4f}")

    x = tensor([0.] + [random() for _ in range(6000)], device='cuda' if cuda.is_available() else 'cpu')
    start_time = time()
    result = calculate_means(x, debug=False)
    end_time = time()

    print(f"Time taken by optimized: {end_time - start_time:.4f} seconds")

    start_time = time()
    result_ref = calculate_means_ref(x)
    end_time = time()

    for i in range(6000):
        if not isclose(result_ref[i], result[i]):
            print(f"Mismatch at index {i}!")
            print(f"result_ref[{i}] = {result_ref[i]:.4f}")
            print(f"result[{i}] = {result[i]:.4f}")

    print(f"Time taken by reference: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
