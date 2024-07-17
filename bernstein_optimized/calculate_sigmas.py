from random import random
from time import time

import torch
from numpy import isclose
from torch import cuda, tensor

from bernstein.calculate_sigmas import calculate_sigmas as calculate_sigmas_ref
from bernstein_optimized.calculate_means import calculate_means


def calculate_sigmas(x: torch.Tensor, mu: torch.Tensor, debug: bool = False) -> torch.Tensor:
    t = x.size(0)

    # Create indices tensor
    indices = torch.arange(2, t + 1, device=x.device)

    # Calculate squared differences
    squared_diff = (x[1:] - mu[1:]) ** 2

    # Calculate cumulative sum of squared differences
    cumsum_squared_diff = torch.cumsum(squared_diff, dim=0)

    # Calculate numerator for all indices at once
    numerator = 0.25 + cumsum_squared_diff

    # Calculate sigmas using vectorized operations
    sigmas = torch.zeros(t, device=x.device)
    sigmas[0] = 0.25
    sigmas[1:] = torch.sqrt(numerator / indices)

    if debug:
        print(f"indices: {indices}")
        print(f"squared_diff: {squared_diff}")
        print(f"cumsum_squared_diff: {cumsum_squared_diff}")
        print(f"numerator: {numerator}")
        print(f"sigmas: {sigmas}")

    return sigmas


def main():
    # Example usage
    x = tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device='cuda' if cuda.is_available() else 'cpu')
    mu = tensor([0., 0.75, 1.1667, 1.6250, 2.1000, 2.5833], device=x.device)

    sigmas = calculate_sigmas(x, mu, debug=True)

    print("Input tensor x:", x)
    print("Means tensor mu:", mu)
    print("Calculated sigmas:", sigmas)

    print("\nCalculated sigmas:")
    for i, sigma in enumerate(sigmas):
        print(f"σ{i} = {sigma:.4f}")

    x = tensor([0.] + [random() for _ in range(6000)], device='cuda' if cuda.is_available() else 'cpu')
    mu = calculate_means(x)
    start_time = time()
    sigmas = calculate_sigmas(x, mu, debug=False)
    end_time = time()

    print(f"Time taken by optimized: {end_time - start_time:.4f} seconds")

    start_time = time()
    sigmas_ref = calculate_sigmas_ref(x, mu, debug=False)
    end_time = time()

    print(f"Time taken by reference: {end_time - start_time:.4f} seconds")

    for i in range(6000):
        if not isclose(sigmas_ref[i], sigmas[i]):
            print(f"Mismatch at index {i}!")
            print(f"sigmas_ref[{i}] = {sigmas_ref[i]:.4f}")
            print(f"sigmas[{i}] = {sigmas[i]:.4f}")


if __name__ == "__main__":
    main()
