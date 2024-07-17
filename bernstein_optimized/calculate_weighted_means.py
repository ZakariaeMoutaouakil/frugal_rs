from random import random
from time import time

import torch
from numpy import isclose

from bernstein.calculate_weighted_means import calculate_weighted_means as calculate_weighted_means_ref


def calculate_weighted_means(x: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    t = x.size(0)

    # Calculate cumulative sums
    lambda_cumsum = torch.cumsum(lambdas[1:], dim=0)
    weighted_cumsum = torch.cumsum(lambdas[1:] * x[1:], dim=0)

    # Calculate weighted means
    weighted_means = torch.zeros(t, device=x.device)
    weighted_means[1:] = weighted_cumsum / lambda_cumsum

    return weighted_means


def main():
    # Example usage
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device='cuda' if torch.cuda.is_available() else 'cpu')
    lambdas = torch.tensor([0., 0.5, 0.4, 0.3, 0.2, 0.1], device=x.device)

    weighted_means = calculate_weighted_means(x, lambdas)

    print("Input tensor x:", x)
    print("Lambda tensor:", lambdas)
    print("Calculated weighted means:", weighted_means)

    print("\nCalculated weighted means:")
    for i, mean in enumerate(weighted_means):
        print(f"Weighted mean up to x{i} = {mean:.4f}")

    x = torch.tensor([0.] + [random() for _ in range(6000)], device='cuda' if torch.cuda.is_available() else 'cpu')
    lambdas = torch.tensor([0.25] + [random() for _ in range(6000)], device=x.device)
    start_time = time()
    weighted_means = calculate_weighted_means(x, lambdas)
    end_time = time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")

    start_time = time()
    weighted_means_ref = calculate_weighted_means_ref(x, lambdas)
    end_time = time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")

    assert isclose(weighted_means, weighted_means_ref).all()


if __name__ == "__main__":
    main()
