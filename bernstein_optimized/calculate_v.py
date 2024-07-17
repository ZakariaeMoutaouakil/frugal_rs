from random import random
from time import time

import torch
from numpy import isclose
from torch import cuda, tensor

from bernstein.calculate_v import calculate_v as calculate_v_ref
from bernstein_optimized.calculate_means import calculate_means


def calculate_v(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    t = x.size(0)

    # Create v tensor with zeros
    v = torch.zeros(t, device=x.device)

    # Calculate v[1:] in one vectorized operation
    v[1:] = 4 * (x[1:] - mu[:-1]) ** 2

    return v


def main():
    # Example usage
    x = tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device='cuda' if cuda.is_available() else 'cpu')
    mu = tensor([0., 0.75, 1.1667, 1.6250, 2.1000, 2.5833], device=x.device)

    v = calculate_v(x, mu)

    print("Input tensor x:", x)
    print("Means tensor mu:", mu)
    print("Calculated v:", v)

    print("\nCalculated v values:")
    for i, v_val in enumerate(v):
        print(f"v{i} = {v_val:.4f}")

    x = tensor([0.] + [random() for _ in range(6000)], device='cuda' if cuda.is_available() else 'cpu')
    mu = calculate_means(x)
    start_time = time()
    v = calculate_v(x, mu)
    end_time = time()

    print(f"Time taken by optimized: {end_time - start_time:.4f} seconds")

    start_time = time()
    v_ref = calculate_v_ref(x, mu)
    end_time = time()

    print(f"Time taken by reference: {end_time - start_time:.4f} seconds")

    for i in range(6000):
        if not isclose(v_ref[i], v[i]):
            print(f"Mismatch at index {i}!")
            print(f"v_ref[{i}] = {v_ref[i]:.4f}")
            print(f"v[{i}] = {v[i]:.4f}")


if __name__ == "__main__":
    main()
