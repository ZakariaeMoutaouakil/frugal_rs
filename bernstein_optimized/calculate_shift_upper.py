from random import random
from time import time

import torch
from numpy import isclose
from torch import Tensor, tensor, cat

from bernstein.calculate_shift import calculate_shift as calculate_shift_ref
from bernstein_optimized.calculate_lambdas import calculate_lambdas
from bernstein_optimized.calculate_means import calculate_means
from bernstein_optimized.calculate_sigmas import calculate_sigmas
from bernstein_optimized.calculate_terms import calculate_terms
from bernstein_optimized.calculate_v import calculate_v
from bernstein_optimized.calculate_weighted_means import calculate_weighted_means


def calculate_shift_upper(x: Tensor, alpha_0: float, debug: bool = False) -> float:
    # Tensor with a single zero
    zero_tensor = tensor([0.], device=x.device)

    # Prepend zero to the original tensor
    y = cat((zero_tensor, x))

    alpha = alpha_0 / 2

    means = calculate_means(x=y)
    if debug:
        print(f"means = {means}")

    sigmas = calculate_sigmas(x=y, mu=means)
    if debug:
        print(f"sigmas = {sigmas}")

    lambdas = calculate_lambdas(x=y, sigma=sigmas, alpha=alpha)
    if debug:
        print(f"lambdas = {lambdas}")

    weighted_means = calculate_weighted_means(x=y, lambdas=lambdas)
    if debug:
        print(f"weighted_means = {weighted_means}")

    v = calculate_v(x=y, mu=weighted_means)
    if debug:
        print(f"v = {v}")

    terms = calculate_terms(x=y, lambdas=lambdas, v=v, alpha=alpha)
    if debug:
        print(f"terms = {terms}")

    upper_bound = weighted_means[-1] + terms[-1]
    if debug:
        print(f"weighted_means[-1] = {weighted_means[-1]}")
        print(f"terms[-1] = {terms[-1]}")

    return upper_bound.item()


def main():
    x = tensor([0.] + [random() for _ in range(6000)], device='cuda' if torch.cuda.is_available() else 'cpu')
    alpha_0 = 0.05
    start_time = time()
    shift = calculate_shift_upper(x, alpha_0)
    end_time = time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")

    start_time = time()
    shift_ref = calculate_shift_ref(x, alpha_0)
    end_time = time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")

    assert isclose(shift, shift_ref)


if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt:
            break
