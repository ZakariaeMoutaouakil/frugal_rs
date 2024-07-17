from math import log, sqrt
from random import random
from time import time

from numpy import isclose
from torch import Tensor, log, sqrt, arange, log1p, tensor, zeros, cuda

from bernstein.calculate_lambdas import calculate_lambdas as calculate_lambdas_ref


def calculate_lambdas(x: Tensor, sigma: Tensor, alpha: float, c: float = 0.5, debug: bool = False) -> Tensor:
    t = x.size(0)
    lambdas = zeros(t, device=x.device)

    log_term = log(tensor(2 / alpha))
    i_range = arange(1, t, device=x.device)

    numerator = 2 * log_term
    denominator = sigma[:-1] * i_range * log1p(i_range)

    lambdas[1:] = sqrt(numerator / denominator).clamp(max=c)

    if debug:
        print(f"numerator = {numerator}")
        print(f"denominator = {denominator}")
        print(f"lambdas: {lambdas}")

    return lambdas


def main():
    # Example usage
    x = tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device='cuda' if cuda.is_available() else 'cpu')
    sigma = tensor([0.2500, 0.3536, 0.4082, 0.4564, 0.5000, 0.5408], device=x.device)
    c = 0.75
    alpha = 0.05

    lambdas = calculate_lambdas(x, sigma, alpha, c, debug=True)

    print("Input tensor x:", x)
    print("Sigmas tensor:", sigma)
    print("c value:", c)
    print("alpha value:", alpha)
    print("Calculated lambdas:", lambdas)

    print("\nCalculated lambdas:")
    for i, lambda_val in enumerate(lambdas):
        print(f"λ{i} = {lambda_val:.4f}")

    x = tensor([0.] + [random() for _ in range(6000)], device='cuda' if cuda.is_available() else 'cpu')
    sigma = tensor([0.25] + [random() for _ in range(6000)], device=x.device)
    start_time = time()
    lambdas = calculate_lambdas(x, sigma, alpha, c, debug=False)
    end_time = time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")

    start_time = time()
    lambdas_ref = calculate_lambdas_ref(x, sigma, alpha, c)
    end_time = time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")

    for i in range(6000):
        if not isclose(lambdas[i], lambdas_ref[i]):
            print(f"Mismatch at index {i}!")
            print(f"lambdas_ref[{i}] = {lambdas_ref[i]:.4f}")
            print(f"lambdas[{i}] = {lambdas[i]:.4f}")


if __name__ == "__main__":
    main()
