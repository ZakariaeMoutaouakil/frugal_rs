from torch import Tensor, tensor, cat

from bernstein.calculate_lambdas import calculate_lambdas
from bernstein.calculate_means import calculate_means
from bernstein.calculate_sigmas import calculate_sigmas
from bernstein.calculate_terms import calculate_terms
from bernstein.calculate_v import calculate_v
from bernstein.calculate_weighted_means import calculate_weighted_means


def calculate_shift(x: Tensor, alpha_0: float, debug: bool = False) -> float:
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

    lower_bound = weighted_means[-1] - terms[-1]
    if debug:
        print(f"weighted_means[-1] = {weighted_means[-1]}")
        print(f"terms[-1] = {terms[-1]}")

    return lower_bound.item()
