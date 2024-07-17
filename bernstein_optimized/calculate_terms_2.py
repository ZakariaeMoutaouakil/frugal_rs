from random import random
from time import time

import torch
from numpy import isclose

from bernstein.calculate_terms import calculate_terms as calculate_terms_ref
from bernstein.psi_e import psi_e


def calculate_terms(x: torch.Tensor, lambdas: torch.Tensor, v: torch.Tensor, alpha) -> torch.Tensor:
    t = x.size(0)
    terms = torch.zeros(t, device=x.device)

    log_term = torch.log(torch.tensor(2 / alpha, device=x.device))

    # Precompute psi_e values
    psi_values = torch.tensor([psi_e(l.item()) for l in lambdas[1:]], device=x.device)

    # Compute cumulative sums
    cumulative_v_psi = torch.cumsum(v[1:] * psi_values, dim=0)
    cumulative_lambda = torch.cumsum(lambdas[1:], dim=0)

    # Compute terms
    terms[1:] = (log_term + cumulative_v_psi) / cumulative_lambda

    return terms


def main():
    # Example usage
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device='cuda' if torch.cuda.is_available() else 'cpu')
    lambdas = torch.tensor([0., 0.1, 0.2, 0.3, 0.4, 0.5], device=x.device)
    v = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], device=x.device)
    alpha = 0.05

    terms = calculate_terms(x, lambdas, v, alpha)

    print("Input tensor x:", x)
    print("Lambda tensor:", lambdas)
    print("v tensor:", v)
    print("alpha:", alpha)
    print("Calculated terms:", terms)

    print("\nCalculated terms:")
    for i, term in enumerate(terms):
        print(f"Term up to index {i} = {term:.4f}")

    x = torch.tensor([0.] + [random() for _ in range(6000)], device='cuda' if torch.cuda.is_available() else 'cpu')
    lambdas = torch.tensor([0.25] + [random() for _ in range(6000)], device=x.device)
    v = torch.tensor([0.0] + [random() for _ in range(6000)], device=x.device)
    start_time = time()
    terms = calculate_terms(x, lambdas, v, alpha)
    end_time = time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")

    start_time = time()
    terms_ref = calculate_terms_ref(x, lambdas, v, alpha)
    end_time = time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")

    assert isclose(terms, terms_ref).all()


if __name__ == "__main__":
    main()
