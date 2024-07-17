from math import log

import torch

from bernstein.psi_e import psi_e


def calculate_terms(x: torch.Tensor, lambdas: torch.Tensor, v: torch.Tensor, alpha) -> torch.Tensor:
    t = x.size(0)
    terms = torch.zeros(t, device=x.device)

    log_term = log(2 / alpha)
    cumulative_v_psi = 0.
    cumulative_lambda = 0.

    for i in range(1, t):
        cumulative_v_psi += v[i] * psi_e(lambdas[i].item())
        cumulative_lambda += lambdas[i]
        terms[i] = (log_term + cumulative_v_psi) / cumulative_lambda

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


if __name__ == "__main__":
    main()
