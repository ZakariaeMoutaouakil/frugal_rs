from typing import Tuple, Optional

import torch
from torch.distributions import Dirichlet


def sample_dirichlet(alpha: Tuple[float, float, float], num_samples: int, seed: Optional[int] = None) -> torch.Tensor:
    """
    Generate samples from a 3-dimensional Dirichlet distribution.

    Args:
        alpha (Tuple[float, float, float]): Concentration parameters of the Dirichlet distribution.
        num_samples (int): Number of samples to generate.
        seed (Optional[int]): Seed for reproducible sampling. If None, sampling is not seeded.

    Returns:
        torch.Tensor: A tensor of shape (num_samples, 3) containing the Dirichlet samples.
    """
    if seed is not None:
        torch.manual_seed(seed)

    dirichlet = Dirichlet(torch.tensor(alpha))
    samples = dirichlet.sample((num_samples,))
    return samples


def main():
    # Example usage
    alpha = (10.0, 2.0, 1.0)
    num_samples = 100

    samples = sample_dirichlet(alpha, num_samples, seed=42)

    print(f"Generated {num_samples} samples from Dirichlet({alpha}):")
    print(samples)

    # Verify that each row sums to 1 (within floating-point precision)
    row_sums = torch.sum(samples, dim=1)
    print("\nRow sums (should be close to 1):")
    print(row_sums)


if __name__ == "__main__":
    main()
