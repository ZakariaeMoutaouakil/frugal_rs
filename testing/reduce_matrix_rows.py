from typing import Callable

from torch import Tensor, tensor

from testing.sample_dirichlet import sample_dirichlet


def reduce_matrix_rows(matrix: Tensor, func: Callable[[float, float, float], float]) -> Tensor:
    """
    Apply a function to each row of a 3-column matrix and return a 1D tensor of results.

    Args:
        matrix (torch.Tensor): Input matrix of shape (n, 3).
        func (Callable[[float, float, float], float]): Function to apply to each row.

    Returns:
        torch.Tensor: A 1D tensor of shape (n,) containing the results.
    """
    assert matrix.shape[1] == 3, "Input matrix must have 3 columns"

    # Use a list comprehension to apply the function to each row
    results = tensor([func(*row) for row in matrix], device=matrix.device)

    return results


def main():
    # Example usage
    alpha = (1.0, 2.0, 3.0)
    num_samples = 5

    samples = sample_dirichlet(alpha, num_samples)

    print("Generated samples from Dirichlet distribution:")
    print(samples)

    # Define an example reduction function (e.g., weighted sum)
    def weighted_sum(a: float, b: float, c: float) -> float:
        return 0.3 * a + 0.3 * b + 0.4 * c

    # Apply the reduction function to each row
    reduced = reduce_matrix_rows(samples, weighted_sum)

    print("\nReduced values:")
    print(reduced)


if __name__ == "__main__":
    main()
