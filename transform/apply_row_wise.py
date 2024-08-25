from typing import Callable

from scipy.stats import norm
from torch import stack, Tensor, float32, tensor, exp, from_numpy


def apply_row_wise(x: Tensor, func: Callable[[Tensor], Tensor]) -> Tensor:
    """
    Apply a function to each row of a 2D torch tensor.

    Args:
    tensor (torch.Tensor): Input 2D tensor
    func (callable): Function to apply to each row

    Returns:
    torch.Tensor: New 2D tensor with func applied to each row
    """
    if x.dim() != 2:
        raise ValueError("Input tensor must be 2-dimensional")

    return stack([func(row) for row in x])


def main():
    # Example usage

    # Create a sample 2D tensor
    input_tensor = tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float32)

    print("Input tensor:")
    print(input_tensor)

    # Example 1: Apply a simple function (e.g., multiply each element by 2)
    def double_elements(row):
        return row * 2

    result1 = apply_row_wise(input_tensor, double_elements)
    print("\nResult after doubling each element:")
    print(result1)

    # Example 2: Apply a more complex function (e.g., softmax)
    def custom_softmax(row):
        exp_row = exp(row)
        return exp_row / exp_row.sum()

    result2 = apply_row_wise(input_tensor, custom_softmax)
    print("\nResult after applying custom softmax:")
    print(result2)

    # Example 3: Using a lambda function
    result3 = apply_row_wise(input_tensor, lambda x: x ** 2 + 1)
    print("\nResult after applying x^2 + 1:")
    print(result3)

    # Example tensor
    x = tensor([0.1, 0.5, 0.9], dtype=float32)

    # Apply norm.ppf in one line
    y = from_numpy(norm.ppf(x.numpy()))

    print(y)


if __name__ == "__main__":
    main()
