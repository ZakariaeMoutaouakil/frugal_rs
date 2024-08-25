from typing import Callable

from torch import Tensor, tensor


def apply_function_to_tensor(x: Tensor, func: Callable[[float], float]) -> Tensor:
    """
    Applies a function element-wise to a 1D PyTorch tensor.

    Args:
    tensor (torch.Tensor): Input 1D tensor.
    func (Callable[[float], float]): Function to apply to each element.

    Returns:
    torch.Tensor: New tensor with the function applied element-wise.

    Raises:
    ValueError: If the input tensor is not 1D.
    """
    if x.dim() != 1:
        raise ValueError("Input tensor must be 1-dimensional")

    # Convert to CPU, apply function, then move back to original device
    cpu_tensor = x.cpu()
    result = tensor([func(x.item()) for x in cpu_tensor], dtype=x.dtype)
    return result.to(x.device)


def main():
    # Example usage
    x = tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = apply_function_to_tensor(x, lambda y: y ** 2)
    print(result)  # x([ 1.,  4.,  9., 16., 25.])


if __name__ == "__main__":
    main()
