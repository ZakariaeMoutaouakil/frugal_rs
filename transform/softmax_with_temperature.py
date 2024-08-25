from torch import Tensor, tensor
from torch.nn.functional import softmax


def softmax_with_temperature(logits: Tensor, temperature: float) -> Tensor:
    """
    Apply softmax with temperature to each row of the input tensor.

    Args:
    logits (torch.Tensor): Input tensor of shape (batch_size, num_classes)
    temperature (float): Temperature parameter for softmax

    Returns:
    torch.Tensor: Tensor of same shape as input with softmax applied to each row
    """
    if temperature <= 0:
        raise ValueError("Temperature must be a positive value")

    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Apply softmax
    return softmax(scaled_logits, dim=1)


def main():
    # Example usage
    # Create a sample 2D tensor
    logits = tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])

    print("Original logits:")
    print(logits)

    # Apply softmax with different temperatures
    temperatures = [0.5, 1.0, 2.0]

    for temp in temperatures:
        result = softmax_with_temperature(logits, temp)
        print(f"\nSoftmax with temperature {temp}:")
        print(result)


if __name__ == "__main__":
    main()
