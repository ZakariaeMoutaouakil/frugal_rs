from math import log, sqrt

from torch import Tensor, var


def calculate_term(vector: Tensor, alpha: float) -> float:
    """
    Calculate the term given by the formula in the image.

    Args:
    vector (torch.Tensor): Input vector
    alpha (float): Alpha value (should be between 0 and 1)

    Returns:
    float: Calculated term
    """
    n = vector.numel()  # Get the total number of elements in the tensor
    if n < 2:
        raise ValueError("Calculation requires at least two data points.")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha should be strictly between 0 and 1.")

    # Calculate variance (sample variance)
    variance = var(vector, unbiased=True).item()
    log_term = log(2 / alpha)

    term1 = (2 * variance * log_term) / n
    term2 = (7 * log_term) / (3 * (n - 1))

    result = sqrt(term1) + term2
    return result
