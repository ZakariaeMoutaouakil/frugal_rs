from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define normalization functions
def tempered_softmax(x: torch.Tensor, temperature: float = 1.0, dim: int = -1) -> torch.Tensor:
    return F.softmax(x / temperature, dim=dim)


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Smoothed predict function
def smoothed_predict(model: nn.Module, x: torch.Tensor, num_samples: int, noise_sd: float,
                     normalize_fn: Callable[[torch.Tensor], torch.Tensor] = F.softmax) -> torch.Tensor:
    with torch.no_grad():
        noisy_inputs = x.repeat(num_samples, 1, 1, 1) \
                       + torch.randn_like(x.repeat(num_samples, 1, 1, 1)) * noise_sd
        outputs = model(noisy_inputs)
        return normalize_fn(outputs)


# Test function
def test_normalization(model, x, num_samples, noise_sd, normalize_fn, name):
    predictions = smoothed_predict(model, x, num_samples, noise_sd, normalize_fn)
    print(f"{name} - Predictions:")
    print(predictions)
    row_sums = torch.sum(predictions, dim=1)
    print(f"{name} - Row sums:")
    print(row_sums)
    print(f"All rows sum to 1: {torch.allclose(row_sums, torch.ones_like(row_sums))}")
    print(f"Mean: {torch.mean(row_sums):.6f}")
    print(f"Std: {torch.std(row_sums):.6f}")
    print(f"Min: {torch.min(row_sums):.6f}")
    print(f"Max: {torch.max(row_sums):.6f}")
    print("\n")


# Main script
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Create a simple model and dummy input
    model = SimpleCNN()
    x = torch.randn(1, 3, 32, 32)  # Single image, 3 channels, 32x32 pixels

    num_samples = 10
    noise_sd = 0.1

    # Test different normalization functions
    test_normalization(model, x, num_samples, noise_sd, F.softmax, "Softmax")
    test_normalization(model, x, num_samples, noise_sd, lambda x: tempered_softmax(x, temperature=0.5),
                       "Tempered Softmax (T=0.5)")
    test_normalization(model, x, num_samples, noise_sd, lambda x: tempered_softmax(x, temperature=2.0),
                       "Tempered Softmax (T=2.0)")
