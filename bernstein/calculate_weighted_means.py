import torch


def calculate_weighted_means(x: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    t = x.size(0)
    weighted_means = torch.zeros(t, device=x.device)

    lambda_sum = 0.
    weighted_sum = 0.

    for i in range(1, t):
        lambda_sum += lambdas[i]
        weighted_sum += lambdas[i] * x[i]
        weighted_means[i] = weighted_sum / lambda_sum

    return weighted_means


def main():
    # Example usage
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device='cuda' if torch.cuda.is_available() else 'cpu')
    lambdas = torch.tensor([0., 0.5, 0.4, 0.3, 0.2, 0.1], device=x.device)

    weighted_means = calculate_weighted_means(x, lambdas)

    print("Input tensor x:", x)
    print("Lambda tensor:", lambdas)
    print("Calculated weighted means:", weighted_means)

    print("\nCalculated weighted means:")
    for i, mean in enumerate(weighted_means):
        print(f"Weighted mean up to x{i} = {mean:.4f}")


if __name__ == "__main__":
    main()
