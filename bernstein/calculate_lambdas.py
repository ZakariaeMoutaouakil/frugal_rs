from math import log, sqrt

from torch import cuda, Tensor, tensor, zeros


def calculate_lambdas(x: Tensor, sigma: Tensor, alpha: float, c: float = 0.5, debug: bool = False) -> Tensor:
    t = x.size(0)
    lambdas = zeros(t, device=x.device)
    lambdas[0] = 0.

    for i in range(1, t):
        numerator = 2 * log(2 / alpha)
        denominator = sigma[i - 1] * i * log(1 + i)

        lambdas[i] = min(sqrt(numerator / denominator), c)

        if debug:
            print(f"i = {i}")
            print(f"numerator = {numerator}")
            print(f"denominator = {denominator}")
            print(f"lambdas: {lambdas}")

    return lambdas


def main():
    # Example usage
    x = tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device='cuda' if cuda.is_available() else 'cpu')
    sigma = tensor([0.2500, 0.3536, 0.4082, 0.4564, 0.5000, 0.5408], device=x.device)
    c = 0.75
    alpha = 0.05

    lambdas = calculate_lambdas(x, sigma, alpha, c, debug=True)

    print("Input tensor x:", x)
    print("Sigmas tensor:", sigma)
    print("c value:", c)
    print("alpha value:", alpha)
    print("Calculated lambdas:", lambdas)

    print("\nCalculated lambdas:")
    for i, lambda_val in enumerate(lambdas):
        print(f"λ{i} = {lambda_val:.4f}")


if __name__ == "__main__":
    main()
