from torch import cuda, Tensor, sqrt, sum, tensor, zeros


def calculate_sigmas(x: Tensor, mu: Tensor, debug: bool = False) -> Tensor:
    t = x.size(0)
    sigmas = zeros(t, device=x.device)
    sigmas[0] = 0.25

    for i in range(2, t + 1):
        if debug:
            print(f"i = {i}")
            print(f"x[1:i] = {x[1:i]}")
            print(f"mu[1:i] = {mu[1:i]}")
        numerator = 0.25 + sum((x[1:i] - mu[1:i]) ** 2)
        denominator = i
        sigmas[i - 1] = sqrt(numerator / denominator)
        if debug:
            print(f"numerator = {numerator}")
            print(f"denominator = {denominator}")
            print(f"sigmas = {sigmas}")

    return sigmas


def main():
    # Example usage
    x = tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device='cuda' if cuda.is_available() else 'cpu')
    mu = tensor([0., 0.75, 1.1667, 1.6250, 2.1000, 2.5833], device=x.device)

    sigmas = calculate_sigmas(x, mu, debug=True)

    print("Input tensor x:", x)
    print("Means tensor mu:", mu)
    print("Calculated sigmas:", sigmas)

    print("\nCalculated sigmas:")
    for i, sigma in enumerate(sigmas):
        print(f"σ{i} = {sigma:.4f}")


if __name__ == "__main__":
    main()
