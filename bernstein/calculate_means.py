from torch import cuda, Tensor, sum, tensor, zeros


def calculate_means(x: Tensor, debug: bool = False) -> Tensor:
    t = x.size(0)
    means = zeros(t, device=x.device)

    for i in range(2, t + 1):
        if debug:
            print(f"i = {i}")
            print(f"x[1:i] = {x[1:i]}")
        numerator = 0.5 + sum(x[1:i])
        denominator = i
        if debug:
            print(f"denominator = {denominator}")
        mean = numerator.item() / denominator
        means[i - 1] = mean

    return means


def main() -> None:
    # Example usage
    x: Tensor = tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device='cuda' if cuda.is_available() else 'cpu')
    result: Tensor = calculate_means(x, debug=True)

    print("Input tensor:", x)
    print("Input tensor device:", x.device)
    print("Calculated means tensor:", result)
    print("Result tensor device:", result.device)
    print("\nCalculated means:")
    for i, mean in enumerate(result):
        print(f"μ{i} = {mean:.4f}")


if __name__ == "__main__":
    main()
