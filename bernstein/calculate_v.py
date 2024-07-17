from torch import cuda, Tensor, tensor, zeros


def calculate_v(x: Tensor, mu: Tensor) -> Tensor:
    t = x.size(0)
    v = zeros(t, device=x.device)

    # For i > 1, we can use the formula directly
    for i in range(1, t):
        v[i] = 4 * (x[i] - mu[i - 1]) ** 2

    return v


def main():
    # Example usage
    x = tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device='cuda' if cuda.is_available() else 'cpu')
    mu = tensor([0., 0.75, 1.1667, 1.6250, 2.1000, 2.5833], device=x.device)

    v = calculate_v(x, mu)

    print("Input tensor x:", x)
    print("Means tensor mu:", mu)
    print("Calculated v:", v)

    print("\nCalculated v values:")
    for i, v_val in enumerate(v):
        print(f"v{i} = {v_val:.4f}")


if __name__ == "__main__":
    main()
