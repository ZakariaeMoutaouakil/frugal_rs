from math import log


def psi_e(lambda_val: float) -> float:
    return (-log(1 - lambda_val) - lambda_val) / 4


def main():
    # Example usage
    lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("λ\tψE(λ)")
    print("-----------------")
    for lambda_val in lambda_values:
        result = psi_e(lambda_val)
        print(f"{lambda_val:.1f}\t{result:.4f}")


if __name__ == "__main__":
    main()
