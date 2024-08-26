from typing import Tuple, Callable

from clopper.calculate_coefficients import calculate_coefficients
from clopper.jit_margin_proba import g
from clopper.margin_proba import margin_proba


def objective_function(p: Tuple[float, float, float], L: float, coefficients: Tuple[Tuple[int, int, int, int], ...],
                       observation: float, h: Callable[[float], float]) -> float:
    if (p[0] + p[1] > 1) or (h(p[0]) - h(p[1]) > L) or (p[0] < 0.5):
        return 1
    return margin_proba(observation=observation, p=p, coefficients=coefficients, h=h)


def main():
    # Example parameters
    n = 10
    observation = 0.2
    L = 0.5
    p_values = [
        (0.6, 0.3, 0.1),
        (0.5, 0.4, 0.1),
        (0.7, 0.2, 0.1),
        (0.8, 0.1, 0.1),
        (0.4, 0.3, 0.3)  # This should return 1 due to p[0] < 0.5
    ]

    coefficients = calculate_coefficients(n)

    print(f"Number of trials (n): {n}")
    print(f"Observation threshold: {observation}")
    print(f"L value: {L}")
    print("\nResults:")

    for p in p_values:
        result = objective_function(p, L, coefficients, observation, g)
        print(f"p = {p}: objective value = {result:.6f}")

    # Timing test
    import time

    num_runs = 10000
    start_time = time.time()
    for _ in range(num_runs):
        objective_function(p_values[0], L, coefficients, observation, g)
    end_time = time.time()

    print(f"\nTime for {num_runs} runs: {end_time - start_time:.6f} seconds")
    print(f"Average time per run: {(end_time - start_time) / num_runs * 1e6:.2f} microseconds")


if __name__ == "__main__":
    main()
