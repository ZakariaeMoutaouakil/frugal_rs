from typing import Callable

from hyperopt import fmin, tpe, hp, Trials
from statsmodels.stats.proportion import proportion_confint

from clopper.calculate_coefficients import calculate_coefficients
from clopper.objective_function import objective_function


def create_stop_fn(alpha: float):
    def stop_fn(trials: Trials):
        if not trials.trials:  # Check if there are any trials
            return False, {}
        last_loss = trials.best_trial['result']['loss']
        return last_loss < 1 - alpha, {}

    return stop_fn


def hyperopt_optimization(objective: Callable[[float, float], float], alpha: float) -> float:
    space = {
        'p0': hp.uniform('p0', 0, 1),
        'p1': hp.uniform('p1', 0, 1)
    }

    stop_fn = lambda trials, alpha: stop_fn(trials, alpha)
    trials = Trials()
    fmin(fn=lambda x: objective(x['p0'], x['p1']),  # Changed this line
         space=space,
         algo=tpe.suggest,
         timeout=60,
         max_evals=1000,
         trials=trials,
         early_stop_fn=create_stop_fn(alpha))

    best_value = trials.best_trial['result']['loss']
    return best_value


def main() -> None:
    # Example usage
    x = (25, 3, 0)
    n = sum(x)
    alpha = 0.001
    p0 = proportion_confint(x[0], n, alpha=alpha, method="beta")[0]
    p1 = proportion_confint(x[1], n, alpha=alpha, method="beta")[1]
    ref_L = p0 - p1
    print(f"Reference L: {ref_L}")
    coefficients = calculate_coefficients(n)
    observation = (x[0] - x[1]) / n
    objective = lambda p0, p1: objective_function((p0, p1, 1 - p0 - p1), ref_L, coefficients, observation,
                                                  lambda x: x)
    best_value = hyperopt_optimization(objective, alpha)
    print(f"Best value: {best_value}")
    print(f"Best parameters: {(1 - best_value)}")

    objective = lambda p0, p1: objective_function((p0, p1, 1 - p0 - p1), observation, coefficients, observation,
                                                  lambda x: x)
    best_value = hyperopt_optimization(objective, alpha)
    print(f"Best value: {best_value}")
    print(f"Best parameters: {(1 - best_value)}")


if __name__ == '__main__':
    main()
