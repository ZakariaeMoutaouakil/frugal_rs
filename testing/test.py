from scipy.stats import norm

from bernstein_optimized.calculate_shift import calculate_shift
from bernstein_optimized.calculate_shift_upper import calculate_shift_upper
from testing.gaussian_quantile_approximation import gaussian_quantile_approximation
from testing.reduce_matrix_rows import reduce_matrix_rows
from testing.sample_dirichlet import sample_dirichlet

alpha = (10.0, 2.0, 1.0)
num_samples = 1000
seed = 42

samples = sample_dirichlet(alpha, num_samples, seed=seed)

print(f"Generated {num_samples} samples from Dirichlet({alpha}):")
print(samples)

order = 25
quantile_function = gaussian_quantile_approximation(order)
x = reduce_matrix_rows(samples, lambda p1, p2, p3: quantile_function(p1) - quantile_function(p2))
print(f"x = {x}")

maximum = quantile_function(1) - quantile_function(0)
print(f"maximum = {maximum}")
x_normalized = x / maximum
print(f"x_normalized = {x_normalized}")

alpha_0 = 0.001
lower_bound = calculate_shift(x_normalized, alpha_0 / 2)
print(f"lower_bound = {lower_bound}")
unnormalized_lower_bound = lower_bound * maximum
print(f"unnormalized_lower_bound = {unnormalized_lower_bound}")

p1 = samples[:, 0]
print(f"p1 = {p1}")
p1_normalized = 2 * (p1 - 0.5)
p2 = samples[:, 1]
print(f"p2 = {p2}")
p2_normalized = 2 * p2
p1_ = calculate_shift(p1_normalized, alpha_0 / 2)
print(f"p1_ = {p1_}")
p2_ = calculate_shift_upper(p2_normalized, alpha_0 / 2)
print(f"p2_ = {p2_}")
p1__ = (p1_ + 1) / 2
p2__ = p2_ / 2
ref_lower_bound = norm.ppf(p1__) - norm.ppf(p2__)
print(f"ref_lower_bound = {ref_lower_bound}")
print(f"lower_bound = {unnormalized_lower_bound}")
