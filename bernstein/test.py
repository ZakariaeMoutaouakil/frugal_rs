from random import random

from numpy import array
from torch import tensor, cuda

from bernstein.calculate_lambdas import calculate_lambdas
from bernstein.calculate_means import calculate_means
from bernstein.calculate_sigmas import calculate_sigmas
from bernstein.calculate_term import calculate_term
from bernstein.calculate_terms import calculate_terms
from bernstein.calculate_v import calculate_v
from bernstein.calculate_weighted_means import calculate_weighted_means

t = 6000
alpha_0 = 0.001
alpha = alpha_0 * 2

vector = [random() * random() ** 0.5 for _ in range(t)]
x1 = array(vector)
print(f"x1 = {x1}")
x = tensor([0.] + vector, device='cuda' if cuda.is_available() else 'cpu')
print(f"x = {x}")

mean = x.mean()
print("""mean = {}""".format(mean))

means = calculate_means(x=x)
print(f"means = {means}")

sigmas = calculate_sigmas(x=x, mu=means)
print(f"sigmas = {sigmas}")

lambdas = calculate_lambdas(x=x, sigma=sigmas, alpha=alpha)
print(f"lambdas = {lambdas}")

weighted_means = calculate_weighted_means(x=x, lambdas=lambdas)
print(f"weighted_means = {weighted_means}")

v = calculate_v(x=x, mu=weighted_means)
print(f"v = {v}")

terms = calculate_terms(x=x, lambdas=lambdas, v=v, alpha=alpha)
print(f"terms = {terms}")

lower_bound = weighted_means[-1] - terms[-1]
print(f"lower_bound = {lower_bound}")

weighted_mean = weighted_means[-1]
print(f"weighted_mean = {weighted_mean}")

upper_bound = weighted_means[-1] + terms[-1]
print(f"upper_bound = {upper_bound}")

term = calculate_term(vector=x1, alpha=alpha_0)
print(f"term = {term}")

ref_lower_bound = mean - term
print(f"ref_lower_bound = {ref_lower_bound}")

ref_upper_bound = mean + term
print(f"ref_upper_bound = {ref_upper_bound}")
