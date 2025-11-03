
from numpy.random import seed, gamma
import numpy as np
import matplotlib.pyplot as plt
import math
import random


population_size   = 200_000_000
number_of_samples = 1_000_000
number_of_bins    = 100
sample_sizes      = [10, 30, 50]

populations = [
    ("Normal(μ=0,σ=1)",             lambda size: np.random.normal(loc=0, scale=1, size=size)),
    ("Uniform[0,1]",                lambda size: np.random.random(size=size)),
    ("Gamma(shape=2,scale=2)",      lambda size: gamma(shape=2, scale=2, size=size)),
    ("Beta(α=5,β=5)",               lambda size: np.random.beta(a=5, b=5, size=size)),
    ("Beta(α=2,β=5)",               lambda size: np.random.beta(a=2, b=5, size=size)),
]

seed(1)
random.seed(1)

fig_num = 1
for dist_name, sampler in populations:

    population = sampler(population_size)
    plt.figure(fig_num); fig_num += 1
    plt.hist(population, number_of_bins)
    plt.title(f"Population distribution ({dist_name})")
    plt.xlabel("x"); plt.ylabel("frequency")
    plt.show()

    pop_mean = np.mean(population)
    pop_sd   = np.std(population, ddof=1)

    for n in sample_sizes:
        sample_means = np.zeros(number_of_samples)
        sample_sds   = np.zeros(number_of_samples)

        for i in range(number_of_samples):
            samp = random.choices(population, k=n)
            sample_means[i] = np.mean(samp)
            sample_sds[i]   = np.std(samp, ddof=1)

        plt.figure(fig_num); fig_num += 1
        plt.hist(sample_means, number_of_bins)
        plt.title(f"Sample‐mean distribution (n={n}) from {dist_name}")
        plt.xlabel("sample mean"); plt.ylabel("frequency")
        plt.show()

        mean_of_means = np.mean(sample_means)
        sd_of_means   = np.std(sample_means, ddof=1)
        theo_sd       = pop_sd / math.sqrt(n)

        print(f"[{dist_name}, n={n}] pop mean = {pop_mean:.4f}, "
              f"mean(sample means) = {mean_of_means:.4f}")
        print(f"[{dist_name}, n={n}] pop sd = {pop_sd:.4f}, "
              f"sd(sample means) = {sd_of_means:.4f}, "
              f"theoretical sd = {theo_sd:.4f}\n")
