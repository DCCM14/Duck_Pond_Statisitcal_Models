
from numpy.random import seed, gamma
import numpy as np
import matplotlib.pyplot as plt
import math
import random


population_size    = 200_000_000
number_of_samples  = 10_000
number_of_bins     = 100
sample_sizes       = [10, 30, 50]
confidence_level   = 95

if confidence_level == 90:
    z_alpha2 = 1.645
elif confidence_level == 95:
    z_alpha2 = 1.96
elif confidence_level == 99:
    z_alpha2 = 2.58
else:
    raise ValueError("Unsupported confidence level")

populations = [
    ("Normal(μ=0,σ=1)",             lambda size: np.random.normal(loc=0, scale=1, size=size)),
    ("Uniform[0,1]",                lambda size: np.random.random(size=size)),
    ("Gamma(shape=2,scale=2)",      lambda size: gamma(shape=2, scale=2, size=size)),
    ("Beta(α=5,β=5)",               lambda size: np.random.beta(a=5, b=5, size=size)),
    ("Beta(α=2,β=5)",               lambda size: np.random.beta(a=2, b=5, size=size)),
]

seed(1)
random.seed(1)

for dist_name, sampler in populations:
    population    = sampler(population_size)
    pop_mean      = np.mean(population)
    pop_sd        = np.std(population, ddof=1)

    plt.figure()
    plt.hist(population, number_of_bins)
    plt.title(f"Population ({dist_name})")
    plt.xlabel("x"); plt.ylabel("frequency")
    plt.show()

    for n in sample_sizes:
        intervals = np.zeros((number_of_samples, 3))
        successes = 0

        for i in range(number_of_samples):
            samp = random.choices(population, k=n)
            xbar = np.mean(samp)
            s    = np.std(samp, ddof=1)
            half = z_alpha2 * s / math.sqrt(n)
            low, high = xbar-half, xbar+half
            intervals[i] = (low, high, 1 if (low < pop_mean < high) else 0)
            successes += intervals[i,2]

        rel_freq = successes/number_of_samples

        print(f"[{dist_name}, n={n}] pop mean={pop_mean:.4f}, pop sd={pop_sd:.4f}")
        print(f"  ConfLevel={confidence_level}%, Successes={successes}, "
              f"Trials={number_of_samples}, RelFreq={rel_freq:.4f}\n")

        plt.figure()
        plt.plot([pop_mean-2*pop_sd, pop_mean+2*pop_sd], [0,0], 'k-')
        plt.plot([pop_mean, pop_mean], [-0.5, 10.5], 'k-')

        for i in range(number_of_samples):
            y = i*(10/number_of_samples)
            color = 'b' if intervals[i,2] else 'r'
            plt.plot([intervals[i,0], intervals[i,1]], [y,y], color=color, linewidth=0.4)


        plt.text(0.90*pop_mean, -1, '$\\mu$', color='green', fontsize='large')
        plt.text(pop_mean-pop_sd, -1.8, f"(Confidence level: {confidence_level}%)",
                 color='green', fontsize='large')
        plt.text(pop_mean-2.2*pop_sd, 11.0,
                 f"(Successes: {successes}, Trials: {number_of_samples}, RelFreq: {rel_freq:.4f})",
                 color='green', fontsize='large')

        plt.axis('off')
        plt.xlabel('$\\mu$')
        plt.show()
