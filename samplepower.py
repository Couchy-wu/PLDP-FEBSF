import numpy as np
import matplotlib.pyplot as plt

def power_law_sampling(alpha, x_min, x_max, size=1):
    # Generate uniform random numbers between 0 and 1
    u = np.random.rand(size)

    # Apply the inverse cumulative distribution function (CDF) to get power-law samples
    x = (x_max**(1-alpha) - x_min**(1-alpha)) * u + x_min**(1-alpha)
    x = x**(1/(1-alpha))

    return x

# Parameters of the power-law distribution
alpha = 1.5  # The power-law exponent (should be greater than 1)
x_min = 1.0  # The minimum value of the distribution
x_max = 100.0  # The maximum value of the distribution
sample_size = 1000

# Generate power-law samples
samples = power_law_sampling(alpha, x_min, x_max, sample_size)

# Define the specified interval
interval_min = 10.0
interval_max = 50.0

# Count the number of samples that fall into the specified interval
samples_in_interval = len([sample for sample in samples if interval_min <= sample <= interval_max])

# Plot the histogram of the generated samples
plt.hist(samples, bins=50, density=True, alpha=0.6)
plt.axvline(interval_min, color='red', linestyle='dashed', linewidth=2, label='Interval')
plt.axvline(interval_max, color='red', linestyle='dashed', linewidth=2)
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Power-law Distribution Sampling')
plt.legend()
plt.show()

# Print the number of samples in the specified interval
print(f"Number of samples in the interval [{interval_min}, {interval_max}]: {samples_in_interval}")

# 使用本方案进行幂律分布的采样