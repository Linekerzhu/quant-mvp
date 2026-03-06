import numpy as np
from scipy.stats import skew, kurtosis

# replicate Scenario 3.5
np.random.seed(42)
metrics = [0.55 + 0.06 * (i / 14) + np.random.normal(0, 0.005) for i in range(15)]
baselines = [0.5] * 15

excess = np.array(metrics) - np.array(baselines)
mean_excess = np.mean(excess)
std_excess = np.std(excess, ddof=1)
n = len(excess)

skewness = np.mean(((excess - mean_excess) / std_excess) ** 3)
kurt_calc = np.mean(((excess - mean_excess) / std_excess) ** 4) - 3

print(f"Mean: {mean_excess}, Std: {std_excess}")
print(f"Skew: {skewness}, Kurtosis: {kurt_calc}")

sr = mean_excess / (std_excess / np.sqrt(n))
print(f"SR: {sr}")

expected_max = 0.0
correction = 1 + (skewness * sr / 6) - (kurt_calc * (sr**2 - 1) / 24)
print(f"Correction: {correction}")

dsr = (sr - expected_max) / max(correction, 0.1)
print(f"DSR: {dsr}")
