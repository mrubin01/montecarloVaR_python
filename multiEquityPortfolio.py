import sys

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

# Simulation parameters
num_simulations = 1000
T = 21  # trading days
dt = 1 / 252  # Daily time step in years, for weekly time steps use 1 / 52

# Define stock tickers and number of shares
tickers = ['AAPL', 'CIM', 'CVX']
num_shares = [10, 10, 10]  # 10 shares each

# Fetch data (adjust the period as needed)
df = yf.download(tickers, start="2023-01-01", end="2024-01-01")['Close']

# Drop NaN values (if any)
df.dropna(inplace=True)

# Calculate log returns
log_returns = np.log(df / df.shift(1)).dropna()

# Compute mean and covariance of log returns
mean_returns = log_returns.mean() * 252  # annualized
cov_matrix = log_returns.cov() * 252  # annualized

# print("Mean Returns:\n", mean_returns)
# print("Covariance Matrix:\n", cov_matrix)

# Use the Cholesky decomposition to factorize the covariance
# matrix into a lower triangular matrix

# Perform Cholesky decomposition
L = cholesky(cov_matrix, lower=True)
# print("Cholesky Decomposition Matrix:\n", L)

# Generate random normal numbers
rand_normals = np.random.randn(T, len(tickers), num_simulations)

# Apply Cholesky decomposition for correlated shocks
correlated_shocks = np.einsum('ij,tjs->tis', L, rand_normals)


# Simulate price paths using GBM
simulated_prices = np.zeros((T + 1, len(tickers), num_simulations))
simulated_prices[0] = df.iloc[-1].values[:, np.newaxis]  # Start at last known prices

for t in range(1, T + 1):
    drift = (mean_returns.values[:, np.newaxis] - 0.5 * np.diag(cov_matrix)[:, np.newaxis]) * dt
    diffusion = correlated_shocks[t - 1] * np.sqrt(dt)
    simulated_prices[t] = simulated_prices[t - 1] * np.exp(drift + diffusion)

# Example simulated path for AAPL
# plt.plot(simulated_prices[:, 0, 0])
# plt.title("Example Simulated Path for AAPL")
# plt.show()

# Compute portfolio values
initial_value = np.dot(df.iloc[-1].values, num_shares)
final_values = np.dot(simulated_prices[-1].T, num_shares)

# Compute profit/loss distribution
losses = initial_value - final_values

# Compute 95% Value at Risk
VaR_95 = np.percentile(losses, 5)
VaR_95_perc = round((VaR_95 / initial_value) * 100, 3)

# Compute 99% Value at Risk
VaR_99 = np.percentile(losses, 1)
VaR_99_perc = round((VaR_99 / initial_value) * 100, 3)


print(f"Initial Portfolio Value: ${initial_value:.2f}")
print()
print(f"{T}-day 95% Monte Carlo VaR: ${VaR_95:.2f} ({VaR_95_perc}%)")
print(f"{T}-day 99% Monte Carlo VaR: ${VaR_99:.2f} ({VaR_99_perc}%)")

# Plot loss distribution
plt.hist(losses, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(VaR_95, color='r', linestyle='dashed', linewidth=2, label=f"VaR 95%: ${VaR_95:.2f}")
plt.title("Loss Distribution & VaR")
plt.xlabel("Loss ($)")
plt.ylabel("Frequency")
plt.legend()
plt.show()
