import sys
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Ito's lemma is a concept used in the Geometric Brownian Motion and the Stochastic Calculus
# It corrects for the fact that stock prices follow a lognormal distribution rather than a simple normal distribution
ITO = 0.5

CONFIDENCE_95 = 5  # for 95% confidence
CONFIDENCE_99 = 1  # for 99% confidence

# This is the number of days to run the simulations over, not the days downloaded from yfinance
T = 5  # trading days

# Download stock data: historical data will be used to calculate returns, log returns, mean and std
stock = yf.download("AAPL", start="2024-01-01", end="2025-01-01")

# Calculate daily log returns
stock["Returns"] = stock['Close'] / stock['Close'].shift(1)
stock['Log Returns'] = np.log(stock['Close'] / stock['Close'].shift(1))

stock = stock.dropna()

# estimate mean and volatility
mu = stock['Log Returns'].mean()
sigma = stock['Log Returns'].std()

print(f"Mean Daily Return: {mu:.5f}, Volatility: {sigma:.5f}")
print()

# This variables will be used for the matrix
S = 10  # no of shares
N = 1000  # Number of simulations
all_S = stock["Close"].values.tolist()
S0 = round(all_S[-1][0], 3) * S  # Last closing price * no of shares (portfolio value)

# daily time step: if weekly then 1 / 52
dt = 1 / 252  # time step
number_of_steps = T  # int(T / dt)


# Initialize matrix for simulated prices
if T == 1:
    simulated_prices = np.zeros((number_of_steps + 1, N))  # N+1 to include initial price
    simulated_prices[0] = S0  # Set initial price for all simulations
elif T > 1:
    simulated_prices = np.zeros((number_of_steps, N))
    simulated_prices[0] = S0  # Set initial price for all simulations

print(f"Initial portfolio ${round(S0, 3)}")

# Generate price paths: simulated_prices contains a list with N simulations
# per each day: if N=10 and trading_days=21. it will be a list with 21 lists of len 10
if T == 1:
    rand = np.random.normal(0, 1, N)  # Generate random stocks prices
    simulated_prices[1] = simulated_prices[0] * np.exp((mu - ITO * sigma**2) + sigma * rand * np.sqrt(dt))
elif T > 1:
    for t in range(1, number_of_steps):
        rand = np.random.normal(0, 1, N)  # Generate random stocks prices
        simulated_prices[t] = simulated_prices[t-1] * np.exp((mu - ITO * sigma**2) * dt + sigma * rand * np.sqrt(dt))


# VaR is calculated out of the final stock price
percentile_95 = np.percentile(simulated_prices[-1], CONFIDENCE_95)  # worst case out of the last simulation, a scalar
percentile_95_array = np.percentile(simulated_prices, CONFIDENCE_95, axis=1)  # this is an array used to plot
max_loss_95 = -round(((S0 - percentile_95) / S0) * 100, 3)
VaR_95 = S0 - percentile_95

percentile_99 = np.percentile(simulated_prices[-1], CONFIDENCE_99)
percentile_99_array = np.percentile(simulated_prices, CONFIDENCE_99, axis=1)
max_loss_99 = -round(((S0 - percentile_99) / S0) * 100, 3)
VaR_99 = S0 - percentile_99


# Min expected price during the whole time period: this can be lower than the VaR
lowest_price = round(np.min(simulated_prices), 3)
lowest_price_perc = -round(((S0 - lowest_price) / S0) * 100, 3)

# Expected Shortfall (ES): the average loss in the worst-case scenarios
shortfall_95 = simulated_prices[simulated_prices < percentile_95].mean()
exp_shortfall_95 = S0 - shortfall_95
exp_shortfall_95_perc = round(exp_shortfall_95 / S0 * 100, 3)

shortfall_99 = simulated_prices[simulated_prices < percentile_99].mean()
exp_shortfall_99 = S0 - shortfall_99
exp_shortfall_99_perc = round(exp_shortfall_99 / S0 * 100, 3)

print("--- Level of confidence 95% ---")
print(f"Value at Risk (VaR) at 95% confidence after {T} days: ${VaR_95:.3f}")
print(f"Max Loss % after {T} days at 95% confidence: {max_loss_95}%")
print(f"Lowest expected portfolio value during a time period of {T} days: ${lowest_price}")
print()
print("--- Level of confidence 99% ---")
print(f"Value at Risk (VaR) at 99% confidence after {T} days: ${VaR_99:.3f}")
print(f"Max Loss % after {T} days at 99% confidence: {max_loss_99}%")
print(f"Lowest expected portfolio value during a time period of {T} days: ${lowest_price}")
print()
print("--- Expected  Shortfall ---")
print(f"ES or cVaR beyond the confidence interval 95%: ${exp_shortfall_95:.3f} (-{exp_shortfall_95_perc}%)")
print(f"ES or cVaR beyond the confidence interval 99%: ${exp_shortfall_99:.3f} (-{exp_shortfall_99_perc}%)")

# CHOLESKY DECOMPOSITION IS FOR MULTIPLE ASSETS

sys.exit()

# Plot All Simulations with Confidence Intervals
plt.figure(figsize=(12, 6))
plt.plot(simulated_prices, alpha=0.1, color="blue")  # All simulations with transparency
plt.plot(percentile_99_array, color='red', linestyle='dashed', label="99% Worst Case")
plt.plot(percentile_95_array, color='orange', linestyle='dashed', label="95% Worst Case")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title(f"Monte Carlo Simulation of AAPL Stock Price ({N} Simulations)")
plt.legend()
plt.show()

# Step 3: Plot Distribution & VaR Lines
plt.figure(figsize=(10, 5))
plt.hist(simulated_prices[-1], bins=50, color='blue', alpha=0.6, density=True)
plt.axvline(percentile_95, color='orange', linestyle='dashed', linewidth=2, label=f'VaR 95%: ${percentile_95:,.0f}')
plt.axvline(percentile_99, color='red', linestyle='dashed', linewidth=2, label=f'VaR 99%: ${percentile_99:,.0f}')
plt.xlabel("Portfolio Loss ($)")
plt.ylabel("Frequency")
plt.title("Value at Risk (VaR) at 95% and 99% Confidence Levels")
plt.legend()
plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("OK!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
