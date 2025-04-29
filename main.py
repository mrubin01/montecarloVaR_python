import sys
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy.linalg import cholesky
import warnings
warnings.filterwarnings("ignore")
import functions

# Ito's lemma is a concept used in the Geometric Brownian Motion and the Stochastic Calculus
# It corrects for the fact that stock prices follow a lognormal distribution rather than a simple normal distribution
ITO = 0.5
DT = 1 / 252  # daily time step: if weekly then 1 / 52

CONFIDENCE_95 = 5  # for 95% confidence
CONFIDENCE_99 = 1  # for 99% confidence

yesterday = datetime.now() - timedelta(days=1)
END_DATE = yesterday.strftime('%Y-%m-%d')

today = datetime.now()
START_DATE = datetime(today.year - 1, 1, 1).strftime("%Y-%m-%d")

print_charts = False

print(f"<-- It will be used data starting from {START_DATE} to {END_DATE} -->")
print()
user_input = input("How many stocks in the portfolio: ")


def main():
    if int(user_input) == 1:
        ticker = input("Type the ticker: ").upper()

        # Download stock data: historical data will be used to calculate returns, log returns, mean and std
        try:
            stock = yf.download(ticker, start=START_DATE, end=END_DATE)
        except Exception as e:
            print(e)
            sys.exit()

        # Check that yfinance does not return an empty dataframe
        if stock.empty:
            print(f"No data returned for ticker '{ticker}'")
            sys.exit()

        # Calculate daily log returns
        stock["Returns"] = stock['Close'] / stock['Close'].shift(1)
        stock['Log Returns'] = np.log(stock['Close'] / stock['Close'].shift(1))

        stock = stock.dropna()

        # estimate mean and volatility
        mu = stock['Log Returns'].mean()
        sigma = stock['Log Returns'].std()

        print()
        print(f"Mean Daily Log Return: {mu:.5f}, Log Volatility: {sigma:.5f}")
        print()

        # These variables will be used for the matrix
        S = int(input(f"How many shares for {ticker}? "))  # no of shares
        N = int(input(f"How many simulations you want to run? Try at least 1000 "))  # Number of simulations
        T = int(input(f"How many trading days? The minimum is 1 "))  # trading days to run the simulations over, not the days downloaded from yfinance

        # Use the last known price to calculate the initial value of the portfolio
        all_S = stock["Close"].values.tolist()
        S0 = round(all_S[-1][0], 3) * S  # Last closing price * no of shares

        number_of_steps = T

        # Initialize matrix for simulated prices
        if T == 1:
            simulated_prices = functions.init_matrix(number_of_steps + 1, N)
            simulated_prices[0] = S0  # Set initial price for all simulations
        elif T > 1:
            simulated_prices = functions.init_matrix(number_of_steps, N)
            simulated_prices[0] = S0  # Set initial price for all simulations

        print()
        print(f"Initial portfolio value: ${round(S0, 3)}")
        print()

        # Generate price paths: simulated_prices contains a list with N simulations
        # per each day: if N=10 and trading_days=21. it will be a list with 21 lists of len 10
        if T == 1:
            rand = functions.generate_normal_samples(N)  # Generate random values with normal distribution
            simulated_prices[1] = simulated_prices[0] * np.exp((mu - ITO * sigma**2) + sigma * rand * np.sqrt(DT))
        elif T > 1:
            for t in range(1, number_of_steps):
                rand = functions.generate_normal_samples(N)
                simulated_prices[t] = simulated_prices[t-1] * np.exp((mu - ITO * sigma**2) * DT + sigma * rand * np.sqrt(DT))

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
        print(f"VaR at 95% confidence after {T} days: -${VaR_95:.3f}")
        print(f"VaR % after {T} days at 95% confidence: {max_loss_95}%")
        print(f"Lowest expected portfolio value during a time period of {T} days: ${lowest_price}")
        print()
        print("--- Level of confidence 99% ---")
        print(f"VaR at 99% confidence after {T} days: -${VaR_99:.3f}")
        print(f"VaR % after {T} days at 99% confidence: {max_loss_99}%")
        print(f"Lowest expected portfolio value during a time period of {T} days: ${lowest_price}")
        print()
        print("--- Expected  Shortfall ---")
        print(f"ES or cVaR beyond the confidence interval 95%: -${exp_shortfall_95:.3f} (-{exp_shortfall_95_perc}%)")
        print(f"ES or cVaR beyond the confidence interval 99%: -${exp_shortfall_99:.3f} (-{exp_shortfall_99_perc}%)")

        if print_charts:
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

    elif int(user_input) > 1:
        # ask input for the tickers and the number of shares
        ask_tickers = input("Type the tickers separated by a comma like aapl,msft,cvx: ")
        tickers = [ticker.strip().upper() for ticker in ask_tickers.split(",")]

        ask_shares = input("For each ticker, type the number of shares separated by a comma: ")
        try:
            num_shares = [int(num.strip()) for num in ask_shares.split(",")]
        except Exception as e:
            print(e)
            sys.exit()

        # check that each ticker has a number of shares
        if len(num_shares) != len(tickers):
            print("The number of tickers is different from the shares you provided! Cannot proceed")
            sys.exit()

        # This variables will be used for the matrix
        num_simulations = int(input(f"How many simulations you want to run? Try at least 1000 "))  # Number of simulations
        T = int(input(f"How many trading days? The minimum is 1 "))  # trading days to run the simulations over, not the days downloaded from yfinance

        # Fetch data
        try:
            df = yf.download(tickers, start=START_DATE, end=END_DATE)['Close']
        except Exception as e:
            print(e)
            sys.exit()

        # Check that yfinance does not return an empty dataframe
        if df.empty:
            print(f"No data returned for this ticker list")
            sys.exit()

        # Drop NaN values (if any)
        df.dropna(inplace=True)

        # Calculate Log returns
        log_returns = np.log(df / df.shift(1)).dropna()

        # Compute mean and covariance of log returns
        # if you use numpy for the cov_matrix (np.cov(log_returns, rowvar=False) * 252)
        # the result is the same, with very tiny differences
        mean_returns = log_returns.mean() * 252  # annualized
        cov_matrix = log_returns.cov() * 252  # annualized

        # print("Mean Returns:\n", mean_returns)
        # print("Covariance Matrix:\n", cov_matrix)

        # Perform Cholesky decomposition
        L = cholesky(cov_matrix, lower=True)

        # Generate random normal numbers
        rand_normals = np.random.randn(T, len(tickers), num_simulations)

        # Apply Cholesky decomposition for correlated shocks
        correlated_shocks = np.einsum('ij,tjs->tis', L, rand_normals)

        # Create a tensor and set the values to zero
        simulated_prices = functions.init_tensor(T + 1, len(tickers), num_simulations)
        # Set the value of the first simulation to the last known price
        simulated_prices[0] = df.iloc[-1].values[:, np.newaxis]

        # Simulate price paths using Geometric Brownian Motion
        for t in range(1, T + 1):
            drift = (mean_returns.values[:, np.newaxis] - 0.5 * np.diag(cov_matrix)[:, np.newaxis]) * DT
            diffusion = correlated_shocks[t - 1] * np.sqrt(DT)
            simulated_prices[t] = simulated_prices[t - 1] * np.exp(drift + diffusion)

        # Example simulated paths: replace the second index in simulated_prices[:, 1, 0]
        # plt.plot(simulated_prices[:, 1, 0])
        # plt.title(f"Simulated paths for {tickers[1]} using GBM")
        # plt.xlabel("Days")
        # plt.ylabel("Price")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # for i in range(len(tickers)):
        #     plt.plot(simulated_prices[:, i, 0], label=f"Item {tickers[i]}")
        #
        # plt.title("Simulated paths using GBM")
        # plt.xlabel("Days")
        # plt.ylabel("Price")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # Compute portfolio values at the beginning and at the end
        initial_value = np.dot(df.iloc[-1].values, num_shares)
        final_values = np.dot(simulated_prices[-1].T, num_shares)
        print(simulated_prices)
        # Compute profit/loss distribution
        losses = initial_value - final_values

        # Compute 95% Value at Risk
        VaR_95 = np.percentile(losses, 5)
        VaR_95_perc = round((VaR_95 / initial_value) * 100, 3)

        # Compute 99% Value at Risk
        VaR_99 = np.percentile(losses, 1)
        VaR_99_perc = round((VaR_99 / initial_value) * 100, 3)

        print(VaR_95)
        print(f"Initial Portfolio Value: ${initial_value:.2f}")
        print()
        print(f"VaR after {T} days with confidence interval 95%: ${VaR_95:.2f} ({VaR_95_perc}%)")
        print(f"VaR after {T} days with confidence interval 99%: ${VaR_99:.2f} ({VaR_99_perc}%)")

        # Plot loss distribution
        if print_charts:
            plt.hist(losses, bins=50, edgecolor='black', alpha=0.7)
            plt.axvline(VaR_95, color='r', linestyle='dashed', linewidth=2, label=f"VaR 95%: ${VaR_95:.2f}")
            plt.title("Loss Distribution & VaR - Confidence Interval 95%")
            plt.xlabel("Loss ($)")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

    else:
        print("Not sure about the number of equities. Closing...")
        sys.exit()


if __name__ == '__main__':
    main()

