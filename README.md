## Montecarlo VaR
This program calculates the Value at Risk (VaR) for one or more equities using the Monte Carlo simulations.
Moreover, it calculates the Expected Shortfall (ES) and the maximum drawdown.
If the portfolio is composed of just one ticker, the Cholesky decomposition is not necessary

## Steps
1. Fetch data from Yahoo Finance
2. Calculate daily returns and from those the log daily returns, mean (mu) and std (sigma)
3. In case of one ticker, create an empty matrix and fill it with random prices having a normal distribution
   In case of more tickers, create an empty tensor and fill it with random prices having a normal distribution
4. In both cases, the random prices are generated using the Geometric Brownian Motion
5. Compute the Value at Risk using the last simulation in the matrix/tensor with confidence interval 95% and 99%
6. Compute the lowest value of the portfolio during the simulations
7. Compute the Expected Shortfall, that is the the average loss in the worst-case scenarios (beyond
   the confidence threshold)
