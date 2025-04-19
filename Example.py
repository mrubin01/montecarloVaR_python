
"""
EXAMPLE STEP BY STEP WITH A PORTFOLIO CONTAINING ONE EQUITY
A portfolio with only one equity does not need the Cholesky decomposition

1 Download close price for a specific time range (let's say 1 year): the last
  price in this range will be the first for the simulations

2 Calculate daily returns and from those the log daily returns, mean mu and std sigma

3 Variables:
    T as the number of days to run the simulations,
    S as the number of shares, 
    N the simulations, 
    S0 as the last close price * S (the portfolio at the beginning of the simulation),
    dt as 1 / 252 for daily simulation, 1 / 52 if weekly (time step),
    number_of_steps equals to T as int(T / dt),
    confidence as 5 (95%) or 1 (99%)

4 Create a matrix with size 
    2 x N if T == 1
    T x N if T > 1
  The matrix is a list of list, matrix[0] will contain S0 for each simulation.
  Ex: if T == 21 and N == 100, the matrix is 21x100

5 Generate random prices based on normal distribution and mu/sigma/ITO and fill the matrix
  from matrix[1] on  

6 Compute the percentile out of the last day (last simulation in the matrix): this will 
  be the worst case after running all the simulations. From this value compute the VaR

7 Additionally compute the lowest price out of the whole matrix (any day, not only the last day)

8 Compute Expected Shortfall (ES): it measures the average loss in the worst-case scenarios (beyond 
  the confidence threshold)

"""
