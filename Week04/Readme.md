This is for the Week05 project.\n


The p1_return_three_mean_sd.ipynb file is for problem 1. In this problem, we calculate and compare the expected value and standard deviation of the price at time t, given
each of the 3 types of price returns, assuming return value r is normally distributed. We simulate each return equation and show the mean and standard deviation match your expectations

The p2_arithmetic_return_VaR.ipynb file is for problem 2. In this file, we implement a function similar to the “return_calculate()” in this week’s code. Allow the user to
specify the method of return calculation. And we use DailyPrices.csv. Calculate the arithmetic returns for all prices. Next, we remove the mean from the series so that the mean(META)=0
Calculate VaR using the following 5 distributions:
1. Using a normal distribution.
2. Using a normal distribution with an Exponentially Weighted variance (λ = 0. 94)
3. Using a MLE-fitted T distribution.
4. Using a fitted AR(1) model.
5. Using a Historic Simulation.
Finally, we compare the 5 values.

The p3_exponential_weighted_model_VaR.ipynb file is for problem 3. In this problem, we use Portfolio.csv and DailyPrices.csv. Assume the expected return on all stocks is 0. This file contains the stock holdings of 3 portfolios. You own each of these portfolios. Using an
exponentially weighted covariance with lambda = 0.94, calculate the VaR of each portfolio as well as the total VaR (VaR of the total holdings). And we choose a different model for returns and calculate VaR again. 
