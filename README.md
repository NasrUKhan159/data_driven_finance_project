# A Comparison of Portfolio Optimisation Strategies for Portfolio Allocation

The strategy for this project is the following:

1. Data loading and processing: Load in the data of interes
The end aim of this is to produce a clean data-frame of closing
prices of stocks of interest.

2. Data modelling - models: Compute optimal portfolio weights for 
the following strategies:

    Method 1: Mean-variance optimisation
    Method 2: Hierarchical Risk Parity
    Method 3: Mean conditional Value at Risk 

3. Backtesting and evaluation metrics: Use different metrics to 
evaluate the quality of portfolio allocation, using the following:]

    Metric 1: Sharpe Ratio

### Method 1: Mean-Variance Portfolio Optimisation

This methodology finds the optimal set of weights in a portfolio, that obtains the maximum return of a portfolio for a 
minimal risk.
The Markowitz model built on the mean-variance framework of asset returns, called the Mean-Variance Optiimisation (MVO) 
model, solves a multi-objective optimisation problem subject to basic constraints imposed on portfolio.

**Solving the MVO model**

This problem can be solved by decomposing it into the following 3 sub-problems:

i. Obtain maximal expected return of portfolio subject to basic constraints.

ii. Obtain optimal expected return corresponding to minimum risk portfolio, subject to basic constraints.

iii. Obtain optimal weights of portfolio sets that minimize risk and whose returns lie between minimum risk return and 
maximal expected return subject to basic constraints.

The optimal portfolio set is called the efficient set.

### Method 2: Hierarchical Risk Parity (HRP)

HRP is a novel portfolio optimisation method. Here is a rough overview of how HRP works:

1. From a universe of assets, form a distance matrix based on correlation of assets.
2. Using the distance matrix, cluster assets into tree via hierarchical clustering.
3. Within each branch of the tree, form the minimum variance portfolio.
4. Iterate over each level, optimally combining mini-portfolios at each node.

### Method 3: Mean Conditional Value at Risk (mCVaR)

The mCVaR is a popular alternative to mean variance optimisation. It works by measuring the worst-case scenarios for 
each asset in the portfolio i.e. losing the most money. The worst-case loss for each asset is used to calculate weights 
to be used for allocation of each asset.

### Metric 1: Sharpe Ratio

The Sharpe Ratio is the difference between the return of the portfolio and the risk-free rate, divided by the standard
deviation of the portfolio. Usually, the risk-free rate is set to be 0.01. A Sharpe ratio of greater than 1 is 
considered to be good since it offers excess returns relative to volatility.

### References (for code and documentation): 

1: https://ml4trading.io/chapter/4

2: https://blog.quantinsti.com/portfolio-management-strategy-python/

3: https://builtin.com/data-science/portfolio-optimization-python

4: https://www.kaggle.com/code/vijipai/lesson-5-mean-variance-optimization-of-portfolios/notebook

5: https://algotrading101.com/learn/yfinance-guide/

6: https://pyquantnews.com/use-kelly-criterion-optimal-position-sizing/

7: https://hudsonthames.org/an-introduction-to-the-hierarchical-risk-parity-algorithm/

8: https://pyportfolioopt.readthedocs.io/en/latest/OtherOptimizers.html

9: https://pyportfolioopt.readthedocs.io/en/latest/_modules/index.html

10: https://pyportfolioopt.readthedocs.io/en/latest/_modules/pypfopt/efficient_frontier/efficient_cvar.html
