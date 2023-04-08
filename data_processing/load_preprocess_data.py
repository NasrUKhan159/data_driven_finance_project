import numpy as np
import yfinance as yf
from typing import List

def import_stock_data(stock_list: List, start_date: str, end_date: str, interval: str):
    """
    Import the stock data from Yahoo Finance API, aimed to be used for research and educational purposes
    """
    stock_data_close = yf.Tickers(stock_list).history(start=start_date, end=end_date, interval=interval)['Close']
    returns = stock_data_close.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return returns, mean_returns, cov_matrix

def portfolio_mean_sd(weights, meanReturns, covMatrix):
    """
    For an inputted set of weights, compute the expected return and the standard deviation of the portfolio.
    """
    exp_returns = np.sum(meanReturns*weights)
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))
    return exp_returns, std