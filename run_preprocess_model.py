import numpy as np
import cvxpy as cp
from data_processing.load_preprocess_data import import_stock_data, portfolio_mean_sd
from metrics.sharpe_ratio import sharpe_ratio
from models.hierarchical_risk_parity import hrp_optimize
from models.mean_conditional_var import min_cvar
from models.meanvariance_optimisation import find_optimal_weights_meanvariance

def run():
    stock_list = ['AAPL', 'AMZN', 'MSFT', 'GOOG']
    end_date = "2022-01-30"
    start_date = "2022-01-01"
    interval = "1h"
    # range of possible values for interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    returns, meanReturns, covMatrix = import_stock_data(stock_list, start_date, end_date, interval)
    returns = returns.dropna()
    weights = [0.25, 0.25, 0.25, 0.25]
    weights /= np.sum(weights)
    exp_returns, std = portfolio_mean_sd(weights, meanReturns, covMatrix)

    _w = cp.Variable(len(meanReturns))
    _beta = 0.95 #_beta can be in [0,1) - it is the confidence level where typical values are
    #0.8, 0.9, 0.95
    _alpha = cp.Variable()
    _u = cp.Variable(len(returns))
    _constraints = []
    _verbose = False
    _lower_bounds = None
    _solver = 'SCS' # name of solver - can get available solvers with cxvpy.installed_solvers()
    # will use SCS solver since it can solve the maximal amt of linear programming, quadratic programming, etc type problems
    _solver_options = {}
    # solver_options is parameters for the given solver, and is an optional dictionary
    weight_bounds = (0,1)
    upper_limit = 0.99
    lower_limit = 0.01
    risk_free_rate = 0.01

    optimal_meanvariance_weights = find_optimal_weights_meanvariance(meanReturns, covMatrix, stock_list)
    optimal_mcvar_weights = min_cvar(_alpha, _beta, _u, upper_limit, lower_limit, returns, _constraints, _w, _solver,
                                     _verbose, _solver_options, stock_list, _lower_bounds, market_neutral=False)
    optimal_hrp_weights = hrp_optimize(covMatrix, returns)

    sharpe_meanvariance = sharpe_ratio(optimal_meanvariance_weights, risk_free_rate, meanReturns, covMatrix)
    sharpe_hrp = sharpe_ratio(optimal_hrp_weights, risk_free_rate, meanReturns, covMatrix)
    sharpe_mcvar = sharpe_ratio(optimal_mcvar_weights, risk_free_rate, meanReturns, covMatrix)
    return optimal_meanvariance_weights, optimal_hrp_weights, optimal_mcvar_weights, sharpe_meanvariance, sharpe_hrp, \
           sharpe_mcvar

if __name__ == "__main__":
    optimal_meanvariance_weights, optimal_hrp_weights, optimal_mcvar_weights, sharpe_meanvariance, sharpe_hrp, \
    sharpe_mcvar = run()
    print(optimal_hrp_weights)
    print(optimal_meanvariance_weights)
    print(optimal_mcvar_weights)
    print("The Sharpe ratio using the mean-variance portfolio optimisation methodology is {}."
          .format(sharpe_meanvariance))
    print("The Sharpe ratio using the Hierarchical Risk Parity portfolio optimisation methodology is {}."
          .format(sharpe_hrp))
    print("The Sharpe ratio using the mean conditional VaR portfolio optimisation methodology is {}."
          .format(sharpe_mcvar))





