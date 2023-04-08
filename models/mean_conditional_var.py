import cvxpy as cp
from collections import OrderedDict
import numpy as np

def _objective_value(w, obj):
    """
    Helper function to return either the value of the objective function
    or the objective function as a cvxpy object depending on whether
    w is a cvxpy variable or np array.

    :param w: weights
    :type w: np.ndarray OR cp.Variable
    :param obj: objective function expression
    :type obj: cp.Expression
    :return: value of the objective function OR objective function expression
    :rtype: float OR cp.Expression
    """
    if isinstance(w, np.ndarray):
        if np.isscalar(obj):
            return obj
        elif np.isscalar(obj.value):
            return obj.value
        else:
            return obj.value.item()
    else:
        return obj

def add_constraint(new_constraint, _constraints, _w):
    """
    Add a new constraint to the optimization problem. This constraint must satisfy DCP rules,
    i.e be either a linear equality constraint or convex inequality constraint.
    """
    _constraints.append(new_constraint(_w))

def _make_weight_sum_constraint(is_market_neutral, _lower_bounds, _constraints, _w):
    """
    Helper method to make the weight sum constraint. If market neutral,
    validate the weights proided in the constructor.
    """
    if is_market_neutral:
        #  Check and fix bounds
        portfolio_possible = np.any(_lower_bounds < 0)
        _map_bounds_to_constraints((-1, 1))
        # Delete original constraints
        del _constraints[0]
        del _constraints[0]
        add_constraint(lambda w: cp.sum(w) == 0, _constraints, _w)
    else:
        add_constraint(lambda w: cp.sum(w) == 1, _constraints, _w)

def ind_weight_unitinterval_constraint(upper_limit, lower_limit, _constraints, _w):
    """
    Helper function to add constraint that the individual weights must be in (0,1)
    """
    add_constraint(lambda w: w >= lower_limit, _constraints, _w)
    add_constraint(lambda w: w <= upper_limit, _constraints, _w)


def _map_bounds_to_constraints(test_bounds, n_assets, _constraints, _w):
    """
    Convert input bounds into a form acceptable by cvxpy and add to the constraints list.

    :param test_bounds: minimum and maximum weight of each asset OR single min/max pair
                        if all identical OR pair of arrays corresponding to lower/upper bounds. defaults to (0, 1).
    :type test_bounds: tuple OR list/tuple of tuples OR pair of np arrays
    :raises TypeError: if ``test_bounds`` is not of the right type
    :return: bounds suitable for cvxpy
    :rtype: tuple pair of np.ndarray
    """
    # If it is a collection with the right length, assume they are all bounds.
    if len(test_bounds) == n_assets and not isinstance(
        test_bounds[0], (float, int)
    ):
        bounds = np.array(test_bounds, dtype=float)
        _lower_bounds = np.nan_to_num(bounds[:, 0], nan=-np.inf)
        _upper_bounds = np.nan_to_num(bounds[:, 1], nan=np.inf)
    else:
        lower, upper = test_bounds

        # Replace None values with the appropriate +/- 1
        if np.isscalar(lower) or lower is None:
            lower = -1 if lower is None else lower
            _lower_bounds = np.array([lower] * n_assets)
            upper = 1 if upper is None else upper
            _upper_bounds = np.array([upper] * n_assets)
        else:
            _lower_bounds = np.nan_to_num(lower, nan=-1)
            _upper_bounds = np.nan_to_num(upper, nan=1)

    add_constraint(lambda w: w >= _lower_bounds, _constraints, _w)
    add_constraint(lambda w: w <= _upper_bounds, _constraints, _w)

def portfolio_return(w, expected_returns, negative=True):
    """
    Calculate the (negative) mean return of a portfolio

    :param w: asset weights in the portfolio
    :type w: np.ndarray OR cp.Variable
    :param expected_returns: expected return of each asset
    :type expected_returns: np.ndarray
    :param negative: whether quantity should be made negative (so we can minimise)
    :type negative: boolean
    :return: negative mean return
    :rtype: float
    """
    sign = -1 if negative else 1
    mu = w @ expected_returns
    return _objective_value(w, sign * mu)

def _solve_cvxpy_opt_problem(_objective, _constraints, _solver, _verbose, _solver_options, _w, stock_list):
    """
    Helper method to solve the cvxpy problem and check output,
    once objectives and constraints have been defined

    :raises exceptions.OptimizationError: if problem is not solvable by cvxpy
    """
    _opt = cp.Problem(cp.Minimize(_objective), _constraints)
    _opt.solve(
        solver=_solver, verbose=_verbose, **_solver_options
    )
    weights = _w.value.round(16) + 0.0  # +0.0 removes signed zero
    output_weights = OrderedDict(zip(stock_list, weights))
    return output_weights

def min_cvar(_alpha, _beta, _u, upper_limit, lower_limit, returns, _constraints, _w, _solver, _verbose, _solver_options,
             stock_list, _lower_bounds, market_neutral=False):
    """
    Minimise portfolio CVaR

    :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                            defaults to False. Requires negative lower weight bound.
    :param market_neutral: bool, optional
    :return: asset weights for the volatility-minimising portfolio
    :rtype: OrderedDict
    """
    _objective = _alpha + 1.0 / (
        len(returns) * (1 - _beta)
    ) * cp.sum(_u)

    add_constraint(lambda _: _u >= 0.0, _constraints, _w)
    add_constraint(
        lambda w: returns.values @ w + _alpha + _u >= 0.0, _constraints, _w
    )
    ind_weight_unitinterval_constraint(upper_limit, lower_limit, _constraints, _w)
    _make_weight_sum_constraint(market_neutral, _lower_bounds, _constraints, _w)
    return _solve_cvxpy_opt_problem(_objective, _constraints, _solver, _verbose, _solver_options, _w, stock_list)

def portfolio_performance(returns, _alpha, _beta, _u):
    """
    After optimising, calculate (and optionally print) the performance of the optimal
    portfolio, specifically: expected return, CVaR

    :param verbose: whether performance should be printed, defaults to False
    :type verbose: bool, optional
    :return: CVaR.
    :rtype: (float)
    """

    cvar = _alpha + 1.0 / (len(returns) * (1 - _beta)) * cp.sum(
        _u
    )
    cvar_val = cvar.value

    return cvar_val