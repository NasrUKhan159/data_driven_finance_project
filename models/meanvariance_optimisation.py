from scipy import optimize
import numpy as np
from collections import OrderedDict

def MaximizeReturns(MeanReturns, PortfolioSize):
    """
    Function to obtain maximal return portfolio using linear programming
    """
    c = (np.multiply(-1, MeanReturns))
    A = np.ones([PortfolioSize, 1]).T
    b = [1]
    res = optimize.linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method='simplex')

    return res

def MinimizeRisk(CovarReturns, PortfolioSize):
    """
    Function to obtain minimal risk portfolio
    """

    def f(x, CovarReturns):
        func = np.matmul(np.matmul(x, CovarReturns), x.T)
        return func

    def constraintEq(x):
        A = np.ones(x.shape)
        b = 1
        constraintVal = np.matmul(A, x.T) - b
        return constraintVal

    xinit = np.repeat(0.1, PortfolioSize)
    cons = ({'type': 'eq', 'fun': constraintEq})
    lb = 0
    ub = 1
    bnds = tuple([(lb, ub) for x in xinit])

    opt = optimize.minimize(f, x0=xinit, args=(CovarReturns), bounds=bnds, \
                            constraints=cons, tol=10 ** -3)

    return opt


def MinimizeRiskConstr(MeanReturns, CovarReturns, PortfolioSize, R):
    """
    Function to obtain minimal risk and maximum return portfolios
    """

    def f(x, CovarReturns):
        func = np.matmul(np.matmul(x, CovarReturns), x.T)
        return func

    def constraintEq(x):
        AEq = np.ones(x.shape)
        bEq = 1
        EqconstraintVal = np.matmul(AEq, x.T) - bEq
        return EqconstraintVal

    def constraintIneq(x, MeanReturns, R):
        AIneq = np.array(MeanReturns)
        bIneq = R
        IneqconstraintVal = np.matmul(AIneq, x.T) - bIneq
        return IneqconstraintVal

    xinit = np.repeat(0.1, PortfolioSize)
    cons = ({'type': 'eq', 'fun': constraintEq},
            {'type': 'ineq', 'fun': constraintIneq, 'args': (MeanReturns, R)})
    lb = 0
    ub = 1
    bnds = tuple([(lb, ub) for x in xinit])

    opt = optimize.minimize(f, args=(CovarReturns), method='trust-constr', \
                            x0=xinit, bounds=bnds, constraints=cons, tol=10 ** -3)

    return opt

def find_optimal_weights_meanvariance(meanReturns, covMatrix, stock_list):
    result1 = MaximizeReturns(meanReturns, len(stock_list))
    maxReturnWeights = result1.x
    maxExpPortfolioReturn = np.matmul(meanReturns.T, maxReturnWeights)

    result2 = MinimizeRisk(covMatrix, len(stock_list))
    minRiskWeights = result2.x
    minRiskExpPortfolioReturn = np.matmul(meanReturns.T, minRiskWeights)

    if minRiskExpPortfolioReturn < maxExpPortfolioReturn:
        opt = MinimizeRiskConstr(meanReturns, covMatrix, len(stock_list), minRiskExpPortfolioReturn).x
    opt_dict = OrderedDict(zip(stock_list, opt))
    return opt_dict