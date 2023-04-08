import numpy as np

def sharpe_ratio(method_weights, risk_free_rate, meanReturns, covMatrix):
    exp_returns = 0
    for index, stock in enumerate(method_weights.keys()):
        exp_returns += method_weights[stock]*meanReturns[index]

    method_weights_list = []
    for value in method_weights.values():
        method_weights_list.append(value)
    method_weights_list = np.array(method_weights_list)

    sd_method = method_weights_list.T @ covMatrix @ method_weights_list
    sharpe_ratio_results = (exp_returns - risk_free_rate) / (sd_method)
    return sharpe_ratio_results

