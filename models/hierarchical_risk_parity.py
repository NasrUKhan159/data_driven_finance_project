import collections
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch
from scipy.spatial import distance as ssd

def _get_cluster_var(covMatrix, cluster_items):
    """
    Compute the variance per cluster

    :param cov: covariance matrix
    :type cov: np.ndarray
    :param cluster_items: tickers in the cluster
    :type cluster_items: list
    :return: the variance per cluster
    :rtype: float
    """
    # Compute variance per cluster
    cov_slice = covMatrix.loc[cluster_items, cluster_items]
    weights = 1 / np.diag(cov_slice)  # Inverse variance weights
    weights /= weights.sum()
    return np.linalg.multi_dot((weights, cov_slice, weights))

def _get_quasi_diag(link):
    """
    Sort clustered items by distance

    :param link: linkage matrix after clustering
    :type link: np.ndarray
    :return: sorted list of indices
    :rtype: list
    """
    return sch.to_tree(link, rd=False).pre_order()

def _raw_hrp_allocation(covMatrix, ordered_tickers):
    """
    Given the clusters, compute the portfolio that minimises risk by
    recursively traversing the hierarchical tree from the top.

    :param cov: covariance matrix
    :type cov: np.ndarray
    :param ordered_tickers: list of tickers ordered by distance
    :type ordered_tickers: str list
    :return: raw portfolio weights
    :rtype: pd.Series
    """
    w = pd.Series(1, index=ordered_tickers)
    cluster_items = [ordered_tickers]  # initialize all items in one cluster

    while len(cluster_items) > 0:
        cluster_items = [
            i[j:k]
            for i in cluster_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]  # bi-section
        # For each pair, optimize locally.
        for i in range(0, len(cluster_items), 2):
            first_cluster = cluster_items[i]
            second_cluster = cluster_items[i + 1]
            # Form the inverse variance portfolio for this pair
            first_variance = _get_cluster_var(covMatrix, first_cluster)
            second_variance = _get_cluster_var(covMatrix, second_cluster)
            alpha = 1 - first_variance / (first_variance + second_variance)
            w[first_cluster] *= alpha  # weight 1
            w[second_cluster] *= 1 - alpha  # weight 2
    return w

def hrp_optimize(covMatrix, returns, linkage_method="single"):
    """
    Construct a hierarchical risk parity portfolio, using Scipy hierarchical clustering
    :param linkage_method: which scipy linkage method to use
    :type linkage_method: str
    :return: weights for the HRP portfolio
    :rtype: OrderedDict
    """
    if linkage_method not in sch._LINKAGE_METHODS:
        raise ValueError("linkage_method must be one recognised by scipy")
    corrMatrix = returns.corr()

    # Compute distance matrix, with ClusterWarning fix as
    # per https://stackoverflow.com/questions/18952587/

    # this can avoid some nasty floating point issues
    corr_dist_matrix = np.sqrt(np.clip((1.0 - corrMatrix) / 2.0, a_min=0.0, a_max=1.0))
    pairwise_dist_matrix = ssd.squareform(corr_dist_matrix, checks=False)

    clusters = sch.linkage(pairwise_dist_matrix, linkage_method)
    sort_ix = _get_quasi_diag(clusters)
    ordered_tickers = corrMatrix.index[sort_ix].tolist()
    hrp = _raw_hrp_allocation(covMatrix, ordered_tickers)
    weights = collections.OrderedDict(hrp.sort_index())
    return weights