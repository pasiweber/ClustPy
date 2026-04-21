# Implementation of VIASCKDE by
# - Author: Ali Şenol - Github user `senoali`
# - Source: https://github.com/senolali/VIASCKDE/blob/main/VIASCKDE.py
# - License: GPL-3.0 licence (https://github.com/senolali/VIASCKDE/blob/main/LICENSE)

# Paper: VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary-Shaped Clusters Based on the Kernel Density Estimation
# Link: https://doi.org/10.1155/2022/4059302
# Authors: Ali Şenol


from scipy.spatial import KDTree
from sklearn.neighbors import KernelDensity
import numpy as np


def _closest_node(point: np.ndarray, candidates: np.ndarray) -> float:
    """
    Compute the distance from a point to its nearest neighbor in a candidate set.

    Parameters
    ----------
    point : np.ndarray
        The query point.
    candidates : np.ndarray
        Candidate points used to construct the KDTree.

    Returns
    -------
    distance : float
        Distance to the nearest candidate point.
    """
    kdtree = KDTree(candidates)
    distance, _ = kdtree.query(point)
    return distance


def viasckde_score(X: np.ndarray, labels: np.ndarray, krnl: str = "gaussian", b_width: float = 0.05) -> float:
    """
    Evaluate the quality of predicted labels using the VIASCKDE cluster validity index.

    VIASCKDE (Validity Index for Arbitrary-Shaped Clusters based on Kernel Density
    Estimation) combines density-aware local separation and compactness information.
    It is specifically designed for arbitrary-shaped clusters and incorporates kernel
    density estimation to weight samples according to local density.

    Higher values indicate better clustering quality.

    Parameters
    ----------
    X : np.ndarray
        The data set of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels predicted by a clustering algorithm.
    krnl : str
        Kernel type used for density estimation in sklearn.neighbors.KernelDensity
        (default: "gaussian").
    b_width : float
        Bandwidth parameter used for kernel density estimation
        (default: 0.05).

    Returns
    -------
    viasckde : float
        The VIASCKDE score. Higher values indicate better clustering structure.
        Returns np.nan if only one cluster is present.

    References
    ----------
    Şenol, Ali.
    "VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary-Shaped
    Clusters Based on the Kernel Density Estimation."
    Computational Intelligence and Neuroscience. 2022.

    Link: https://doi.org/10.1155/2022/4059302
    Source: https://github.com/senolali/VIASCKDE/blob/main/VIASCKDE.py
    License: GPL-3.0 licence (https://github.com/senolali/VIASCKDE/blob/main/LICENSE)
    """
    CoSeD = np.array([], [])
    num_k = np.unique(labels)
    kde = KernelDensity(kernel=krnl, bandwidth=b_width).fit(X)
    iso = kde.score_samples(X)

    ASC = np.array([])
    numC = np.array([])
    CoSeD = np.array([])
    viasc = 0
    if len(num_k) > 1:
        for i in num_k:
            data_of_cluster = X[labels == i]
            data_of_not_its = X[labels != i]
            isos = iso[labels == i]
            isos = (isos - min(isos)) / (max(isos) - min(isos))
            for j in range(len(data_of_cluster)):  # for each data of cluster j
                row = np.delete(data_of_cluster, j, 0)  # exclude the data j
                XX = data_of_cluster[j]
                a = _closest_node(XX, row)
                b = _closest_node(XX, data_of_not_its)
                ASC = np.hstack((ASC, ((b - a) / max(a, b)) * isos[j]))
            numC = np.hstack((numC, ASC.size))
            CoSeD = np.hstack((CoSeD, ASC.mean()))
        for k in range(len(numC)):
            viasc += numC[k] * CoSeD[k]
        viasc = viasc / sum(numC)
        # print("viasc=%0.4f" % viasc)
    else:
        viasc = float("nan")
    return viasc
