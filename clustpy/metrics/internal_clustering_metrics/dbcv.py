import numpy as np
from clustpy.metrics._metrics_utils import _check_length_data_and_labels
from clustpy.metrics._metrics_utils import handle_noise
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
from hdbscan._hdbscan_linkage import mst_linkage_core
from hdbscan.hdbscan_ import isclose
from hdbscan.validity import (
    distances_between_points,
    internal_minimum_spanning_tree,
    density_separation,
)


def dbcv_score(
    X,
    labels,
    metric="sqeuclidean",
    d=None,
    per_cluster_scores=False,
    mst_raw_dist=False,
    get_internal_mst=False,
    verbose=False,
    **kwd_args,
):
    """
    Compute the density based cluster validity index for the
    clustering specified by `labels` and for each cluster in `labels`.

    Parameters
    ----------
    X : array (n_samples, n_features) or (n_samples, n_samples)
        The input data of the clustering. This can be the data, or, if
        metric is set to `precomputed` the pairwise distance matrix used
        for the clustering.

    labels : array (n_samples)
        The label array output by the clustering, providing an integral
        cluster label to each data point, with -1 for noise points.

    metric : optional, string (default 'sqeuclidean')
        The metric used to compute distances for the clustering (and
        to be re-used in computing distances for mr distance). If
        set to `precomputed` then X is assumed to be the precomputed
        distance matrix between samples.

    d : optional, integer (or None) (default None)
        The number of features (dimension) of the dataset. This need only
        be set in the case of metric being set to `precomputed`, where
        the ambient dimension of the data is unknown to the function.

    per_cluster_scores : optional, boolean (default False)
        Whether to return the validity index for individual clusters.
        Defaults to False with the function returning a single float
        value for the whole clustering.

    mst_raw_dist : optional, boolean (default False)
        If True, the MST's are constructed solely via 'raw' distances (depending on the given metric, e.g. sqeuclidean distances)
        instead of using mutual reachability distances. Thus setting this parameter to True avoids using 'all-points-core-distances' at all.
        This is advantageous specifically in the case of elongated clusters that lie in close proximity to each other <citation needed>.

    get_internal_mst : optional, boolean (default False)
        Whether to also return the mst_nodes, mst_edges.
        Defaults to False with the function returning a single float
        value for the whole clustering. Mutually exclusive with <per_cluster_scores>.

    **kwd_args :
        Extra arguments to pass to the distance computation for other
        metrics, such as minkowski, Mahanalobis etc.

    Returns
    -------
    validity_index : float
        The density based cluster validity index for the clustering. This
        is a numeric value between -1 and 1, with higher values indicating
        a 'better' clustering.

    per_cluster_validity_index : array (n_clusters,)
        The cluster validity index of each individual cluster as an array.
        The overall validity index is the weighted average of these values.
        Only returned if per_cluster_scores is set to True.


    Source
    ------
    Original Code from: https://github.com/scikit-learn-contrib/hdbscan/blob/master/hdbscan/validity.py
    Their License: BSD-3-Clause license (https://github.com/scikit-learn-contrib/hdbscan/blob/master/LICENSE)
    Our modifications:
        - Add fix to also handle labelings that are not continues and/or start at zero


    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    Link: https://epubs.siam.org/doi/abs/10.1137/1.9781611973440.96
    """
    X, labels = _check_length_data_and_labels(X, labels)
    assert isinstance(labels, np.ndarray), "labels must be of type np.ndarray. Your input has type {0}".format(
        type(labels)
    )

    mask = labels != -1
    le = LabelEncoder()
    labels[mask] = le.fit_transform(labels[mask])

    labels, X = handle_noise(labels, strategy=noise_strategy, X=X)

    core_distances = {}
    density_sparseness = {}
    mst_nodes = {}
    mst_edges = {}

    max_cluster_id = len(set(labels))
    density_sep = np.inf * np.ones((max_cluster_id, max_cluster_id), dtype=np.float64)
    cluster_validity_indices = np.empty(max_cluster_id, dtype=np.float64)

    for cluster_id in range(max_cluster_id):

        if np.sum(labels == cluster_id) == 0:
            continue

        distances_for_mst, core_distances[cluster_id] = distances_between_points(
            X,
            labels,
            cluster_id,
            metric,
            d,
            no_coredist=mst_raw_dist,
            print_max_raw_to_coredist_ratio=verbose,
            **kwd_args,
        )

        mst_nodes[cluster_id], mst_edges[cluster_id] = internal_minimum_spanning_tree(
            distances_for_mst.astype(np.double)
        )
        density_sparseness[cluster_id] = mst_edges[cluster_id].T[2].max()

    for i in range(max_cluster_id):

        if np.sum(labels == i) == 0:
            continue

        internal_nodes_i = mst_nodes[i]
        for j in range(i + 1, max_cluster_id):

            if np.sum(labels == j) == 0:
                continue

            internal_nodes_j = mst_nodes[j]
            density_sep[i, j] = density_separation(
                X,
                labels,
                i,
                j,
                internal_nodes_i,
                internal_nodes_j,
                core_distances[i],
                core_distances[j],
                metric=metric,
                no_coredist=mst_raw_dist,
                **kwd_args,
            )
            density_sep[j, i] = density_sep[i, j]

    n_samples = float(X.shape[0])
    result = 0

    for i in range(max_cluster_id):

        if np.sum(labels == i) == 0:
            continue

        min_density_sep = density_sep[i].min()
        # print(min_density_sep, density_sparseness[i], min_density_sep, density_sparseness[i])
        cluster_validity_indices[i] = (min_density_sep - density_sparseness[i]) / max(
            min_density_sep, density_sparseness[i]
        )

        if verbose:
            print("Minimum density separation: " + str(min_density_sep))
            print("Density sparseness: " + str(density_sparseness[i]))

        cluster_size = np.sum(labels == i)
        result += (cluster_size / n_samples) * cluster_validity_indices[i]

    if per_cluster_scores:
        return result, cluster_validity_indices
    elif get_internal_mst:
        return result, mst_nodes, mst_edges
    else:
        return result
