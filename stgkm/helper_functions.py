"""Helper functions for running STGkM."""

import numpy as np
import kmedoids
from stgkm.STGKM import STGKM
from stgkm.distance_functions import s_journey
from stgkm.STGKM import agglomerative_clustering


def penalize_distance(distance_matrix: np.ndarray, penalty: float) -> np.ndarray:
    """
    Penalize the distance matrix, to make infinite journies finite.

    Inputs:
        distance_matrix (np.ndarray): Array storing distances between pairs of vertices.
        penalty (float): Finite penalty assigned to vertices with infinite distance.
    Returns:
        (np.ndarray) Penalized distance matrix.
    """
    penalized_distance = np.where(distance_matrix == np.inf, penalty, distance_matrix)
    return penalized_distance


def calculate_cluster_connectivity(distance_matrix: np.ndarray, membership: np.ndarray):
    """
    Calculate intra vs inter cluster connectivity.

    Args:
        distance_matrix (np.ndarray): Array storing distances between pairs of vertices.
        membership (np.ndarray): Array storing cluster membership assignments.
    """
    timesteps, num_vertices, _ = distance_matrix.shape
    unique_clusters = np.unique(membership)
    num_clusters = len(unique_clusters)

    intra_cluster_similarities = np.zeros((num_clusters, timesteps))
    inter_cluster_similarities = np.zeros((num_clusters, timesteps))

    for index, cluster in enumerate(unique_clusters):
        for time in range(timesteps):
            member_ind = np.where(membership[time] == cluster)[0]
            non_member_ind = np.setdiff1d(np.arange(num_vertices), member_ind)
            membership_submatrix = distance_matrix[time, member_ind][:, member_ind]
            non_membership_submatrix = distance_matrix[time, member_ind][
                :, non_member_ind
            ]
            intra_connections = np.sum(membership_submatrix) / 2
            inter_connections = np.sum(non_membership_submatrix) / 2

            intra_cluster_similarities[index, time] = intra_connections
            inter_cluster_similarities[index, time] = inter_connections

    return intra_cluster_similarities, inter_cluster_similarities


def run_stgkm(
    connectivity_matrix,
    penalty,
    num_clusters,
    max_drift,
    drift_time_window,
    max_iter,
    random_state,
):
    """Run STGkM."""
    distance_matrix = s_journey(connectivity_matrix)
    penalized_distance = np.where(distance_matrix == np.inf, penalty, distance_matrix)

    km = kmedoids.KMedoids(
        num_clusters,
        method="fasterpam_time",
        max_drift=max_drift,
        drift_time_window=drift_time_window,
        max_iter=max_iter,
        random_state=random_state,
        online=False,
    )
    c = km.fit(penalized_distance)
    opt_labels = c.labels_
    opt_ltc = agglomerative_clustering(weights=opt_labels.T, num_clusters=num_clusters)
    return c, opt_labels, opt_ltc


def return_avg_cluster_connectivity(
    intra_cluster_similarities: np.ndarray, inter_cluster_similarities: np.ndarray
):
    return np.average(intra_cluster_similarities, axis=1), np.average(
        inter_cluster_similarities, axis=1
    )


def choose_num_clusters_og(
    min_clusters: int,
    max_clusters: int,
    distance_matrix: np.ndarray,
    max_drift: int,
    drift_time_window: np.ndarray,
    max_iterations: int,
    penalty: int,
    method: str,
    tie_breaker=False,
):
    """
    Function for choosing the optimal number of clusters based on original STGkM implementation.

    Must be run for a "reasonable" number of clusters or minimum will
    be when all points are their own cluster centers.

    Args:
        min_clusters (int): Minimum number of clusters to test
        max_clusters (int): Maximum number of clusters to test
        distance_matrix (np.ndarray): Distance between all pairs of vertices
        penalty (float): Penalty to assign to disconnected vertices during pre-processing.
        max_drift (int): Maximum distance between cluster centers over time.
        drift_time_window (int): Number of timesteps centers must remain within max_drift
            of one another.
        max_iterations (int): Maximum number of iterations for each run of stgkm.
    Returns:
        sum_distance_from_centers (List): List containing total sum distance of points from
            their cluster centers for each value of k.
    """
    obj_values = []
    label_history = []
    medoid_history = []

    k_range = range(min_clusters, max_clusters)

    for num_clusters in k_range:
        stgkm = STGKM(
            distance_matrix=distance_matrix,
            penalty=penalty,
            max_drift=max_drift,
            drift_time_window=drift_time_window,
            num_clusters=num_clusters,
            tie_breaker=tie_breaker,
            iterations=max_iterations,
        )
        stgkm.run_stgkm(method=method)

        penalized_distance = stgkm.penalize_distance()

        obj_value = calculate_objective(
            stgkm.full_centers,
            labels=stgkm.full_assignments,
            penalized_distance=penalized_distance,
        )

        obj_values.append(obj_value)
        label_history.append(stgkm.full_assignments)
        medoid_history.append(stgkm.full_centers)

    return obj_values, k_range[np.argmin(obj_values)], label_history, medoid_history


def choose_num_clusters(
    min_clusters: int,
    max_clusters: int,
    connectivity_matrix: np.ndarray,
    penalized_distance: np.ndarray,
    max_drift: int,
    drift_time_window: np.ndarray,
    max_iterations: int,
    random_state,
    medoid_selection: str = "connectivity",
):
    """
    Function for choosing the optimal number of clusters using STGkM based on k-medoids FasterPAM implementation.

    Must be run for a "reasonable" number of clusters or minimum will
    be when all points are their own cluster centers.

    Args:
        min_clusters (int): Minimum number of clusters to test
        max_clusters (int): Maximum number of clusters to test
        distance_matrix (np.ndarray): Distance between all pairs of vertices
        penalty (float): Penalty to assign to disconnected vertices during pre-processing.
        max_drift (int): Maximum distance between cluster centers over time.
        drift_time_window (int): Number of timesteps centers must remain within max_drift
            of one another.
        max_iterations (int): Maximum number of iterations for each run of stgkm.
    Returns:
        sum_distance_from_centers (List): List containing total sum distance of points from
            their cluster centers for each value of k.
    """
    obj_values = []
    label_history = []
    medoid_history = []

    k_range = range(min_clusters, max_clusters)

    for num_clusters in k_range:
        if medoid_selection == "connectivity":
            # medoids = np.argsort(np.sum(penalized_distance[0], axis=1))[:num_clusters]
            medoids = np.argsort(np.sum(connectivity_matrix[0], axis=1))[::-1][
                :num_clusters
            ]
        elif medoid_selection == "num":
            medoids = num_clusters

        stgkm = kmedoids.KMedoids(
            medoids,
            method="fasterpam_time",
            max_drift=max_drift,
            drift_time_window=drift_time_window,
            max_iter=max_iterations,
            random_state=random_state,
        )
        result = stgkm.fit(penalized_distance)

        obj_value = calculate_silhouette(
            medoids=result.medoid_indices_, penalized_distance=penalized_distance
        )

        obj_values.append(obj_value)
        label_history.append(result.labels_)
        medoid_history.append(result.medoid_indices_)

    return obj_values, k_range[np.argmax(obj_values)], label_history, medoid_history


def calculate_objective(
    medoids: np.ndarray, labels: np.ndarray, penalized_distance: np.ndarray
):
    """
    Calculate the final objective value of the output of STGkM.

    inputs:
        medoids (np.ndarray): txk medoids predicted for every time step
        labels (np.ndarray): txn labelels predicted for every vertex at every time step
        penalized_distance (np.ndarray): txnxn matrix containing the penalized s_journey distance
            between every pair of nodes
    """

    timesteps, _, _ = penalized_distance.shape
    timesteps, k = medoids.shape

    obj_sum = 0
    for time in range(timesteps):
        p_slice = penalized_distance[time]
        medoid_slice = medoids[time]
        for cluster in range(k):
            members = np.where(labels[time] == cluster)[0]
            obj_sum += np.sum(p_slice[medoid_slice[cluster], members])
    return obj_sum


def calculate_silhouette(medoids: np.ndarray, penalized_distance: np.ndarray):
    """
    Calculate the average silhouette score for a given set of chosen medoids."""

    timesteps, _, _ = penalized_distance.shape

    s_scores = []
    for time in range(timesteps):
        p_slice = penalized_distance[time]
        medoid_slice = medoids[time]

        s_score, _ = kmedoids.medoid_silhouette(p_slice, medoid_slice)
        s_scores.append(s_score)
    return np.average(s_scores)
