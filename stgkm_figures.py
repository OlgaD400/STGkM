"""Generate figures for STGKM experiments."""
import pickle
from typing import List, Optional
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from stgkm.STGKM import STGKM, similarity_matrix, similarity_measure


def choosing_num_clusters(min_clusters: int,
                        max_clusters: int,
                        distance_matrix: np.ndarray,
                        penalty: int,
                        max_drift: int,
                        drift_time_window: np.ndarray,
                        tie_breaker: bool,
                        max_iterations: int):
    """
    Function for choosing the optimal number of clusters.

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
        tie_breaker (bool): Whether to force unique vertex assignment.
        max_iterations (int): Maximum number of iterations for each run of stgkm.
    Returns:
        sum_distance_from_centers (List): List containing total sum distance of points from
            their cluster centers for each value of k.
    """
    sum_distance_from_centers = []
    times, _, _ = distance_matrix.shape
    for num_clusters in range(min_clusters, max_clusters):
        stgkm = STGKM(distance_matrix = distance_matrix,
                      penalty = penalty,
                      max_drift = max_drift,
                      drift_time_window = drift_time_window,
                      num_clusters = num_clusters,
                      tie_breaker=tie_breaker,
                      iterations = max_iterations)
        penalized_distance = stgkm.penalize_distance()
        stgkm.run_stgkm(method = 'proxy')

        total_cluster_sum = 0

        for time in range(times):
            centers = stgkm.full_centers[time].astype(int)
            intra_cluster_distances = np.sum(
                np.where(
                    stgkm.full_assignments[time*num_clusters:time*num_clusters + num_clusters] == 1,
                    penalized_distance[time, centers, :], 0))

            total_cluster_sum += intra_cluster_distances

        sum_distance_from_centers.append(total_cluster_sum)

    return sum_distance_from_centers

def choosing_num_clusters_plot(min_num_clusters: int,
                               max_num_clusters: int,
                               sum_distance_from_centers: List[float],
                               filepath: str):
    """
    Create figure tracking objective value vs number of clusters.

    Args:
        min_num_clusters (int): Minimum number of clusters
        max_num_clusters (int): Maximum number of clusters
        sum_distance_from_centers (List[float]): Sum of the vertex distance from centers
    """

    plt.figure(figsize = (10,5))
    plt.plot(range(min_num_clusters, max_num_clusters), sum_distance_from_centers)
    plt.xlabel('Number of Clusters k', size = 20)
    plt.tick_params(labelsize = 16)
    plt.ylabel('Objective Function Value', size = 20)
    plt.title('Objective Value vs. Number of Clusters', size = 20)

    plt.scatter(np.argmin(sum_distance_from_centers)+min_num_clusters,
                np.min(sum_distance_from_centers),
                marker = 'o',
                facecolors = 'none',
                edgecolors = 'r', s = 500,
                linewidths = 2,
                label = 'Optimal Objective Value')
    plt.legend(fontsize = 16)
    plt.savefig(filepath, format = 'pdf')

    return None

def three_snapshots_dynamic_clustering(connectivity_matrix: np.ndarray,
                           timesteps: List[int],
                           num_clusters: int,
                           membership: np.ndarray,
                           centers: np.ndarray,
                           fig_title: str,
                           snapshot_title: str,
                           filepath: str,
                           color_dict: Optional[dict] = None,
                           pkl_path: Optional[str] = None):
    """
    Show three snapshots of dynamic clusters in a dynamic graph.

    Args:
        connectivity_matrix (np.ndarray): Dynamic graph connectivity matrix
        timestpes (List[int]): Three timestpes to visualize
        membership (np.ndarray): Cluster membership history
        centers (np.ndarray): Cluster center history
        fig_title (str): Title for figure
        snapshot_title (str): Title for subfigure
        filepath (str): Filepath at which to save the figure
        color_dict (Optional[dict]): Color dictionary to use in the figures
        pos (Optional[str]) : Path to pkl file of saved positions for nodes in dynamic graph.
            If not provided, positions are generated from the random locations at time zero.
    """

    if color_dict is None:
        color_dict = {0: "dodgerblue", 1: "limegreen", 2: "red", 3: "green", -1: "cyan"}

    assert len(timesteps) ==3, "Can only visualize three time steps."

    _, _, num_points = connectivity_matrix.shape

    #Set layout for graph in visualization
    init_graph = nx.Graph(connectivity_matrix[0])
    init_graph.remove_edges_from(nx.selfloop_edges(init_graph))
    if pkl_path is None:
        pos = nx.spring_layout(init_graph)
    else:
        with open(pkl_path, 'rb') as f:
            pos = pickle.load(f)

    #Create figure
    fig, axs = plt.subplots(nrows =1, ncols = 3, figsize = (60,20))
    fig.suptitle(fig_title, fontsize = 100)

    for index, time in enumerate(timesteps):
        time_labels = np.argmax(membership[time*num_clusters: time*(num_clusters) + num_clusters],
                                axis = 0)
        center_size = np.ones(num_points) * 200
        center_size[centers[time].astype(int)] = 1000
        final_graph = nx.Graph(connectivity_matrix[time])
        final_graph.remove_edges_from(nx.selfloop_edges(final_graph))
        nx.draw(
                        final_graph,
                        with_labels=False,
                        node_size = center_size,
                        node_color = [color_dict[label] for label in time_labels],
                        pos= pos,
                        ax = axs[index%3]
                    )
        axs[index%3].set_title('%s %i' %(snapshot_title, timesteps[index]), size = 100)
    plt.savefig(filepath, format = 'pdf')
    plt.show()

    return fig

def similarity_matrix_figure(full_assignments: np.ndarray,
                      long_term_clusters: np.ndarray,
                      fig_title: str,
                      filepath: str):

    """Make similarity matrix figure."""

    sim_mat = similarity_matrix(
        weights=full_assignments.T,
        similarity_function=similarity_measure)

    communities = []
    for unique_label in np.unique(long_term_clusters):
        communities.append(np.where(long_term_clusters == unique_label)[0])

    sorted_communities = sorted(communities, key = lambda x: len(x), reverse = True)
    cols = []
    for community_1 in sorted_communities:
        col = []
        for community_2 in sorted_communities:
            col_entry = sim_mat[community_1,:][:, community_2]
            col.append(col_entry)
        cols.append(np.hstack(col))

    reordered_mat = np.vstack(cols)

    _, axs = plt.subplots(figsize = (20,20))
    axs.axis('off')
    plt.imshow(reordered_mat, cmap = 'viridis')
    cbar = plt.colorbar(pad = .01, ticks = np.arange(.2,1.1,.1))
    cbar.ax.tick_params(labelsize=30)
    plt.title(fig_title, size = 40, y = 1.05)
    plt.savefig(filepath, format = 'pdf')
    plt.show()

    return None
