"""Synthetic Two Cluster Graph Script"""
import networkx as nx
import numpy as np
from stgkm.distance_functions import s_journey
from stgkm.STGKM import STGKM
from stgkm.graph_visualization import visualize_graph
from stgkm_figures import three_snapshots_dynamic_clustering

two_cluster_connectivity_matrix = np.array(
    [
        [
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ],
        [
            [0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1],
        ],
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ],
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 0],
        ],
    ]
)

### Visualize the connectivity matrix ###
g = nx.Graph(two_cluster_connectivity_matrix[0])
pos = nx.spring_layout(g)
visualize_graph(connectivity_matrix= two_cluster_connectivity_matrix)

### Calculate the s-journey ###
distance_matrix = s_journey(two_cluster_connectivity_matrix)

### Run STGkM ###
stgkm = STGKM(distance_matrix = distance_matrix, penalty = 5, max_drift = 1,
              drift_time_window = 1, num_clusters = 2, tie_breaker=False,
              iterations = 100)
stgkm.run_stgkm(method = 'full')

### Visualize three snapshots of the dynamic graph ###
fig = three_snapshots_dynamic_clustering(timesteps = [0,1,2],
                                   membership = stgkm.full_assignments,
                                   num_clusters = 2,
                                   connectivity_matrix= two_cluster_connectivity_matrix,
                                   centers = stgkm.full_centers,
                                   fig_title = 'Cluster Evolution',
                                   snapshot_title = 'Timestep ',
                                   filepath = 'temp')
