"""Functions for visualizing an evolving graph at every time step."""
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(
    connectivity_matrix: np.ndarray,
    labels: Optional[List] = None,
    centers: Optional[List] = None,
    color_dict: Optional[dict] = None,
    figsize = (10,10)
):
    """
    Visualize the dynamic graph at each time step. 
    """
    timesteps, num_vertices, _ = connectivity_matrix.shape

    if labels is None:
        labels = []
    if centers is None:
        centers = []
    if color_dict is None:
        color_dict = {0: "red", 1: "gray", 2: "green", 3: "blue", -1: "cyan"}

    if len(np.unique(labels)) > len(color_dict):
        raise Exception("Color dictionary requires more than 4 keys/values")

    #Set layout for figures
    g_0 = nx.Graph(connectivity_matrix[0])
    g_0.remove_edges_from(nx.selfloop_edges(g_0))
    pos = nx.spring_layout(g_0)

    for time in range(timesteps):
        plt.figure(figsize = figsize)
        # No labels
        if len(labels) == 0:
            nx.draw(nx.Graph(connectivity_matrix[time]), pos=pos, with_labels=True)
        # Static long term labels
        elif len(labels) == num_vertices:
            graph = nx.Graph(connectivity_matrix[time])
            graph.remove_edges_from(nx.selfloop_edges(graph))
            nx.draw(
                graph,
                pos=pos,
                node_color=[color_dict[label] for label in labels],
                with_labels=True,
            )
        # Changing labels at each time step
        elif len(labels) == timesteps:
            if len(centers) != 0:
                center_size = np.ones(num_vertices) * 300
                center_size[centers[time].astype(int)] = 500
                graph = nx.Graph(connectivity_matrix[time])
                graph.remove_edges_from(nx.selfloop_edges(graph))
                nx.draw(
                    graph,
                    pos=pos,
                    node_color=[color_dict[label] for label in labels[time]],
                    node_size=center_size,
                    with_labels=True,
                )
            else:
                graph = nx.Graph(connectivity_matrix[time])
                graph.remove_edges_from(nx.selfloop_edges(graph))
                nx.draw(
                    graph,
                    pos=pos,
                    node_color=[color_dict[label] for label in labels[time]],
                    with_labels=True,
                )

        plt.show()