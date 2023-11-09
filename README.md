# Spatiotemporal Graph k Means
We introduce _Spatiotemporal Graph k-means_ (STGkM), a novel, unsupervised method to cluster vertices within a dynamic network. Drawing inspiration from traditional k-means, STGkM finds both short-term dynamic clusters and a ``long-lived'' partitioning of vertices within a network whose topology is evolving over time. We illuminate the algorithm's operation on synthetic data and apply it to detect political parties from a dynamic network of voting data in the United States House of Representatives.
One of the main advantages of STGkM is that it has only one required parameter, namely k; we therefore include an analysis of the range of this parameter and guidance on selecting its optimal value.

## Repository Description
This repository implements Spatiotemporal Graph k-Means (STGkM) and provides scripts that run EPCA on synthetic and real datasets.

## Files 
* stgkm/STGKM.py: Implementation of Spatiotemporal Graph k-Means (STGkM).
* stgkm/distance_functions.py: Implementation of s-journey, as described in the related paper.
* stgkm/graph_visualization.py: Code to visualize an evolving dynamic graph.
* tests/tests.py: Contains tests for STGkM.
* stgkm_figures.py: Contains functions for generating all visualizations from experiments. 

## Scripts
* roll_call_data.py: Script to get data directly from the House of Representatives website and form dataframes.
* roll_call_data_clustering.py: Script to run STGkM on roll call dataset.
* synthetic_three_cluster_script.py: Script to run STGkM on synthetic three cluster dataset.
* synthetic_two_cluster_script.py: Script to run STGkM on synthetic two cluster dataset.
  



