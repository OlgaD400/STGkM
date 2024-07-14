# Spatiotemporal Graph k Means
We introduce _Spatiotemporal Graph k-means_ (STGkM), a novel, unsupervised method to cluster vertices within a dynamic network. Drawing inspiration from traditional k-means, STGkM finds both short-term dynamic clusters and a ``long-lived'' partitioning of vertices within a network whose topology is evolving over time. We illuminate the algorithm's operation on synthetic data and apply it to detect political parties from a dynamic network of voting data in the United States House of Representatives.
One of the main advantages of STGkM is that it has only one required parameter, namely k; we therefore include an analysis of the range of this parameter and guidance on selecting its optimal value.

## Repository Description
This repository implements Spatiotemporal Graph k-Means (STGkM) and provides scripts that run EPCA on synthetic and real datasets.

In order to run the code, you must first fork the following repository: https://github.com/OlgaD400/python-kmedoids. You will also have to have Cargo (rust programming language) installed. Then, run the following to compile the k-medoids package from source. 

```# activate your desired virtual environment first
pip install maturin
git clone https://github.com/kno10/python-kmedoids.git
cd python-kmedoids
# build and install the package:
maturin develop --release
```


## Files 
* stgkm/STGKM.py: Implementation of Spatiotemporal Graph k-Means (STGkM).
* stgkm/distance_functions.py: Implementation of s-journey, as described in the related paper.
* stgkm/graph_visualization.py: Code to visualize an evolving dynamic graph.
* stgkm/helper_functions.py: Helpful functions for running STGkM.
* stgkm/synthetic_graphs.py: Contains classes for all syntehtic graphs.
* tests/tests.py: Contains tests for STGkM.
* stgkm_figures.py: Contains functions for generating all visualizations from experiments.

## Scripts
* clique_cross_clique_script.py: Script for clique-cross-clique experiments.
* compare_performance_script.py: Script for comparing the performance of different vertex clustering methods across various synthetic datasets.
* reddit_script.py: Script for running STGkM on reddit data.
* roll_call_data_creation_cript.py: Script to get data directly from the House of Representatives website and form dataframes.
* roll_call_data_clustering_script.py: Script to run STGkM on roll call dataset.
* semantic_scholar_script.py: Script to run STGkM on semantic scholar data.
* synthetic_three_cluster_script.py: Script to run STGkM on synthetic three cluster dataset.
* synthetic_two_cluster_script.py: Script to run STGkM on synthetic two cluster dataset.
* theseus_clique_script.py: Script to run STGkM on theseus clique.
  



