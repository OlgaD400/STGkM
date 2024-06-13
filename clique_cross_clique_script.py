"""Functions and script for running clique-cross-clique experiments."""

import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score
import pickle
from stgkm.helper_functions import run_stgkm
from stgkm.synthetic_graphs import CliqueCrossClique, RandomCliqueCrossClique
from stgkm_figures import plot_expectation_heatmap, plot_sensitivity_figure

NUM_TIMESTEPS = 10
NUM_CLUSTERS = 3
NUM_MEMBERS = 10
P_INTRA = 0.3
P_INTER = 0.2

CXC1 = CliqueCrossClique(
    num_clusters=NUM_CLUSTERS,
    num_members=NUM_MEMBERS,
    num_timesteps=NUM_TIMESTEPS,
)

CXC2 = RandomCliqueCrossClique(
    num_clusters=NUM_CLUSTERS,
    num_members=NUM_MEMBERS,
    num_timesteps=NUM_TIMESTEPS,
    p_intra=P_INTRA,
    p_inter=P_INTER,
)

CXC1.create_clique_cross_clique()
CXC2.create_clique_cross_clique()

FILEPATH_1 = "STGKM_Figures/clique_cross_clique.pdf"
FILEPATH_2 = "STGKM_Figures/random_clique_cross_clique.pdf"
CXC1.plot_clique_cross_clique(filepath=FILEPATH_1)
CXC2.plot_clique_cross_clique(filepath=FILEPATH_2)

## Compare expectations for a pair of nodes at a specific time
results = CXC2.compare_expectations(
    num_simulations=100, penalty=5, node_u=0, node_v=1, time=5, verbose=True
)

#### Look at all distances across all times
print(
    "Check that E[d^t(u,v)] < E[d^t(u, w)] for all u,v in the same cluster and u,w in different clusters across all times t."
)
CXC2.compare_expectations_all_vertices(num_simulations=1000, penalty=5)
expectations = CXC2.expectation_heatmap(num_simulations=1000, penalty=5)

plot_expectation_heatmap("cxc_heatmap.pdf", expectations)

#### Run STGkM on both connectivity matrices
MAX_DRIFT = 1
DRIFT_TIME_WINDOW = 1
PENALTY = 5

#####

for connectivity_matrix in [CXC1.connectivity_matrix, CXC2.connectivity_matrix]:
    c, opt_labels, opt_ltc = run_stgkm(
        connectivity_matrix=connectivity_matrix,
        penalty=PENALTY,
        num_clusters=3,
        max_drift=1,
        drift_time_window=1,
        max_iter=100,
        random_state=1,
    )

    print(opt_ltc)


num_members_list = [5, 10, 25, 50, 100]
num_clusters_list = [3, 5, 10, 20]
P_INTRA = 0.1
P_INTER = 0.01
filepath = "sensitivity_intra1_inter001.pkl"

# 1,1; 9,8; 8,7; 7,6; 6,5; 5,4; 4,3; 3,2


def test_sensitivity(num_members_list, num_clusters_list, p_intra, p_inter):

    data = {"num_members": [], "num_clusters": [], "num_timesteps": [], "ami_score": []}

    for NUM_MEMBERS in num_members_list:
        for NUM_CLUSTERS in num_clusters_list:
            print("Processing ", NUM_CLUSTERS, " clusters with ", NUM_MEMBERS, " each.")

            if (p_intra is None) and (p_inter is None):
                CXC = CliqueCrossClique(
                    num_clusters=NUM_CLUSTERS,
                    num_members=NUM_MEMBERS,
                    num_timesteps=100,
                )
            else:
                CXC = RandomCliqueCrossClique(
                    num_clusters=NUM_CLUSTERS,
                    num_members=NUM_MEMBERS,
                    num_timesteps=100,
                    p_intra=p_intra,
                    p_inter=p_inter,
                )
            connectivity_matrix = CXC.create_clique_cross_clique()

            true_labels = np.concatenate(
                [[i] * NUM_MEMBERS for i in range(NUM_CLUSTERS)]
            )

            for time in range(1, 50):
                _, _, opt_ltc = run_stgkm(
                    connectivity_matrix=connectivity_matrix[:time],
                    penalty=10,
                    num_clusters=NUM_CLUSTERS,
                    max_drift=1,
                    drift_time_window=1,
                    max_iter=100,
                    random_state=1,
                )

                score = adjusted_mutual_info_score(
                    labels_true=true_labels, labels_pred=opt_ltc
                )

                data["num_members"].append(NUM_MEMBERS)
                data["num_clusters"].append(NUM_CLUSTERS)
                data["num_timesteps"].append(time)
                data["ami_score"].append(score)
    return data


data = test_sensitivity(
    num_clusters_list=num_clusters_list,
    num_members_list=num_members_list,
    p_intra=P_INTRA,
    p_inter=P_INTER,
)

with open(filepath, "wb") as file:
    pickle.dump(data, file)
