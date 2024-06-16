""" Roll Call Data Clustering Script"""

import pandas as pd
import numpy as np
import time
import kmedoids
from stgkm.distance_functions import s_journey
from stgkm.STGKM import STGKM
from stgkm_figures import (
    choosing_num_clusters,
    choosing_num_clusters_plot,
    three_snapshots_dynamic_clustering,
    similarity_matrix_figure,
)
from stgkm.helper_functions import calculate_objective, choose_num_clusters
from stgkm.STGKM import agglomerative_clustering

### Load Data ###
final_voter_data = pd.read_csv("data/final_voter_data.csv")
fvd_0 = pd.read_csv("data/fvd_0")
voter_connectivity = np.load("data/roll_call_connectivity_2.npy")
voter_dict_reverse = dict(zip(np.arange(len(fvd_0)), fvd_0["legislator"].values))

### Calculate or load s-journey ###
# distance_matrix = s_journey(voter_connectivity)
## np.save('data/roll_call_distance.npy', distance_matrix)
distance_matrix = np.load("data/roll_call_distance.npy")
# Run STGkM on a subset of 100 roll call votes
# SUBSET = 100
# subset_matrix = distance_matrix[:SUBSET]
TIME, NUM_VERTICES, _ = distance_matrix.shape
NUM_CLUSTERS = 2
MIN_CLUSTERS = 2
MAX_CLUSTERS = 11
START = 0
END = 100
random_state = np.random.choice(100, 1)[
    0
]  # 79  # np.random.choice(100, 1)[0]  # 79  # np.random.choice(200, 1)[0]  # 79  #
MAX_DRIFT = 1
DRIFT_TIME_WINDOW = 5
PENALTY = 5

penalized_distance = np.where(distance_matrix == np.inf, PENALTY, distance_matrix)


# obj_values, opt_k, label_history, medoid_history = choose_num_clusters(
#     min_clusters=MIN_CLUSTERS,
#     max_clusters=MAX_CLUSTERS,
#     penalized_distance=penalized_distance[START:END],
#     connectivity_matrix=voter_connectivity[START:],
#     random_state=random_state,
#     max_drift=MAX_DRIFT,
#     drift_time_window=DRIFT_TIME_WINDOW,
#     max_iterations=100,
#     medoid_selection="num",
# )


# opt_labels = label_history[np.argmax(obj_values)]
# opt_medoids = medoid_history[np.argmax(obj_values)]
# opt_ltc = agglomerative_clustering(weights=opt_labels.T, num_clusters=opt_k)
# print("k: ", opt_k)
# for i in range(opt_k):
#     ind = np.where(opt_ltc == i)[0]
#     print(len(ind))


# similarity_matrix_figure(
#     full_assignments=opt_labels,
#     long_term_clusters=opt_ltc,
#     fig_title="Roll Call Data \n Short Term Cluster Similarity Scores k=%d" % opt_k,
#     filepath="temp" + ".pdf",
# )

# choosing_num_clusters_plot(
#     min_num_clusters=MIN_CLUSTERS,
#     max_num_clusters=MAX_CLUSTERS,
#     sum_distance_from_centers=obj_values,
#     filepath="roll_call_choose_k_temp.pdf",
#     fig_title="Avg. Silhouette Score vs. Number of Clusters",
#     ylabel="Avg. Silhouette Score",
# )

# for i in range(opt_k):
#     mems = np.where(opt_ltc == i)[0]
#     voters = [voter_dict_reverse[index] for index in mems]
#     names = fvd_0[fvd_0["legislator"].isin(voters)]["last_name"].values
#     print(i, names)

# 273
# for time in range(100):
#     print("\n\n", time)
#     for k in range(opt_k):
#         mems = np.where(opt_labels[time] == k)[0]
#         votes = fvd_0[fvd_0["legislator_id"].isin(mems)]["vote"].value_counts()
#         party = fvd_0[fvd_0["legislator_id"].isin(mems)]["party"].value_counts()
#         print(k, votes, party)

start = time.time()
stgkm = STGKM(
    distance_matrix=distance_matrix[START:END],
    penalty=PENALTY,
    max_drift=MAX_DRIFT,
    drift_time_window=DRIFT_TIME_WINDOW,
    num_clusters=NUM_CLUSTERS,
    tie_breaker=False,
    iterations=100,
)
penalized_distance = stgkm.penalize_distance()
stgkm.run_stgkm(method="proxy")
end = time.time()

print(end - start)

start = time.time()
km = kmedoids.KMedoids(
    2,
    method="fasterpam_time",
    max_drift=MAX_DRIFT,
    drift_time_window=DRIFT_TIME_WINDOW,
    max_iter=100,
    random_state=np.random.choice(100, 1)[0],
    online=False,
)
c = km.fit(penalized_distance[START:END])
opt_ltc = agglomerative_clustering(weights=c.labels_.T, num_clusters=opt_k)
end = time.time()


print(end - start)
opt_k = 2
opt_ltc = agglomerative_clustering(weights=c.labels_.T, num_clusters=opt_k)
opt_labels = c.labels_
opt_medoids = c.medoid_indices_


# for i in range(opt_k):
#     mems = np.where(opt_ltc == i)[0]
#     voters = [voter_dict_reverse[index] for index in mems]
#     names = fvd_0[fvd_0["legislator"].isin(voters)]["last_name"].values
#     print(i, names)
# end = time.time()

### Visualize long term cluster similarity matrix ####
# similarity_matrix_figure(
#     full_assignments=opt_labels,
#     long_term_clusters=opt_ltc,
#     fig_title="Roll Call Data \n Short Term Cluster Similarity Scores k = 3",
#     filepath="similarity_matrix_k2.pdf",
# )

### Visualize three snapshots of dynamic graph ###
# fig = three_snapshots_dynamic_clustering(
#     connectivity_matrix=voter_connectivity,
#     timesteps=[10, 20, 30],
#     num_clusters=3,
#     membership=stgkm.full_assignments,
#     centers=stgkm.full_centers,
#     fig_title="Roll Call Vote Evolution",
#     snapshot_title="Vote #",
#     filepath="voter_fig_attempt.pdf",
#     pkl_path="STGKM_Figures/saved_pos.pkl",
# )

# fig = three_snapshots_dynamic_clustering(
#     connectivity_matrix=voter_connectivity,
#     timesteps=[10, 20, 30],
#     num_clusters=3,
#     membership=opt_labels,
#     centers=opt_medoids,
#     fig_title="Roll Call Vote Evolution",
#     snapshot_title="Vote #",
#     filepath="roll_call_evolution_k3.pdf",
#     pkl_path="STGKM_Figures/saved_pos.pkl",
# )

### Find optimal number of clusters ###

# sum_distance_from_centers = choosing_num_clusters(min_clusters =1,
#                                                   max_clusters = 11,
#                                                   distance_matrix = subset_matrix,
#                                                   penalty=5,
#                                                   max_drift =1,
#                                                   drift_time_window = 5,
#                                                   tie_breaker = False,
#                                                   iterations = 100)
