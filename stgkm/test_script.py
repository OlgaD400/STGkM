import requests
import pandas as pd
import numpy as np
from stgkm.distance_functions import s_journey
import kmedoids
from stgkm.STGKM import agglomerative_clustering
import matplotlib.pyplot as plt
from stgkm_figures import (
    similarity_matrix_figure,
    choosing_num_clusters_plot,
    community_matrix_figure,
)
from stgkm.helper_functions import (
    calculate_objective,
    choose_num_clusters,
)

final_df = pd.read_pickle("semantic_scholar_df.pkl")


def get_venues(df: pd.DataFrame):
    """
    Return venues that have at least 10 citations and are both cited and referenced.
    """
    venue_list = []
    ref_venue_list = []
    for _, row in df.iterrows():
        venue = row["venue"]
        if len(venue) > 0:
            for paper in row["references"]:
                ref_venue = paper["venue"]
                if len(ref_venue) > 0:
                    venue_list.append(venue)
                    ref_venue_list.append(ref_venue)

    vals, counts = np.unique(ref_venue_list, return_counts=True)
    ten_citations_ind = np.where(counts >= 50)[0]
    ten_citations = vals[ten_citations_ind]
    final_venues = np.intersect1d(venue_list, ten_citations)

    return final_venues


final_venues = get_venues(final_df)
print(final_venues)
unique_years = sorted(final_df["year"].dropna().unique())
year_dict = dict(zip(unique_years, np.arange(len(unique_years))))
# dict(zip(np.arange(2017, 2024), np.arange(7)))
connectivity_matrix = np.zeros(
    (len(unique_years), len(final_venues), len(final_venues))
)
weight_matrix = np.ones((len(unique_years), len(final_venues), len(final_venues)))
venue_dict = dict(zip(final_venues, np.arange(len(final_venues))))
venue_dict_reverse = dict(zip(np.arange(len(final_venues)), final_venues))
opt_labels = np.load("SS_loaded_data/opt_labels_11010_14.npy")
opt_medoids = np.load("SS_loaded_data/opt_medoids_11010_14.npy")
opt_ltc = np.load("SS_loaded_data/opt_ltc_11010_14.npy")
opt_k = 18

for ind in range(opt_k):
    member_ind = np.where(opt_labels == ind)[1]

    vals, counts = np.unique(member_ind, return_counts=True)
    arg_sorted_counts = np.argsort(counts)[::-1]
    # print(arg_sorted_counts)
    sorted_counts = np.sort(counts)[::-1]
    print(sorted_counts)
    print([venue_dict_reverse[index] for index in vals[arg_sorted_counts]])
