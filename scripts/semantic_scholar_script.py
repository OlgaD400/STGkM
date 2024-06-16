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

########
# INCREASE THE TIME PERIOD TO YEARLY CITATIONS


def retreive_papers(year: int):
    """Retreive 1000 most highly cited papers for a given year."""

    # Define the paper search endpoint URL
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    frames = []
    for index in range(5):
        # Define the required query parameter and its value (in this case, the keyword we want to search for)
        query_params = {
            "query": "dynamic network",
            "limit": 100,
            "offset": index * 100,
            # "publicationDateOrYear": "2022-01-01:2022-01-31",
            "year": year,
            # "minCitationCount": 10,
            "sort": "citationCount:desc",
            "publicationTypes": "JournalArticle",
            "fields": "venue,publicationDate,authors,references.authors,references.venue",
        }

        # Directly define the API key (Reminder: Securely handle API keys in production environments)
        # api_key = "your api key goes here"  # Replace with the actual API key

        # Define headers with API key
        headers = {"x-api-key": "nRELYbPhEK8AYSDFRCZud1aJyjG3r904alKwt0e2"}

        # Make the GET request with the URL and query parameters
        searchResponse = requests.get(url, params=query_params, headers=headers)
        if searchResponse.status_code == 200:
            json_data = searchResponse.json()
            df = pd.DataFrame.from_dict(json_data["data"])
            frames.append(df)
        else:
            print(searchResponse.json())
            break
    if len(frames) > 0:
        fin_df = pd.concat(frames)

        fin_df["publicationDate"] = fin_df["publicationDate"].apply(
            lambda x: pd.to_datetime(x)
        )
        fin_df["month"] = fin_df["publicationDate"].dt.month
        fin_df["year"] = fin_df["publicationDate"].dt.year
        return fin_df
    else:
        print("no data generated for ", year)
        return None


# for year in [2020, 2021, 2022]:
# fin_dfs = []
# year_df = None
# count = 0
# for year in range(2000, 2024):
#     while (year_df is None) and (count < 3):
#         year_df = retreive_papers(year=year)
#         count += 1

#         if year_df is None:
#             print("Year ", year, "failed ", count, " times")

#     if year_df is not None:
#         fin_dfs.append(year_df)
#         year_df = None
#         count = 0

# final_df = pd.concat(fin_dfs)
# final_df.to_pickle(path="semantic_scholar_df.pkl")

final_df = pd.read_pickle("SS_loaded_data/semantic_scholar_df.pkl")


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
unique_years = sorted(final_df["year"].dropna().unique())
year_dict = dict(zip(unique_years, np.arange(len(unique_years))))
# dict(zip(np.arange(2017, 2024), np.arange(7)))
connectivity_matrix = np.zeros(
    (len(unique_years), len(final_venues), len(final_venues))
)
weight_matrix = np.ones((len(unique_years), len(final_venues), len(final_venues)))
venue_dict = dict(zip(final_venues, np.arange(len(final_venues))))
venue_dict_reverse = dict(zip(np.arange(len(final_venues)), final_venues))
for index, row in final_df.iterrows():
    year = row["year"]
    if ~np.isnan(year):
        time = year_dict[year]
        venue = row["venue"]
        if venue in final_venues:
            for paper in row["references"]:
                ref_venue = paper["venue"]
                if ref_venue in final_venues:
                    venue_index = venue_dict[venue]
                    ref_venue_index = venue_dict[ref_venue]

                    connectivity_matrix[int(time) - 1, venue_index, ref_venue_index] = 1
                    connectivity_matrix[int(time) - 1, ref_venue_index, venue_index] = 1
                    weight_matrix[int(time) - 1, venue_index, ref_venue_index] += 1
                    weight_matrix[int(time) - 1, ref_venue_index, venue_index] += 1


def run_stgkm(
    medoids,
    random_state,
    penalized_distance,
    max_drift=1,
    drift_time_window=1,
    max_iter=100,
):
    km = kmedoids.KMedoids(
        medoids,
        method="fasterpam_time",
        max_drift=max_drift,
        drift_time_window=drift_time_window,
        max_iter=max_iter,
        random_state=random_state,
        online=False,
    )
    c = km.fit(penalized_distance)
    return c


MAX_DRIFT = 1
DRIFT_TIME_WINDOW = 1
START = 0
END = -10
MIN_CLUSTERS = 2
MAX_CLUSTERS = 20
STR_ID = str(MAX_DRIFT) + str(DRIFT_TIME_WINDOW) + str(START) + str(-END) + "_temp"


NUM_CLUSTERS = 2
t, num_vertices, _ = connectivity_matrix.shape
PENALTY = t + END


distance_matrix = s_journey(connectivity_matrix=connectivity_matrix)
weighted_distance = 1 / weight_matrix * distance_matrix
penalized_distance = np.where(distance_matrix == np.inf, PENALTY, distance_matrix)
penalized_distance = 1 / weight_matrix * penalized_distance
# medoids = np.random.choice(num_vertices, NUM_CLUSTERS)

# num_connections = np.sum(np.sum(connectivity_matrix, axis=0), axis=1)
# connection_ind = np.where(num_connections >= 30)[0]
# penalized_distance_submat = penalized_distance[7:-5, connection_ind][
# #     :, :, connection_ind
# # ]
# medoids = np.argsort(
#     np.sum(connectivity_matrix[0, connection_ind][:, connection_ind], axis=1)
# )[::-1][:NUM_CLUSTERS]

# medoids = np.argsort(np.sum(connectivity_matrix[START], axis=1))[::-1][:NUM_CLUSTERS]
medoids = np.argsort(np.sum(penalized_distance[START], axis=1))[:NUM_CLUSTERS]
random_state = np.random.choice(200, 1)[0]

# c = run_stgkm(
#     medoids=medoids,
#     random_state=random_state,
#     penalized_distance=penalized_distance[START:END],
#     max_drift=MAX_DRIFT,
#     drift_time_window=DRIFT_TIME_WINDOW,
#     max_iter=100,
# )

# ltc = agglomerative_clustering(weights=c.labels_.T, num_clusters=NUM_CLUSTERS)
# for i in range(NUM_CLUSTERS):
#     ind = np.where(ltc == i)[0]
#     print([venue_dict_reverse[index] for index in ind], "\n\n")


# for time in range(t):
#     print([venue_dict_reverse[index] for index in c.medoid_indices_[time]], "\n\n")

obj_values, opt_k, label_history, medoid_history = choose_num_clusters(
    min_clusters=MIN_CLUSTERS,
    max_clusters=MAX_CLUSTERS,
    penalized_distance=penalized_distance[START:END],
    connectivity_matrix=connectivity_matrix[START:],
    random_state=random_state,
    max_drift=MAX_DRIFT,
    drift_time_window=DRIFT_TIME_WINDOW,
    max_iterations=100,
)

print(opt_k)
opt_labels = label_history[np.argmax(obj_values)]
opt_ltc = agglomerative_clustering(weights=opt_labels.T, num_clusters=opt_k)
for i in range(opt_k):
    ind = np.where(opt_ltc == i)[0]
    print([venue_dict_reverse[index] for index in ind], "\n\n")

similarity_matrix_figure(
    full_assignments=opt_labels,
    long_term_clusters=opt_ltc,
    fig_title="Semantic Scholar Data \n Short Term Cluster Similarity Scores k=%d"
    % opt_k,
    filepath="SS_figures/SS_short_term_cluster_similarity_" + STR_ID + ".pdf",
)


choosing_num_clusters_plot(
    min_num_clusters=MIN_CLUSTERS,
    max_num_clusters=MAX_CLUSTERS,
    sum_distance_from_centers=obj_values,
    fig_title="Avg. Silhouette Score vs. Number of Clusters",
    ylabel="Avg. Silhouette Score",
    filepath="SS_figures/SS_opt_num_clusters_" + STR_ID + ".pdf",
)
saved_labels = np.save("SS_loaded_data/opt_labels_" + STR_ID, opt_labels)
saved_medoids = np.save(
    "SS_loaded_data/opt_medoids_" + STR_ID, medoid_history[np.argmin(obj_values)]
)
saved_ltc = np.save("SS_loaded_data/opt_ltc_" + STR_ID, opt_ltc)


# Nature, Science, PNAS, bioRxiv "power players, always clustered together"


# for i in range(opt_k):
#     member_ind = np.where(opt_labels == i)[1]
#     num_clustered = len(member_ind)
#     print(num_clustered)

#     vals, counts = np.unique(member_ind, return_counts=True)
#     arg_sorted_counts = np.argsort(counts)[::-1]
#     sorted_counts = np.sort(counts)[::-1]
#     print(sorted_counts)
#     print([venue_dict_reverse[index] for index in vals[arg_sorted_counts]])

print("\n\n\n")
purity = []
for col in range(199):
    num_classifications = len(np.unique(opt_labels[:, col]))
    purity.append(num_classifications)

# for purity_level in np.unique(purity):
#     ind = np.where(purity == purity_level)[0]
#     purity_journals = [venue_dict_reverse[index] for index in ind]
#     if "arXiv.org" in purity_journals:
#         print("TRUE")
#     print(purity_level, purity_journals)
#     print("\n\n")


# purity_mat = np.zeros((opt_k, 199))
# for col in range(199):
#     for cluster in range(4):
#         num_classifications = len(np.where(opt_labels[:, col] == cluster)[0])
#         purity_mat[cluster, col] = num_classifications / 14


# for time in range(14):
#     print(time)
#     for i in range(opt_k):
#         mems = np.where(opt_labels[time] == i)[0]
#         print(i, [venue_dict_reverse[index] for index in mems])
#     print("\n\n")


# lens = []
# for i in range(opt_k):
#     mems = np.where(opt_ltc == i)[0]
#     lens.append(len(mems))
# len_args = np.argsort(lens)[::-1]

# for ind in len_args:
#     member_ind = np.where(opt_ltc == ind)[0]
#     print(len(member_ind))
#     print([venue_dict_reverse[index] for index in member_ind])

# for ind in range(opt_k):
#     member_ind = np.where(opt_labels == ind)[1]

#     vals, counts = np.unique(member_ind, return_counts=True)
#     arg_sorted_counts = np.argsort(counts)[::-1]
#     sorted_counts = np.sort(counts)[::-1]
#     print(sorted_counts)
#     print([venue_dict_reverse[index] for index in vals[arg_sorted_counts]])


_, axs = plt.subplots(figsize=(20, 20))
axs.hist(purity)
axs.set_title("Number of Clusters to which a Journal Belongs over Time", fontsize=40)
axs.set_ylabel("Count", fontsize=100)
axs.axvline(np.average(purity), color="r", linestyle="dashed", linewidth=4)
axs.set_xlabel("Number of Clusters", fontsize=50)
axs.tick_params(labelsize=30)
plt.savefig("SS_figures/cluster_membership_hist" + STR_ID + ".pdf", format="pdf")
