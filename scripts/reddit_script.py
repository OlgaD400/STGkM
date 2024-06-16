import numpy as np
import pandas as pd
from stgkm.distance_functions import s_journey
import kmedoids
import matplotlib.pyplot as plt
from stgkm.helper_functions import (
    calculate_cluster_connectivity,
    return_avg_cluster_connectivity,
    choose_num_clusters,
)
from stgkm.STGKM import (
    STGKM,
    similarity_matrix,
    similarity_measure,
    agglomerative_clustering,
)
from stgkm_figures import (
    similarity_matrix_figure,
    choosing_num_clusters_plot,
    community_matrix_figure,
)
import os
from typing import Optional

SENTIMENT = "positive"


def get_reddit_df(
    out_filepath: os.PathLike, sentiment: str, in_filepath: Optional[os.PathLike] = None
):
    """
    Get the reddit dataframe. Create it if it doesn't exist. Load it if it does. Output the desired sentiment dataframe.
    """
    assert sentiment in [
        "positive",
        "negative",
    ], "Sentiment can only be positive or negative."

    if not os.path.isfile(out_filepath):
        assert (
            in_filepath is not None
        ), "If out_filepath does not exist, in_filepath must be provided."
        df = pd.read_csv(in_filepath, sep="\t")
        df["TIMESTAMP"] = df["TIMESTAMP"].apply(lambda x: pd.to_datetime(str(x)))
        df["DATE"] = df["TIMESTAMP"].dt.date
        unique_dates = np.sort(df["DATE"].unique())
        date_dict = dict(zip(unique_dates, np.arange(len(unique_dates))))
        df["TIMESTAMP_INT"] = df["DATE"].apply(lambda x: date_dict[x])
        df.to_pickle(out_filepath)
    else:
        df = pd.read_pickle(out_filepath)

    if sentiment == "positive":
        fin_df = df[df["LINK_SENTIMENT"] == 1]
    elif sentiment == "negative":
        fin_df = df[df["LINK_SENTIMENT"] == -1]
    fin_df["DATE"] = fin_df["TIMESTAMP"].dt.to_period("M")
    return fin_df


# in_filepath = "data/soc-redditHyperlinks-title.tsv"
# out_filepath = "Reddit_loaded_data/reddit_df.pkl"

fin_df = get_reddit_df(
    out_filepath="Reddit_loaded_data/reddit_df.pkl", sentiment="positive"
)

# og_df = pd.read_pickle("Reddit_loaded_data/reddit_df.pkl")
# og_df["YEAR"] = og_df["TIMESTAMP"].dt.year
# og_df["MONTH"] = og_df["TIMESTAMP"].dt.month
# pos_df = og_df[
#     (og_df["LINK_SENTIMENT"] == 1) & (og_df["YEAR"] == 2015) & (og_df["MONTH"] == 2)
# ]
# if SENTIMENT == "positive":
#     pos_df = og_df[og_df["LINK_SENTIMENT"] == 1]
# elif SENTIMENT == "negative":
#     pos_df = og_df[og_df["LINK_SENTIMENT"] == -1]
# pos_df["DATE"] = pos_df["TIMESTAMP"].dt.to_period("M")


def load_reddit_connectivity_matrix(out_folder: str, sentiment: str):
    """
    Load reddit connectivity matrix and connection weights.
    """
    if SENTIMENT == "positive":
        connectivity_matrix = np.load(out_folder + "/reddit_connectivity_positive.npy")
        weight_matrix = np.load(out_folder + "/reddit_weight_positive.npy")
    elif SENTIMENT == "negative":
        connectivity_matrix = np.load(out_folder + "/reddit_connectivity_negative.npy")
        weight_matrix = np.load(out_folder + "/reddit_weight_negative.npy")
    return connectivity_matrix, weight_matrix


def create_reddit_connectivity_matrix(
    reddit_df, subreddits, sentiment: str, out_folder: str
):
    """
    Create the connectivity matrix and the weight matrix.
    """
    unique_dates = np.sort(reddit_df["DATE"].unique())
    date_dict = dict(zip(unique_dates, np.arange(len(unique_dates))))
    timestamps = len(unique_dates)
    subreddit_dict = dict(zip(subreddits, np.arange(len(subreddits))))

    connectivity_matrix = np.zeros((timestamps, len(subreddits), len(subreddits)))
    weight_matrix = np.ones((timestamps, len(subreddits), len(subreddits)))

    for index, row in reddit_df.iterrows():
        source = row["SOURCE_SUBREDDIT"]
        target = row["TARGET_SUBREDDIT"]

        if (source in subreddits) & (target in subreddits):
            time = date_dict[row["DATE"]]
            V_A = subreddit_dict[source]
            V_B = subreddit_dict[target]
            connectivity_matrix[time, V_A, V_B] = 1
            connectivity_matrix[time, V_B, V_A] = 1
            weight_matrix[time, V_A, V_B] += 1
            weight_matrix[time, V_B, V_A] += 1
    ##########################
    np.save(
        out_folder + "/reddit_connectivity_" + sentiment + ".npy", connectivity_matrix
    )
    np.save(out_folder + "/reddit_weight_" + sentiment + ".npy", weight_matrix)
    print("matrices created")
    return None


def get_reddit_connectivity_matrix(sentiment: str, fin_df):
    "Get the reddit connectivity_matrix. Create it if it doesn't exist. Load it if it does."

    # unique_dates = np.sort(fin_df["DATE"].unique())
    # date_dict = dict(zip(unique_dates, np.arange(len(unique_dates))))

    ########### EXTRACT SOURCE TARGET PAIRS ####################
    source_vc = fin_df["SOURCE_SUBREDDIT"].value_counts()
    source_reddits = source_vc[source_vc.values > 20].index.values

    source_target_pairs = fin_df[fin_df["SOURCE_SUBREDDIT"].isin(source_reddits)][
        "TARGET_SUBREDDIT"
    ]

    target_vc = fin_df["TARGET_SUBREDDIT"].value_counts()
    target_reddits = target_vc[target_vc.values > 20].index.values

    source_target_threshold = np.intersect1d(source_target_pairs, target_reddits)

    subreddits = np.intersect1d(source_reddits, source_target_threshold)

    ##########################################################
    if not os.path.isfile(
        out_folder + "/reddit_connectivity_" + sentiment + ".npy",
    ):
        create_reddit_connectivity_matrix(
            reddit_df=fin_df, subreddits=subreddits, out_folder="Reddit_loaded_data"
        )
    else:
        load_reddit_connectivity_matrix(out_folder, sentiment=sentiment)

    # subreddit_dict = dict(zip(subreddits, np.arange(len(subreddits))))
    index_to_subreddit = dict(zip(np.arange(len(subreddits)), subreddits))
    # timestamps = len(fin_df["DATE"].unique())

    # connectivity_matrix = np.zeros((timestamps, len(subreddits), len(subreddits)))
    # weight_matrix = np.ones((timestamps, len(subreddits), len(subreddits)))

    # for index, row in pos_df.iterrows():
    #     source = row["SOURCE_SUBREDDIT"]
    #     target = row["TARGET_SUBREDDIT"]

    #     if (source in subreddits) & (target in subreddits):
    #         time = date_dict[row["DATE"]]
    #         V_A = subreddit_dict[source]
    #         V_B = subreddit_dict[target]
    #         connectivity_matrix[time, V_A, V_B] = 1
    #         connectivity_matrix[time, V_B, V_A] = 1
    #         weight_matrix[time, V_A, V_B] += 1
    #         weight_matrix[time, V_B, V_A] += 1

    ###########################
    # np.save("reddit_connectivity_negative.npy", connectivity_matrix)
    # np.save("reddit_weight_negative.npy", weight_matrix)
    # print("matrices created")

    # if SENTIMENT == "positive":
    #     connectivity_matrix = np.load(
    #         "Reddit_loaded_data/reddit_connectivity_positive.npy"
    #     )
    #     weight_matrix = np.load("Reddit_loaded_data/reddit_weight_positive.npy")
    # elif SENTIMENT == "negative":
    #     connectivity_matrix = np.load(
    #         "Reddit_loaded_data/reddit_connectivity_negative.npy"
    #     )
    #     weight_matrix = np.load("Reddit_loaded_data/reddit_weight_negative.npy")


# # # calculate s_journey
MAX_DRIFT = 1
DRIFT_TIME_WINDOW = 1
START = 0
END = -5
MIN_CLUSTERS = 2
MAX_CLUSTERS = 20
STR_ID = (
    "_"
    + SENTIMENT
    + "_"
    + str(MAX_DRIFT)
    + str(DRIFT_TIME_WINDOW)
    + str(START)
    + str(-END)
    + "_TEMP"
)


t, num_vertices, _ = connectivity_matrix.shape
PENALTY = t + END


distance_matrix = s_journey(connectivity_matrix)

penalized_distance = np.where(distance_matrix == np.inf, PENALTY, distance_matrix)
penalized_distance = 1 / weight_matrix * penalized_distance

random_state = np.random.choice(100, 1)[0]

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
# for i in range(opt_k):
#     ind = np.where(opt_ltc == i)[0]
#     print([rev_user_dict[index] for index in ind], "\n\n")

similarity_matrix_figure(
    full_assignments=opt_labels,
    long_term_clusters=opt_ltc,
    fig_title="Negative Sentiment Reddit Data \n Short Term Cluster Similarity Scores k=%d"
    % opt_k,
    filepath="reddit_sentiment" + STR_ID + ".pdf",
)

choosing_num_clusters_plot(
    min_num_clusters=MIN_CLUSTERS,
    max_num_clusters=MAX_CLUSTERS,
    sum_distance_from_centers=obj_values,
    fig_title="Avg. Silhouette Score vs. Number of Clusters",
    ylabel="Avg. Silhouette Score",
    filepath="Reddit_figures/Reddit_opt_num_clusters" + STR_ID + ".pdf",
)

########### PURITY HISTOGRAM
purity = []
for col in range(41):
    num_classifications = len(np.unique(opt_labels[:, col]))
    purity.append(num_classifications)

_, axs = plt.subplots(figsize=(20, 20))
axs.hist(purity)
axs.set_title(
    "Number of Clusters to which \n a Negative Subreddit Belongs over Time", fontsize=50
)
axs.set_ylabel("Count", fontsize=50)
axs.axvline(np.average(purity), color="r", linestyle="dashed", linewidth=4)
axs.set_xlabel("Number of Clusters", fontsize=50)
axs.tick_params(labelsize=30)
plt.savefig("Reddit_cluster_membership_hist" + STR_ID + ".pdf", format="pdf")

###############################################
for i in range(opt_k):
    mems = np.where(opt_ltc == i)[0]
    print([index_to_subreddit[index] for index in mems])


for i in range(opt_k):
    member_ind = np.where(opt_labels == i)[1]
    num_clustered = len(member_ind)
    print(num_clustered)

    vals, counts = np.unique(member_ind, return_counts=True)
    arg_sorted_counts = np.argsort(counts)[::-1]
    sorted_counts = np.sort(counts)[::-1]
    print(sorted_counts)
    print([index_to_subreddit[index] for index in vals[arg_sorted_counts]])


saved_labels = np.save("Reddit_loaded_data/Reddit_opt_labels_" + STR_ID, opt_labels)
saved_medoids = np.save(
    "Reddit_loaded_data/Reddit_opt_medoids" + STR_ID,
    medoid_history[np.argmin(obj_values)],
)
saved_ltc = np.save("Reddit_loaded_data/Reddit_opt_ltc_" + STR_ID, opt_ltc)
