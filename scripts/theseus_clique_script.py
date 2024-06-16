import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from stgkm.helper_functions import run_stgkm
from sklearn.metrics.cluster import adjusted_mutual_info_score
import pickle
import seaborn as sns
from stgkm.synthetic_graphs import TheseusClique

NUM_MEMBERS = 5
NUM_TIMESTEPS = 20
TC = TheseusClique(num_members=NUM_MEMBERS, num_timesteps=100)
_ = TC.create_theseus_clique()

#### CREATE FIGURE
labels = [[i] * NUM_MEMBERS for i in range(2)]
labels = np.concatenate(labels)
color_dict = {0: "dodgerblue", 1: "red"}
fig, axs = plt.subplots(2, 5, figsize=(40, 15))
axs = axs.flatten()
for time, ax in enumerate(axs):
    graph = nx.from_numpy_array(TC.connectivity_matrix[time])

    # if time == 0:
    #     pos = nx.spring_layout(graph, k = .8)

    pos = nx.spring_layout(graph, k=0.9)

    # plt.figure()
    nx.draw(
        graph,
        nodelist=np.arange(NUM_MEMBERS * 2),
        node_color=[color_dict[label] for label in labels],
        node_size=2000,
        pos=pos,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos=pos,
        labels=dict(zip(np.arange(10), np.arange(10))),
        font_size=40,
        ax=ax,
    )
    ax.set_title("Timestep %d" % time, fontsize=40)

for i in range(1, 5):
    plt.plot(
        [i / 5, i / 5],
        [0, 1],
        color="k",
        lw=5,
        transform=plt.gcf().transFigure,
        clip_on=False,
    )
plt.plot(
    [0, 1],
    [0.5, 0.5],
    color="k",
    lw=5,
    transform=plt.gcf().transFigure,
    clip_on=False,
)
fig.tight_layout()
plt.savefig("theseus_clique_pattern.pdf", format="pdf")
plt.show()


num_members_list = [5, 10, 25, 50]


def test_sensitivity(num_members_list):
    data = {"num_members": [], "num_timesteps": [], "ami_score": []}

    for NUM_MEMBERS in num_members_list:
        print("Processing clusters with ", NUM_MEMBERS, " each.")
        PENALTY = NUM_MEMBERS
        TC = TheseusClique(
            num_members=NUM_MEMBERS,
            num_timesteps=100,
        )
        connectivity_matrix = TC.create_theseus_clique()

        true_labels = np.concatenate([[i] * NUM_MEMBERS for i in range(2)])

        for time in range(1, 100):
            c, opt_labels, opt_ltc = run_stgkm(
                connectivity_matrix=connectivity_matrix[:time],
                penalty=PENALTY,
                num_clusters=2,
                max_drift=1,
                drift_time_window=1,
                max_iter=100,
                random_state=1,
            )

            score = adjusted_mutual_info_score(
                labels_true=true_labels, labels_pred=opt_ltc
            )

            data["num_members"].append(NUM_MEMBERS)
            data["num_timesteps"].append(time)
            data["ami_score"].append(score)
    return data


filepath = "theseus_clique_data.pkl"
data = test_sensitivity(
    num_members_list=num_members_list,
)

with open(filepath, "wb") as file:
    pickle.dump(data, file)

df = pd.DataFrame.from_dict(data=data)
df.rename(
    {
        "num_timesteps": "Number of time steps",
        "ami_score": "AMI score",
        "num_members": "Cluster size",
    },
    axis=1,
    inplace=True,
)

min = 10
max = 200

plt.figure(figsize=(5, 5))
sns.scatterplot(
    data=df,
    x="Number of time steps",
    y="AMI score",
    size="Cluster size",
    alpha=0.5,
    sizes=(min, max),
    palette="colorblind",
)
plt.show()
