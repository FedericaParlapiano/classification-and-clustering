import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from yellowbrick.cluster import KElbowVisualizer
from matplotlib import cm
from sklearn.cluster import KMeans
from time import time
from sklearn import metrics
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


def silhouette_analysis(range_n_clusters, X):
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(12, 6)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        colors = iter([plt.cm.Paired(i) for i in range(0,20)])
        list = []
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            list.append(next(colors))
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                color=list[-1],
                edgecolor='w',
                alpha=0.7,
            )
            next(colors)
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples


        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = []
        for c in range(0, len(X)):
            if cluster_labels[c] == 0:
                colors.append(list[0])
            elif cluster_labels[c] == 1:
                colors.append(list[1])
            elif cluster_labels[c] == 2:
                colors.append(list[2])

        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=np.array(colors), edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.savefig('grafici/silhouette_kmeans', bbox_inches='tight')
    plt.show()



def elbow_method(X):
    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer1 = KElbowVisualizer(model, k=(2, 10))
    visualizer1.fit(X)
    visualizer1.show(outpath="grafici/kelbow_distorsion.png")

    visualizer2 = KElbowVisualizer(model, k=(2, 10), metric='calinski_harabasz')
    visualizer2.fit(X)
    visualizer2.show(outpath="grafici/kelbow_calinksi.png")

    visualizer3 = KElbowVisualizer(model, k=(2, 10), metric='silhouette')
    visualizer3.fit(X)
    visualizer3.show(outpath="grafici/kelbow_silhouette.png")


def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(None, kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))


data = pd.read_csv("..\\data\\obesity_dataset_clean.csv")
(n_samples, n_features), n_digits = data.shape, 3

print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

data = data.iloc[:, 1:]

labels = data["Nutritional Status"].values

data = data.drop("Nutritional Status", axis=1)

number = LabelEncoder()
data['Gender'] = number.fit_transform(data['Gender'])
data['Transportation Used'] = number.fit_transform(data['Transportation Used'])

elbow_method(data.values)

print(80 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI   \tAMI   \tsilhouette")

kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=10, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)

kmeans = KMeans(init="random", n_clusters=n_digits, n_init=10, random_state=0)
bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

d_pca1 = data.copy()
d_pca2 = data.copy()
d_pca3 = data.copy()
d_pca4 = data.copy()

pca = PCA(n_components=2).fit_transform(d_pca1)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=1, random_state=0)
bench_k_means(kmeans=kmeans, name="PCA2-based", data=pca, labels=labels)

pca = PCA(n_components=3).fit_transform(d_pca2)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init='auto', random_state=0)
bench_k_means(kmeans=kmeans, name="PCA3-based", data=pca, labels=labels)

pca = PCA(n_components=8)
pca_x = pca.fit_transform(d_pca3)
data_reduced = pd.DataFrame(pca_x).values
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init='auto', random_state=0)
bench_k_means(kmeans=kmeans, name="PCA8-based", data=pca_x, labels=labels)
kmeans.fit(data_reduced)

print(80 * "_")

pca = PCA(n_components=2)
pca_x = pca.fit_transform(d_pca4)
data_reduced = pd.DataFrame(pca_x).values

kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init='auto')
kmeans.fit(data_reduced)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = data_reduced[:, 0].min() - 1, data_reduced[:, 0].max() + 1
y_min, y_max = data_reduced[:, 1].min() - 1, data_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(data_reduced[:, 0], data_reduced[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title("K-means clustering")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.savefig('grafici/kmeans', bbox_inches='tight')
plt.show()

silhouette_analysis([3], data_reduced)
