import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN

from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from cdbw import CDbw
#from DBCV import DBCV


def get_metrics(eps, min_samples, dataset, iter_):
    # Fitting ======================================================================

    dbscan_model_ = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_model_.fit(dataset)

    # Mean Noise Point Distance metric =============================================
    noise_indices = dbscan_model_.labels_ == -1

    if True in noise_indices:
        neighboors = NearestNeighbors(n_neighbors=6).fit(dataset)
        distances, indices = neighboors.kneighbors(dataset)
        noise_distances = distances[noise_indices, 1:]
        noise_mean_distance = round(noise_distances.mean(), 3)
    else:
        noise_mean_distance = None

    number_of_clusters = len(set(dbscan_model_.labels_[dbscan_model_.labels_ >= 0]))

    print("%3d | Tested with eps = %3s and min_samples = %3s | %5s %4s" % (
        iter_, eps, min_samples, str(noise_mean_distance), number_of_clusters))

    return noise_mean_distance, number_of_clusters


file = '../data/obesity_dataset_clean.csv'
data = pd.read_csv(file)
data = data.iloc[:, 1:]
labels = data["Nutritional Status"].values
data = data.drop("Nutritional Status", axis=1)

number = LabelEncoder()
data['Gender'] = number.fit_transform(data['Gender'])
data["Transportation Used"] = number.fit_transform(data["Transportation Used"].astype('str'))

scaler = StandardScaler()
scaled_array = scaler.fit_transform(data)
data_scaled = pd.DataFrame(scaled_array, columns=data.columns)

data_copy = data.copy()
pca = PCA(n_components=2)
pca_x = pca.fit_transform(data_copy)
data_reduced = pd.DataFrame(pca_x)

#data_reduced = data_scaled

n = 5 # con scaling
#n = 10 # senza scaling e senza PCA
#n = 10 # con scaling e con PCA, solo PCA
neighbors = NearestNeighbors(n_neighbors=n)
neighbors_fit = neighbors.fit(data_reduced)
distances, indices = neighbors_fit.kneighbors(data_reduced)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]

kl = KneeLocator(range(1, len(distances) + 1), distances, curve="convex")
kl.plot_knee()
# plt.show()

x = kl.elbow
eps = kl.knee_y

print("eps=" + str(eps))

eps_to_test = [round(eps, 2) for eps in np.arange((eps - 0.5), eps + 0.5, 0.1)]
min_samples_to_test = range(10, 30, 2)
print(eps_to_test)

results_noise = pd.DataFrame(
    data=np.zeros((len(eps_to_test), len(min_samples_to_test))),  # Empty dataframe
    columns=min_samples_to_test,
    index=eps_to_test
)

# Dataframe per la metrica sul numero di cluster
results_clusters = pd.DataFrame(
    data=np.zeros((len(eps_to_test), len(min_samples_to_test))),  # Empty dataframe
    columns=min_samples_to_test,
    index=eps_to_test
)

iter_ = 0

print("ITER| INFO%s |  DIST    CLUS" % (" " * 39))
print("-" * 65)

for eps in eps_to_test:
    for min_samples in min_samples_to_test:
        iter_ += 1

        # Calcolo le metriche
        noise_metric, cluster_metric = get_metrics(eps, min_samples, data_reduced, iter_)

        # Inserisco i risultati nei relativi dataframe
        results_noise.loc[eps, min_samples] = noise_metric
        results_clusters.loc[eps, min_samples] = cluster_metric

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8) )

sns.heatmap(results_noise, annot = True, ax = ax1, cbar = False).set_title("METRIC: Mean Noise Points Distance")
sns.heatmap(results_clusters, annot = True, ax = ax2, cbar = False).set_title("METRIC: Number of clusters")

ax1.set_xlabel("N"); ax2.set_xlabel("N")
ax1.set_ylabel("EPSILON"); ax2.set_ylabel("EPSILON")

plt.tight_layout()
plt.show()

db = DBSCAN(eps=5.3, min_samples=10).fit(data_reduced)

ymeans = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(ymeans)) - (1 if -1 in ymeans else 0)
n_noise_ = list(ymeans).count(-1)

print(f"Homogeneity: {metrics.homogeneity_score(ymeans, labels):.3f}")
print(f"Completeness: {metrics.completeness_score(ymeans, labels):.3f}")
print(f"V-measure: {metrics.v_measure_score(ymeans, labels):.3f}")
print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(ymeans, labels):.3f}")
print(
    "Adjusted Mutual Information:"
    f" {metrics.adjusted_mutual_info_score(ymeans, labels):.3f}"
)
print(f"Silhouette Coefficient: {metrics.silhouette_score(data_reduced, ymeans):.3f}")

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

print(CDbw(data_reduced.values, ymeans, metric="euclidean"))

'''plt.figure(figsize=(15,8))
plt.title('Cluster of PCAs', fontsize = 30)

plt.scatter(pca_x[ymeans == -1, 0], pca_x[ymeans == -1, 1], s = 100, c = 'black')
plt.scatter(pca_x[ymeans == 0, 0], pca_x[ymeans == 0, 1], s = 100, c = 'pink')
plt.scatter(pca_x[ymeans == 1, 0], pca_x[ymeans == 1, 1], s = 100, c = 'orange')
plt.scatter(pca_x[ymeans == 2, 0], pca_x[ymeans == 2, 1], s = 100, c = 'lightgreen')
plt.scatter(pca_x[ymeans == 3, 0], pca_x[ymeans == 3, 1], s = 100, c = 'blue')
plt.scatter(pca_x[ymeans == 4, 0], pca_x[ymeans == 4, 1], s = 100, c = 'gray')
plt.scatter(pca_x[ymeans == 5, 0], pca_x[ymeans == 5, 1], s = 100, c = 'red')

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()'''


