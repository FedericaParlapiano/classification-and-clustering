import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


def elbow_method(dataset, n_knn):
    neighbors = NearestNeighbors(n_neighbors=n_knn)
    neighbors_fit = neighbors.fit(dataset)
    distances, indices = neighbors_fit.kneighbors(dataset)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    kl = KneeLocator(range(1, len(distances) + 1), distances, curve="convex")
    kl.plot_knee()
    plt.show()
    plt.savefig('grafici/elbow_dbscan', bbox_inches='tight')

    return kl.elbow, kl.knee_y


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
data_reduced = data_scaled

n = 5  # con scaling
# n = 10 # senza scaling e senza PCA, con scaling e con PCA, solo PCA
x, eps = elbow_method(data_reduced, n)
print("eps=" + str(eps))

eps_to_test = [round(eps, 2) for eps in np.arange((eps - 0.05), eps + 0.05, 0.01)]
min_samples_to_test = range(5, 15, 1)
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

sns.heatmap(results_noise, annot=True, ax=ax1, cbar=False).set_title("METRIC: Mean Noise Points Distance")
sns.heatmap(results_clusters, annot=True, ax=ax2, cbar=False).set_title("METRIC: Number of clusters")

ax1.set_xlabel("N");
ax2.set_xlabel("N")
ax1.set_ylabel("EPSILON");
ax2.set_ylabel("EPSILON")

plt.tight_layout()
plt.show()

min_points = 5
db = DBSCAN(eps=5.26, min_samples=min_points).fit(data_reduced)

ymeans = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(ymeans)) - (1 if -1 in ymeans else 0)
n_noise_ = list(ymeans).count(-1)

# metriche DBSCAN per il salvataggio su file
homogeneity = metrics.homogeneity_score(ymeans, labels)
completeness = metrics.completeness_score(ymeans, labels)
v_measure = metrics.v_measure_score(ymeans, labels)
ari = metrics.adjusted_rand_score(ymeans, labels)
ami = metrics.adjusted_rand_score(ymeans, labels)
silhouette = metrics.silhouette_score(data_reduced, ymeans)
m_calinski = metrics.calinski_harabasz_score(data_reduced, ymeans)
m_bouldin = metrics.davies_bouldin_score(data_reduced, ymeans)

df = pd.DataFrame({'Scaling': ['yes'],
                   'knn': n,
                   'eps': eps,
                   'min points': min_points,
                   'n_cluster': n_clusters_,
                   'homogeneity': homogeneity,
                   'completeness': completeness,
                   'v_measure': v_measure,
                   'ari': ari,
                   'ami': ami,
                   'calinksi': m_calinski,
                   'bouldin': m_bouldin,
                   'silhouette': silhouette,
                   })

df.to_csv('metrics_DBSCAN.csv', header=None, index=False, mode='a')

print(f"Homogeneity: {metrics.homogeneity_score(ymeans, labels):.3f}")
print(f"Completeness: {metrics.completeness_score(ymeans, labels):.3f}")
print(f"V-measure: {metrics.v_measure_score(ymeans, labels):.3f}")
print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(ymeans, labels):.3f}")
print("Adjusted Mutual Information:"f" {metrics.adjusted_mutual_info_score(ymeans, labels):.3f}")
print(f"Calinski Harabasz Score: {metrics.calinski_harabasz_score(data_reduced, ymeans):.3f}")
print(f"Davies Bouldin Score: {metrics.davies_bouldin_score(data_reduced, ymeans):.3f}")
print(f"Silhouette Coefficient: {metrics.silhouette_score(data_reduced, ymeans):.3f}")
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

'''
print(CDbw(data_reduced.values, ymeans, metric="euclidean"))

plt.figure(figsize=(15,8))
plt.title('Cluster of PCAs', fontsize = 30)

for i in range(-1, n_clusters_+1):
    plt.scatter(pca_x[ymeans == i, 0], pca_x[ymeans == i, 1], s = 100)
    if i == -1:
        plt.scatter(pca_x[ymeans == i, 0], pca_x[ymeans == i, 1], s=100, c='black')

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()'''

plt.figure(figsize=(15, 8))
plt.title('Cluster of PCAs', fontsize=30)

for i in range(-1, n_clusters_ + 1):
    plt.scatter(data_reduced.values[ymeans == i, 2], data_reduced.values[ymeans == i, 3], s=100)
    if i == -1:
        plt.scatter(data_reduced.values[ymeans == i, 2], data_reduced.values[ymeans == i, 3], s=100, c='black')

plt.xlabel('Weight')
plt.ylabel('Height')
plt.legend()
plt.show()
