from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score, mutual_info_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from main import MyKMeans

def internal_measurement(x, y, k, method="KMean"):
    model_1 = MyKMeans(k=k, x=x, true_label=y, random_state=42)
    if method == "KMean++":
        model_1.train(method = "KMean++")
    else:
        model_1.train()
    my_average_silhouette, _ = model_1.silhouette_score()
    mutual_info_score = model_1.normalized_mutual_information_score()

    # K-means clustering
    if method == "KMean++":
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    else:
        kmeans = KMeans(n_clusters=k, init="random", random_state=42)
    kmeans.fit(x)
    y_kmeans = kmeans.predict(x)
    real_silhouette_avg = silhouette_score(x, y_kmeans)
    nmi_score = normalized_mutual_info_score(y, y_kmeans) # mutual

    print(f'k = {k}, : WCSS: {kmeans.inertia_}, My Silhouette Score: {my_average_silhouette}, Real Silhouette Score: {real_silhouette_avg}')
    return kmeans.inertia_, my_average_silhouette, real_silhouette_avg

if __name__ == "__main__":
    iris_df = pd.read_csv('iris/iris.data', header=None)

    # Add column names manually if they aren't in the file
    iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    Y = iris_df['species']
    k = range(2, 10)

    wcss = []
    my_silhouette = []
    real_silhouette = []

    for cluster_num in k:
        one_wcss, one_my_silhouette, one_real_silhouette = internal_measurement(X, Y, cluster_num)
        wcss.append(one_wcss)
        my_silhouette.append(one_my_silhouette)
        real_silhouette.append(one_real_silhouette)


    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot the Elbow Curve (WCSS) on the first subplot
    axes[0].plot(k, wcss, marker='o', linestyle='--', color='b')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('WCSS')
    axes[0].set_title('Elbow Method for Optimal k')

    # Plot the Silhouette Score (my_silhouette) on the second subplot
    axes[1].plot(k, my_silhouette, marker='o', linestyle='--', color='g')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('my_silhouette')
    axes[1].set_title('Self-implemented Silhouette Score for Optimal k')

    # Plot the Silhouette Score (real_silhouette) on the third subplot
    axes[2].plot(k, real_silhouette, marker='o', linestyle='--', color='r')
    axes[2].set_xlabel('Number of Clusters (k)')
    axes[2].set_ylabel('real_silhouette')
    axes[2].set_title('Built-in Silhouette Score for Optimal k')

    plt.tight_layout()
    plt.show()
