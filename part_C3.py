import pandas as pd
import matplotlib.pyplot as plt
from part_C1 import internal_measurement
from main import MyKMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mutual_info_score, normalized_mutual_info_score




def external_measurement(x, y, k, method="KMeans"):
    model_1 = MyKMeans(k=k, x=x, true_label=y, random_state=42)
    if method == "KMean++":
        model_1.train(method="KMean++")
    else:
        model_1.train()
    #my_average_silhouette, _ = model_1.silhouette_score()
    mutual_info_score = model_1.normalized_mutual_information_score()

    # K-means clustering
    if method == "KMean++":
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    else:
        kmeans = KMeans(n_clusters=k, init="random", random_state=42)
    kmeans.fit(x)
    y_kmeans = kmeans.predict(x)
    #real_silhouette_avg = silhouette_score(x, y_kmeans)
    nmi_score = normalized_mutual_info_score(y, y_kmeans)  # mutual

    print(f'k = {k}, : my NMI: {mutual_info_score}, real NMI: {nmi_score}')
    return  mutual_info_score, nmi_score


if __name__ == '__main__':
    iris_df = pd.read_csv('iris/iris.data', header=None)
    wine_df = pd.read_csv('wine/wine.data', header=None)

    iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    Y = iris_df['species']

    X2 = wine_df.iloc[:, 1:]
    Y2 = wine_df.iloc[:, 0]

    k2 = range(2, 10)

    for cluster_num in k2:
        print('iris dataset:')#iris dataset
        external_measurement(X, Y, cluster_num)
        print('wine dataset:')
        external_measurement(X2, Y2, cluster_num)


