import pandas as pd
import matplotlib.pyplot as plt
from part_C1 import internal_measurement


if __name__ == "__main__":
    wine_df = pd.read_csv('wine/wine.data', header=None)

    X2 = wine_df.iloc[:, 1:]
    Y2 = wine_df.iloc[:, 0]
    k2 = range(2, 10)


    wcss = []
    my_silhouette = []
    real_silhouette = []

    for cluster_num in k2:
        one_wcss, one_my_silhouette, one_real_silhouette = internal_measurement(X2, Y2, cluster_num)
        wcss.append(one_wcss)
        my_silhouette.append(one_my_silhouette)
        real_silhouette .append(one_real_silhouette)


    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot the Elbow Curve (WCSS) on the first subplot
    axes[0].plot(k2, wcss, marker='o', linestyle='--', color='b')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('WCSS')
    axes[0].set_title('Elbow Method for Optimal k')

    # Plot the Silhouette Score (my_silhouette) on the second subplot
    axes[1].plot(k2, my_silhouette, marker='o', linestyle='--', color='g')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('my_silhouette')
    axes[1].set_title('Self-implemented Silhouette Score for Optimal k')

    # Plot the Silhouette Score (real_silhouette) on the third subplot
    axes[2].plot(k2, real_silhouette, marker='o', linestyle='--', color='r')
    axes[2].set_xlabel('Number of Clusters (k)')
    axes[2].set_ylabel('real_silhouette')
    axes[2].set_title('Built-in Silhouette Score for Optimal k')

    plt.tight_layout()
    plt.show()

