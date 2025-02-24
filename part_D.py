import pandas as pd
import matplotlib.pyplot as plt
from part_C1 import internal_measurement
from part_C3 import external_measurement

if __name__ == '__main__':
    iris_df = pd.read_csv('iris/iris.data', header=None)
    wine_df = pd.read_csv('wine/wine.data', header=None)

    iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    Y = iris_df['species']

    X2 = wine_df.iloc[:, 1:]
    Y2 = wine_df.iloc[:, 0]

    k2 = range(2, 10)

    #iris dataset
    #internal
    my_sil_internal = []
    real_sil_internal = []
    plus_my_sil_internal = []
    plus_real_sil_internal = []

    #external
    my_NMI_external = []
    real_NMI_external = []
    plus_my_NMI_external = []
    plus_real_NMT_external = []

    #wine dataset
    # internal
    w_my_sil_internal = []
    w_real_sil_internal = []
    w_plus_my_sil_internal = []
    w_plus_real_sil_internal = []

    # external
    w_my_NMI_external = []
    w_real_NMI_external = []
    w_plus_my_NMI_external = []
    w_plus_real_NMT_external = []

    #internel
    for cluster_num in k2:
        print("iris dataset")
        print("KMean++")
        _, my_sil, real_sil = internal_measurement(X,Y, cluster_num, method="KMean++")
        plus_my_sil_internal.append(my_sil)
        plus_real_sil_internal.append(real_sil)

        print("kMean:")
        _, my_sil, real_sil = internal_measurement(X, Y, cluster_num)
        my_sil_internal.append(my_sil)
        real_sil_internal.append(real_sil)

    #externel
    print("----------------------------------")
    for cluster_num in k2:
        print("iris dataset")
        print("KMean++")
        my_NMI, real_NMI = external_measurement(X, Y, cluster_num, method="KMean++")
        plus_my_NMI_external.append(my_NMI)
        plus_real_NMT_external.append(real_NMI)


        print("kMean:")
        my_NMT, real_NMT = external_measurement(X, Y, cluster_num)
        my_NMI_external.append(my_NMT)
        real_NMI_external.append(real_NMT)


    print("----------------------------------")
    print("----------------------------------")
    #external
    for cluster_num in k2:
        print("Wine dataset")
        print("KMean++")
        my_NMI, real_NMI = external_measurement(X2, Y2, cluster_num, method="KMean++")
        w_plus_my_NMI_external.append(my_NMI)
        w_plus_real_NMT_external.append(real_NMI)

        print("kMean:")
        my_NMI, real_NMI = external_measurement(X2, Y2, cluster_num)
        w_my_NMI_external.append(my_NMI)
        w_real_NMI_external.append(real_NMI)
    print("----------------------------------")

    for cluster_num in k2:
        print("Wine dataset")
        print("KMean++")
        _, my_sil, real_sil = internal_measurement(X2, Y2, cluster_num, method="KMean++")
        w_plus_my_sil_internal.append(my_sil)
        w_plus_real_sil_internal.append(real_sil)

        print("kMean:")
        _, my_sil, real_sil = internal_measurement(X2, Y2, cluster_num)
        w_my_sil_internal.append(my_sil)
        w_real_sil_internal.append(real_sil)

    # Create a figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Internal without plus graph (top left)
    axes[0, 0].plot(k2, my_sil_internal, label='Self-implemented', marker='o')
    axes[0, 0].plot(k2, real_sil_internal, label='Built-in', marker='x')
    axes[0, 0].set_title('Self-implemented vs Built-in Silhouette Score on KMeans on Iris')
    axes[0, 0].set_xlabel('k')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].legend()

    # Internal with plus graph (top right)
    axes[1, 0].plot(k2, plus_my_sil_internal, label='Self-implemented', marker='o')
    axes[1, 0].plot(k2, plus_real_sil_internal, label='Built-in', marker='x')
    axes[1, 0].set_title('Self-implemented vs Built-in Silhouette Score on KMeans++ on Iris')
    axes[1, 0].set_xlabel('k')
    axes[1, 0].set_ylabel('Silhouette Score')
    axes[1, 0].legend()

    # External without plus graph (bottom left)
    axes[0, 1].plot(k2, my_NMI_external, label='Self-implemented', marker='o')
    axes[0, 1].plot(k2, real_NMI_external, label='Built-in', marker='x')
    axes[0, 1].set_title('Self-implemented vs Built-in NMI Score on KMeans on Iris')
    axes[0, 1].set_xlabel('k')
    axes[0, 1].set_ylabel('NMI')
    axes[0, 1].legend()

    # External with plus graph (bottom right)
    axes[1, 1].plot(k2, plus_my_NMI_external, label='Self-implemented', marker='o')
    axes[1, 1].plot(k2, plus_real_NMT_external, label='Built-in', marker='x')
    axes[1, 1].set_title('Self-implemented vs Built-in NMI Score on KMeans++ on Iris')
    axes[1, 1].set_xlabel('k')
    axes[1, 1].set_ylabel('NMI')
    axes[1, 1].legend()

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

    # Create a figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Internal without plus graph (top left)
    axes[0, 0].plot(k2, w_my_sil_internal, label='Self-implemented', marker='o')
    axes[0, 0].plot(k2, w_real_sil_internal, label='Built-in', marker='x')
    axes[0, 0].set_title('Self-implemented vs Built-in Silhouette Score on KMeans on Wine')
    axes[0, 0].set_xlabel('k')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].legend()

    # Internal with plus graph (top right)
    axes[1, 0].plot(k2, w_plus_my_sil_internal, label='Self-implemented', marker='o')
    axes[1, 0].plot(k2, w_plus_real_sil_internal, label='Built-in', marker='x')
    axes[1, 0].set_title('Self-implemented vs Built-in Silhouette Score on KMeans++ on Wine')
    axes[1, 0].set_xlabel('k')
    axes[1, 0].set_ylabel('Silhouette Score')
    axes[1, 0].legend()

    # External without plus graph (bottom left)
    axes[0, 1].plot(k2, w_my_NMI_external, label='Self-implemented', marker='o')
    axes[0, 1].plot(k2, w_real_NMI_external, label='Built-in', marker='x')
    axes[0, 1].set_title('Self-implemented vs Built-in NMI Score on KMeans on Wine')
    axes[0, 1].set_xlabel('k')
    axes[0, 1].set_ylabel('NMI')
    axes[0, 1].legend()

    # External with plus graph (bottom right)
    axes[1, 1].plot(k2, w_plus_my_NMI_external, label='Self-implemented', marker='o')
    axes[1, 1].plot(k2, w_plus_real_NMT_external, label='Built-in', marker='x')
    axes[1, 1].set_title('Self-implemented vs Built-in NMI Score on KMeans++ on Wine')
    axes[1, 1].set_xlabel('k')
    axes[1, 1].set_ylabel('NMI')
    axes[1, 1].legend()

    # Adjust layout and show
    plt.tight_layout()
    plt.show()