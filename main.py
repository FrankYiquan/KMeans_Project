from collections import defaultdict, Counter
import math
import numpy as np
import random



class MyKMeans:
    def __init__(self, k,  x, random_state: int, true_label, max_iter = 100, min_check = 1e-4):
        #value check


        # Ensure x is a NumPy array with each row being a data point
        self.X = np.asarray(x)

        # Check if X is a 2D array
        if self.X.ndim != 2:
            raise ValueError("x must be a 2D array where each row is a data point.")

        # Validate that the number of data points matches the true_labels
        if len(self.X) != len(true_label):
            raise ValueError("The number of data points in x must match the number of true labels.")


        self.k = k
        self.random_state = random_state
        random.seed(self.random_state)
        #store coordinates
        #store intermediate y or label
        self.Y = defaultdict(list)
        self.centroids = np.array([])
        self.old_centroids = np.array([])
        #optional parameters
        self.max_iter = max_iter
        self.minCheck = min_check
        # evaluation
        self.cluster_to_data = defaultdict(list)
        self.cluster_to_predicted_label = defaultdict(list)
        self.cluster_to_true_labels = defaultdict(list)
        self.true_label = true_label #should be a list


    # perform cluster anaylysis
    def train(self, method = "KMean"):
        random.seed(self.random_state)
        if method == "KMean++":
            self.strategically_choose_centroid()
        #normal Kmean
        else:
            self.pick_random_centroid(self.k)
        # loop until converage or reach max_iter
        for i in range(self.max_iter):
            self.assign_clusters()
            self.reassign_clusters()
            #check if centroid don't change
            if np.all(np.abs(np.array(self.centroids) - np.array(self.old_centroids)) < self.minCheck):
                break

        # used to calculate NMI and silhouette_score
        i = 0
        for data_point, centroid in self.Y.items():
            self.cluster_to_data[tuple(centroid)].append(data_point)
            self.cluster_to_true_labels[tuple(centroid)].append(self.true_label[i])
            i += 1

        for cluster, true_labels in self.cluster_to_true_labels.items():
            self.cluster_to_predicted_label[cluster] = Counter(true_labels).most_common(1)[0][0]



        return dict(self.cluster_to_data)

    def predict(self):
        return dict(self.cluster_to_data)

    #use for kMean++ centroid initialization
    def strategically_choose_centroid(self):
        np.random.seed(self.random_state)  # Ensure reproducibility for NumPy's random functions
        # Pick the first centroid randomly
        self.pick_random_centroid(1)

        # Stop until we have enough k centroids
        while len(self.centroids) < self.k:
            max_distance = []

            for data in self.X:
                # Get distance between data point and the closest centroid
                min_dist = np.min([self.euclidean_distance(data, centroid) for centroid in self.centroids])
                max_distance.append(min_dist)

            # Compute probability
            total_sum = sum(max_distance)  # Sum of distances
            normalized_list = [x / total_sum for x in max_distance]

            # Choose a point based on probabilities using NumPy
            selected_index = np.random.choice(len(self.X), size=1, p=normalized_list)[0]

            # Next centroid is selected
            new_centroid = self.X[selected_index]

            # Append the new centroid to the list of centroids as a row (2D)
            self.centroids = np.append(self.centroids, new_centroid[np.newaxis, :], axis=0)

            max_distance.clear()

    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

    # choose k random data points from dataset as initial k centroids
    def pick_random_centroid(self, k: int):
        np.random.seed(self.random_state)
        self.centroids = self.X[np.random.choice(self.X.shape[0], size=k, replace=False)]

    #assign closest centroid to be the cluster of each datapoint
    def assign_clusters(self):
        for data in self.X:
            distance = []
            for centroid in self.centroids:
                distance.append(self.euclidean_distance(data, centroid))
            self.Y[tuple(data)] = self.centroids[np.argmin(distance)].tolist()

    # update centroid by computing the mean of data points within the cluster
    def reassign_clusters(self):
        track = defaultdict(list)
        new_centroids = []

        for data, cluster in self.Y.items():
            track[tuple(cluster)].append(data)

        for cluster_points in track.values():
            new_centroids.append(np.mean(cluster_points, axis=0))

        self.old_centroids = self.centroids
        self.centroids = new_centroids

    #silhouette_score
    def silhouette_score(self):
        silhouette_values = defaultdict(float)

        for data, cluster in self.Y.items():
            # Mean intra-cluster distance
            intra_cluster_distance = self.calculate_point_and_cluster_distance(data, cluster)

            # Find the nearest other cluster's distance
            outer_cluster_distance = np.min([
                self.calculate_point_and_cluster_distance(data, centroid)
                for centroid in self.centroids
                if not np.array_equal(centroid, cluster)
            ])


            denominator = max(outer_cluster_distance, intra_cluster_distance)
            silhouette_value = (outer_cluster_distance - intra_cluster_distance) / denominator if denominator != 0 else 0.0


            silhouette_values[tuple(data)] = silhouette_value

        avg_silhouette = np.mean(list(silhouette_values.values()))

        return avg_silhouette, silhouette_values

    #helper method for silhouette_score
    def calculate_point_and_cluster_distance(self, data, cluster):
        data_in_cluster = self.cluster_to_data[tuple(cluster)]
        differences = [self.euclidean_distance(np.array(data), np.array(cluster_vector)) for cluster_vector in data_in_cluster]
        data_cluster_distance = np.mean(differences)
        return data_cluster_distance

    # helper method for NMI score
    def entropy_of_class_labels(self):
        # Count the occurrences of each unique label
        label_counts = Counter(self.true_label)

        total_labels = len(self.true_label)

        # entropy
        entropy = 0
        for count in label_counts.values():
            prob = count / total_labels
            if prob > 0:
                entropy -= prob * math.log2(prob)  # Use log base 2

        return entropy

    # helper method for NMI score
    def entropy_of_cluster_labels(self):
        entropy = 0
        total_labels = len(self.true_label)

        for centroid, data_points in self.cluster_to_data.items():
            cluster_size = len(data_points)
            cluster_probability = cluster_size / total_labels

            if cluster_probability > 0:
                entropy -= cluster_probability * math.log2(cluster_probability)

        return entropy

    # helper method for NMI score
    def mutual_information(self):
        entropy_sum = 0
        total_labels = len(self.true_label)

        for centroid, data_points in self.cluster_to_data.items():
            entropy = 0
            cluster_size = len(data_points)
            cluster_probability = cluster_size / total_labels #P(Y| C = c),
            label_count_in_cluster = Counter(self.cluster_to_true_labels[tuple(centroid)])
            for label_count in label_count_in_cluster.values():
                prob = label_count / cluster_size #P(Y=y | C)
                if prob > 0:
                    entropy += prob * math.log2(prob)
            entropy *= -cluster_probability
            entropy_sum += entropy

        return self.entropy_of_class_labels() - entropy_sum

    # NMI score
    def normalized_mutual_information_score(self):
        return 2 * self.mutual_information() / abs(self.entropy_of_class_labels() + self.entropy_of_cluster_labels())









