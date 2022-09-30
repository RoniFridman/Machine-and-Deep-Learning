import argparse
import numpy as np
import pandas as pd


def calculate_distance(x: np.ndarray, y: np.ndarray, p):
    """
    The function calculates the distance between two vectors using L(P) norm.
    :param x: first vector
    :param y: second vector
    :param p: the L norm that will be used
    :return: the distance between the vectors as a scalar.
    """
    distance = np.sqrt(np.sum(np.power(np.abs((x-y)), p)))
    return distance


def get_points(train_points, test_point, k, p):
    """
    The function returns the indexes of the K nearest neighbors of the test point.
    :param train_points: A list of all the training points.
    :param test_point: the test point we want to classify
    :param k: number of neighbors
    :param p: the L norm that will be used to calculate the distance between points.
    :return: a numpy column vector containing k indexes of the nearest neighbors.
            the indexes are sorted in a way that the first index is the closest point,
            while the last one is the farthest point from the test point.
    """
    dist_from_point = np.zeros((len(train_points), ))
    for i in range(len(train_points)):
        distance = calculate_distance(train_points[i], test_point, p)
        dist_from_point[i] = distance
    dist_from_point = np.argsort(dist_from_point, axis=0)[:k]
    return dist_from_point


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p
        self.X = []
        self.y = []
        self.labels_num = 0

        # TODO - Place your student IDs here. Single submitters please use a tuple like so: self.ids = (123456789,)
        self.ids = (205517097, 205451123)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """
        size_of_sample = X.shape[0]
        self.X = np.split(X, size_of_sample)
        self.y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """

        # splitting the test set matrix by rows
        size_of_sample = X.shape[0]
        test_set = np.split(X, size_of_sample)
        labels_predicted = np.zeros((len(test_set),), dtype=np.uint8)

        # Running a for loop, to classify every test point separately
        for i in range(len(test_set)):
            # We get the indexes of the closest neighbors using get_points,
            # and then we get their labels using the labels saved in self.y
            neighbors_indexes = get_points(self.X, test_set[i], self.k, self.p)
            neighbors_labels = np.ndarray((neighbors_indexes.shape[0], ), dtype=np.uint8)
            for j in range(self.k):
                index = neighbors_indexes.item(j)
                neighbors_labels[j] = self.y[index]

            # We'll count the number of labels, and save the max value that appears in the count.
            # If more than one label occurs 'max_val' times, we add it to a list called "most_common_labels"
            # If the list size is 1, then we choose the label in it.
            labels_count = np.bincount(neighbors_labels)
            max_val = np.amax(labels_count)
            most_common_labels = []
            for j in range(len(labels_count)):
                if labels_count[j] == max_val:
                    most_common_labels.append(j)
            if len(most_common_labels) == 1:
                labels_predicted[i] = most_common_labels[0]

            # In case the list size is bigger than 1, we need a tie breaker.
            # Since the labels came from indexes that were already sorted in ascending order,
            # we choose the label that is with the smallest value AND appears in the list "most_common_labels"
            else:
                for j in range(len(neighbors_labels)):
                    if neighbors_labels[j] in most_common_labels:
                        labels_predicted[i] = neighbors_labels[j]
                        break
        return labels_predicted


def main():

    print("*" * 20)
    print("Started HW1_205517097_205451123.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
