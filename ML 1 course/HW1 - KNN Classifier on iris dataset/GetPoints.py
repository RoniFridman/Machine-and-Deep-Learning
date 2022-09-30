import numpy as np
from CalcDistance import calculate_distance


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
