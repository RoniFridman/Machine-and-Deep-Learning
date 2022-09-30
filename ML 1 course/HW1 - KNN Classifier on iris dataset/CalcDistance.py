import numpy as np


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
