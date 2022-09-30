import numpy as np
import matplotlib.pyplot as plt


def generate_samples(num, isplot):
    """
    Generating samples. A random number in [0,1] chooses which mu-parameter we'll use for the normal distribution.
    :param num: number of samples we want to generate.
    :param isplot: if isplot == 1, it will plot all the samples generated as dots.
    else it will not plot it.
    :return: a list of size 2, when index 0 contains all the samples, and index 1 contains all their labels.
    """
    mu1 = [-1, 1]
    mu2 = [-2.5, 2.5]
    mu3 = [-4.5, 4.5]
    sigma = [[1, 0], [0, 1]]
    test_group = []
    test_group_labels = []
    for i in range(num):
        pickmu = np.random.rand()
        if pickmu < 1 / 3:
            mean = mu1
            label = 1
        elif pickmu > 2 / 3:
            mean = mu3
            label = 3
        else:
            mean = mu2
            label = 2

        test_group.append(np.random.multivariate_normal(mean, sigma))
        test_group_labels.append(label)
    if isplot == 1:
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter([i[0] for i in test_group], [i[1] for i in test_group],
                   c=test_group_labels)
        plt.show()

    return [test_group, test_group_labels]
