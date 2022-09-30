from sklearn.neighbors import KNeighborsClassifier


def knnexec(k, trainset, testset, is_train):
    """
    Create a K-NN classifier, using the train set, and predicts the test set or the train set.
    :param k: parameter K in the K-NN classifier
    :param trainset: A list of size two. index 0 contain all the train samples, and index 1 contains all their labels.
    :param testset: A list of size two. index 0 contain all the test samples, and index 1 contains all their labels.
    :param is_train: If is_train == 1, the function will try to predict the train samples labels.
    else, it will predict the test samples labels.
    :return: the error rate of prediction as mentioned in the assignment.
    """
    test_size = len(testset[0])
    train_size = len(trainset[0])
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(trainset[0], trainset[1])
    error = 0
    if is_train == 1:
        pred = classifier.predict(trainset[0])
        for i in range(train_size):
            if pred[i] != trainset[1][i]:
                error += 1
        return error / train_size
    else:
        pred = classifier.predict(testset[0])
        for i in range(test_size):
            if pred[i] != testset[1][i]:
                error += 1
        return error / test_size
