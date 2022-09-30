import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class PerceptronClassifier:
    def __init__(self):
        """
        Constructor for the PerceptronClassifier.
        """
        self.ids = (205517097, 205451123)
        self.model = []
        self.labels = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
        Array datatype is guaranteed to be np.uint8.
        """

        self.labels = np.unique(y).tolist()
        n = X.shape[0]
        att = X.shape[1]
        for i in range(len(self.labels)):
            self.model.append(np.zeros(att))
        sample = 0
        while sample < n:
            inner_prod = np.zeros(len(self.labels))
            for w in range(len(self.labels)):
                inner_prod[w] = (np.inner(self.model[w], X[sample]))
            pred_label = np.argmax(inner_prod)
            correct_label = np.where(self.labels == y[sample])[0].item(0)
            if self.labels[pred_label] != y[sample]:
                self.model[pred_label] = self.model[pred_label] - X[sample]
                self.model[correct_label] = self.model[correct_label] + X[sample]
                sample = 0
            sample = sample + 1


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call PerceptronClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        n = X.shape[0]
        y_pred1 = np.zeros((n,))
        for i in range(n):
            if i==50:
                print("hello")
            inner_prod = np.zeros(len(self.labels))
            for w in range(len(self.labels)):
                inner_prod[w] = np.inner(self.model[w], X[i])
            y_pred1[i] = self.labels[np.argmax(inner_prod)]

        return y_pred1



if __name__ == "__main__":

    print("*" * 20)
    print("Started HW2_205517097_205451123.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    is_separable = model.fit(X, y)
    print("Done")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y.ravel()) / y.shape[0]
    print(f"Train accuracy: {accuracy * 100 :.2f}%")

    print("*" * 20)
