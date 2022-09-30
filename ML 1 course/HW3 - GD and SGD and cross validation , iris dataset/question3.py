import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def cross_validation_error(X, y, model, folds):
    samples_sub_arrays = np.array_split(X, folds)
    labels_sub_arrays = np.array_split(y, folds)
    error_k_train = []
    error_k_test = []
    for k in range(folds):
        # Split the array into sub-arrays of approximatlly the same size.
        # Choose the k-th array as the test array, and combine all others as the training set.
        X_test = samples_sub_arrays.pop(k)
        y_test = labels_sub_arrays.pop(k)
        X_train = np.vstack(samples_sub_arrays)
        y_train = np.concatenate(labels_sub_arrays)
        model.fit(X_train, y_train)
        # Calculate error on training set
        y_pred_train = model.predict(X_train)
        error_count = 0
        for i in range(len(y_train)):
            if y_pred_train[i] != y_train[i]:
                error_count += 1
        error_k_train.append(error_count / len(y_train))
        # Calculate error on test set
        y_pred_test = model.predict(X_test)
        error_count = 0
        for i in range(len(y_pred_test)):
            if y_pred_test[i] != y_test[i]:
                error_count += 1
        error_k_test.append(error_count / len(y_test))
        # Return the chosen sub-arrays back with all training points
        samples_sub_arrays.insert(k, X_test)
        labels_sub_arrays.insert(k, y_test)
    avg_error = (sum(error_k_train)/folds, sum(error_k_test)/folds)
    return avg_error


def svm_results(X_train, X_test, y_train, y_test):
    lambdas = [0.0001, 0.01, 1, 100, 10000]
    error_dict = {}
    for lam in lambdas:
        model = SVC(kernel='linear', C=1/lam)
        avg_error = cross_validation_error(X_train, y_train, model, 5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error_count = 0
        for i in range(len(y_test)):
            if y_pred[i]!=y_test[i]:
                error_count += 1
        error_dict["SVM_lambda_" + str(lam)] = (avg_error[0], avg_error[1], error_count/len(y_test))
    return error_dict



def main():
    iris_data = load_iris()
    X, y = iris_data['data'], iris_data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    lambdas = [0.0001, 0.01, 1, 100, 10000]
    error_dict = svm_results(X_train, X_test, y_train, y_test)
    train_error = []
    val_error = []
    test_error = []
    for key in error_dict.keys():
        train_error.append(error_dict[key][0])
        val_error.append(error_dict[key][1])
        test_error.append(error_dict[key][2])

    lam_rescale = [np.log10(l) for l in lambdas]
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("Lambda(log)")
    ax1.set_ylabel("Error Rate")
    width = 0.5
    ax1.bar([lam-width for lam in lam_rescale], train_error, width=width, color='purple', label='Train')
    ax1.bar([lam for lam in lam_rescale], val_error, width=width, color='orchid', label='Validation')
    ax1.bar([lam+width for lam in lam_rescale], test_error, width=width, color='plum', label='Test')
    ax1.set_xticks(lam_rescale)
    plt.title("Correlational between lambda and error rate")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
