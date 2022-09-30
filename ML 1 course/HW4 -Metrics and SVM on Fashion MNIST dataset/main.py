import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def cross_validation_error(X, y, model, folds):
    samples_sub_arrays = np.array_split(X, folds)
    labels_sub_arrays = np.array_split(y, folds)
    error_k_train = []
    error_k_test = []
    for k in range(folds):
        # Split the array into sub-arrays of approximately the same size.
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


def svm_results(X_train, y_train, X_test, y_test):
    error_results = {}
    error = 0
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            error = error + 1
    test_error = error/len(y_test)
    CV_error = cross_validation_error(X_train, y_train, model, 4)
    error_results['SVM_linear'] = (CV_error[0], CV_error[1], test_error)
    d_list = {2, 4, 6, 8}
    for d in d_list:
        model = SVC(kernel='poly', degree=d)
        error = 0
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        for i in range(len(y_test)):
            if y_test[i] != y_pred[i]:
                error = error + 1
        test_error = error/len(y_test)
        CV_error = cross_validation_error(X_train, y_train, model, 4)
        error_results[f'SVM_poly_{d}'] = (CV_error[0], CV_error[1], test_error)
    rbf_values = [0.001, 0.01, 0.1, 1.0, 10]
    for val in rbf_values:
        model = SVC(kernel='rbf', gamma=val)
        error = 0
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        for i in range(len(y_test)):
            if y_test[i] != y_pred[i]:
                error = error + 1
        test_error = error / len(y_test)
        CV_error = cross_validation_error(X_train, y_train, model, 4)
        error_results[f'SVM_poly_{val}'] = (CV_error[0], CV_error[1], test_error)
    return error_results


def fetch_mnist():
    # Download MNIST dataset
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True)
    X = X.to_numpy()
    y = y.to_numpy()

    # Randomly sample 7000 images
    np.random.seed(2)
    indices = np.random.choice(len(X), 7000, replace=False)
    X, y = X[indices], y[indices]
    return X, y


def main():
    X, y = fetch_mnist()
    print(X.shape, y.shape)
    idx2class = {'0': 'T-shirt/top', '1': 'Trouser', '2': 'Pullover', '3': 'Dress', '4': 'Coat', '5': 'Sandal',
                 '6': 'Shirt', '7': 'Sneaker', '8': 'Bag', '9': 'Ankle'}
    for i in range(10):
        image = X[i].reshape(28, 28)
        s = idx2class[y[i]]
        plt.title(f'{y[i]}, {s}')
        plt.imshow(image, cmap='binary')
        plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    error_res = svm_results(X_train, y_train, X_test, y_test)

    train_cv_error = []
    test_cv_error = []
    test_error = []
    labels_names = []
    index_range = range(len(error_res.keys()))
    for key in error_res.keys():
        train_cv_error.append(error_res[key][0])
        test_cv_error.append(error_res[key][1])
        test_error.append(error_res[key][2])
        labels_names.append(key)

    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Error Rate")
    width = 0.2
    ax1.bar([val - width for val in index_range], train_cv_error, width=width, color='blue', label='Train CV Error')
    ax1.bar([val for val in index_range], test_cv_error, width=width, color='red', label='Test CV Error')
    ax1.bar([val + width for val in index_range], test_error, width=width, color='green', label='Test Error')
    ax1.set_xticks(index_range, labels_names, rotation=90)
    plt.title("Comparison between different models")
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()