import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split




def svm_with_sgd(X, y, lam=0.0, epochs=1000, l_rate=0.01, sgd_type='practical'):
    np.random.seed(2)
    m = np.shape(X)[0]
    d = np.shape(X)[1]
    w = np.random.rand(d,)
    b = np.random.rand()
    if sgd_type == 'practical':
        for j in range(epochs):
            perm = np.random.permutation(m)
            for i in perm:
                flag = 1-y[i]*(np.inner(X[i], w)+b)
                if flag <= 0:
                    dw = lam * w * 2
                    db = 0
                else:
                    dw = lam * w * 2 - X[i] * y[i]
                    db = -y[i]

                w = w - l_rate * dw
                b = b - l_rate * db
        return w, b

    if sgd_type == 'theory':
        w_avg = w / (epochs*m)
        b_avg = b / (epochs*m)
        for j in range(m*epochs):
            i = np.random.randint(0, m)
            flag = 1 - y[i]*(np.inner(X[i], w)+b)
            if flag <= 0:
                dw = lam * w * 2
                db = 0
            else:
                dw = lam * w * 2 - X[i] * y[i]
                db = -y[i]
            w = w - l_rate * dw
            b = b - l_rate * db

            w_avg += w / (epochs * m)
            b_avg += b / (epochs * m)
        return w_avg, b_avg


def calculate_error(X, y, w, b):
    m = np.shape(X)[0]
    error_count = 0
    y_predicted = []
    for i in range(m):
        flag = np.inner(X[i, ], w) + b
        if flag >= 0:
            y_predicted.append(1)
        else:
            y_predicted.append(-1)

    for i in range(m):
        if y_predicted[i] != y[i]:
            error_count += 1

    return error_count/m


def main():
    X, y = load_iris(return_X_y=True)
    X = X[y != 0]
    y = y[y != 0]
    y[y == 2] = -1
    X = X[:, 2:4]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)
    lambdas = [0, 0.05, 0.1, 0.2, 0.5]
    models = []
    margins = []
    for lam in range(len(lambdas)):
        w, b = svm_with_sgd(X_train, y_train, lambdas[lam])
        margin = 1/np.linalg.norm(w)
        margins.append(margin)
        models.append([w, b])
    train_error = []
    val_error = []
    for i in range(len(models)):
        train_error.append(calculate_error(X_train, y_train, models[i][0], models[i][1]))
        val_error.append(calculate_error(X_val, y_val, models[i][0], models[i][1]))

    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("Lambda")
    ax1.set_ylabel("Error Rate")
    ax1.bar([l-0.02/2 for l in lambdas], train_error, width=0.02, color='r', label='Train')
    ax1.bar([l + 0.02 / 2 for l in lambdas], val_error, width=0.02, color='g', label='Validation')
    ax1.set_xticks(lambdas)
    plt.title("Correlational between lambda and error rate for both sets")
    plt.legend()
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.set_xlabel("Lambda")
    ax2.set_ylabel("Margin Size")
    ax2.bar(lambdas, margins, width=0.025, color='b')
    ax2.set_xticks(lambdas)
    plt.title("Correlational between lambda and margin size")
    plt.show()

    epoch_list = range(10, 1000, 10)
    lam1 = 0.05
    train_error_practical = []
    train_error_theory = []
    val_error_practical = []
    val_error_theory = []
    for epoch in epoch_list:
        w_p, b_p = svm_with_sgd(X_train, y_train, epochs=epoch, lam=lam1)
        w_th, b_th = svm_with_sgd(X_train, y_train, epochs=epoch, lam=lam1, sgd_type='theory')
        train_error_practical.append(calculate_error(X_train, y_train, w_p, b_p))
        train_error_theory.append(calculate_error(X_train, y_train, w_th, b_th))

    plt.xlabel("Epochs")
    plt.ylabel("Error Rate on Training Set")
    plt.plot(epoch_list, train_error_practical, color='b', label='Practical')
    plt.plot(epoch_list, train_error_theory, color='g', label='Theory')
    plt.title("Correlation between error on training set and epochs size")
    plt.legend()
    plt.show()

    for epoch in epoch_list:
        w_p, b_p = svm_with_sgd(X_train, y_train, epochs=epoch, lam=lam1)
        w_th, b_th = svm_with_sgd(X_train, y_train, epochs=epoch, lam=lam1, sgd_type='theory')
        val_error_practical.append(calculate_error(X_val, y_val, w_p, b_p))
        val_error_theory.append(calculate_error(X_val, y_val, w_th, b_th))

    plt.xlabel("Epochs")
    plt.ylabel("Error Rate on Validation Set")
    plt.plot(epoch_list, val_error_practical, color='b', label='Practical')
    plt.plot(epoch_list, val_error_theory, color='g', label='Theory')
    plt.title("Correlation between error on validation set and epochs size")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
