from sklearn.datasets import load_wine
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def scatter_plot_df(target_list):
    fig, ax = plt.subplots()
    df1 = target_list[target_list['target'] == 1]
    df2 = target_list[target_list['target'] == 2]
    ax.set_xlabel('Alcohol')
    ax.set_ylabel("Magnesium")
    ax.scatter(df1['alcohol'], df1['magnesium'], c='r')
    ax.scatter(df2['alcohol'], df2['magnesium'], c='y')
    ax.legend(labels=['Winery 1', 'Winery 2'])
    plt.title("Scatter plot of Alcohol and Magnesium levels")
    plt.show()


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='purple', levels=[-1, 0, 1], alpha=1,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=50, linewidth=2, facecolors='none', edgecolor='black')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Alcohol')
    ax.set_ylabel("Magnesium")
    plt.show()


def main():
    # Read the wine dataset
    dataset = load_wine()
    df = pd.DataFrame(data=dataset['data'], columns=dataset['feature_names'])
    df = df.assign(target=pd.Series(dataset['target']).values)

    # Filter the irrelevant columns
    df = df[['alcohol', 'magnesium', 'target']]
    # Filter the irrelevant label
    df = df[df.target != 0]
    train_df, val_df = train_test_split(df, test_size=30, random_state=3)

    # part 1:
    scatter_plot_df(train_df)
    scatter_plot_df(val_df)

    # Parts 2-5:
    train_list = []
    val_list = []
    for i in range(len(train_df['alcohol'].tolist())):
        train_list.append(np.array((train_df['alcohol'].tolist()[i], train_df['magnesium'].tolist()[i])))
    for i in range(len(val_df['alcohol'].tolist())):
        val_list.append(np.array((val_df['alcohol'].tolist()[i], val_df['magnesium'].tolist()[i])))
    c = [0.01, 0.05, 0.1]
    margin_size_train = []
    margin_size_val = []
    error_rate_train = []
    error_rate_val = []
    for i in c:
        model1 = SVC(kernel='linear', C=i)
        model1.fit(train_list, train_df['target'].tolist())
        error_rate_train.append(1-model1.score(train_list, train_df['target'].tolist()))
        error_rate_val.append(1-model1.score(val_list, val_df['target'].tolist()))
        margin_size_train.append(1/np.linalg.norm(model1.coef_))
        plt.title("C =" + str(i) + " on Train Set")
        plt.scatter(train_df['alcohol'].tolist(), train_df['magnesium'].tolist(),
                    c=train_df['target'].tolist(), s=50, cmap='autumn')
        plot_svc_decision_function(model1)
        model2 = SVC(kernel='linear', C=i)
        model2.fit(val_list, val_df['target'].tolist())
        margin_size_val.append(1/np.linalg.norm(model2.coef_))
        plt.scatter(val_df['alcohol'], val_df['magnesium'],
                    c=val_df['target'], s=50, cmap='autumn')
        plt.title("C =" + str(i) + " on Validation Set")
        plot_svc_decision_function(model2)
    # part 4 print
    plt.plot(c, margin_size_train, label="Train set")
    plt.plot(c, margin_size_val, label="Validation set")
    plt.xlabel("C")
    plt.ylabel("Margin Size")
    plt.title("Margin Size to C ratio")
    plt.legend()
    plt.show()

    # part 5 print
    plt.plot(c, error_rate_train, label="Train set")
    plt.plot(c, error_rate_val, label="Validation set")
    plt.xlabel("C")
    plt.ylabel("Error Rate")
    plt.title("Error Rate to C ratio")
    plt.legend()
    plt.show()

    # part 6 + 7
    degree = [i for i in range(2, 9)]
    error_rate_train = np.zeros([7, ])
    error_rate_val = np.zeros([7, ])
    for deg in degree:
        model = SVC(kernel='poly', C=1, degree=deg)
        model.fit(train_list, train_df['target'].tolist())
        train_error = 1-model.score(train_list, train_df['target'].tolist())
        val_error = 1-model.score(val_list, val_df['target'].tolist())
        error_rate_train[deg-2] = train_error
        error_rate_val[deg-2] = val_error
    max_min_train = [np.argmax(error_rate_train)+2, np.argmin(error_rate_train)+2]

    # printing the line graph, and the 4 degrees graphs
    plt.plot(degree, error_rate_train, label="Train set")
    plt.plot(degree, error_rate_val, label="Validation set")
    plt.xlabel("Degree")
    plt.ylabel("Error Rate")
    plt.title("Error Rate to Degree of polynomial ratio")
    plt.legend()
    plt.show()

    for i in max_min_train:
        model = SVC(kernel='poly', C=1, degree=i)
        model.fit(train_list, train_df['target'].tolist())
        plt.scatter(train_df['alcohol'].tolist(), train_df['magnesium'].tolist(),
                    c=train_df['target'].tolist(), s=50, cmap='autumn')
        plt.title("degree = " + str(i) + " on Training Set")
        plot_svc_decision_function(model, plot_support=False)
        plt.scatter(val_df['alcohol'].tolist(), val_df['magnesium'].tolist(),
                    c=val_df['target'].tolist(), s=50, cmap='autumn')
        plt.title("degree = " + str(i) + " on Validation Set")
        plot_svc_decision_function(model, plot_support=False)


if __name__ == '__main__':
    main()
