import matplotlib.pyplot as plt
import numpy as np


def grad_f(x):
    return 2 * x + 2


def grad_update(x, z):
    return x - z * grad_f(x)


def main():
    x = np.linspace(-10, 10, 1000)
    y = 1*(x ** 2) + 2 * x + 3
    fig, ax = plt.subplots()
    plt.title("Graph for a=3, b=2, c=1")
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot(x, y)
    plt.show()
    # part 1-6
    z = 0.1
    y = 0
    x = 10
    epsilon = 0.000001
    x_t = []
    while abs(y-x) > epsilon:
        x_t.append(x)
        y = x
        x = grad_update(x, z)
    print(x)

    # part 1-7
    z_winner = [0, (100*22 ** 2) / epsilon ** 2]
    for z1 in np.arange(0.0, 1.0, 0.1):
        x = 10
        y = 0
        x_t = []
        while abs(y - x) > epsilon:
            x_t.append(x)
            y = x
            x = grad_update(x, z)
        if len(x_t) <= z_winner[1]:
            z_winner = [z1, len(x_t)]

    y = 0
    x = 10
    x_t = []
    while abs(y-x) > epsilon:
        x_t.append(x)
        y = x
        x = grad_update(x, z_winner[0])
    f = [i**2 + 2*i + 3 for i in x_t]
    x = np.linspace(-10, 10, 1000)
    y = 1*(x ** 2) + 2 * x + 3
    plt.title("Graph for a=3, b=2, c=1")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, c='g')
    plt.scatter(x_t, f, c='r', s=30)
    plt.show()


if __name__ == '__main__':
    main()
