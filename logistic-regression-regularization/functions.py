import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def computeCost(X, y, theta, lambda_=0, regularization='none'):
    h = np.clip(sigmoid(X @ theta), 1e-15, 1 - 1e-15)

    logloss = -1 / len(y) * (
        np.dot(y, np.log(h))
        + np.dot(1 - y, np.log(1 - h))
    )

    if regularization == 'l2':
        logloss += lambda_ / (2 * len(y)) * np.sum(theta[1:] ** 2)

    elif regularization == 'l1':
        logloss += lambda_ / len(y) * np.sum(np.abs(theta[1:]))

    elif regularization == 'elastic':
        logloss += (
            lambda_ / len(y) * np.sum(np.abs(theta[1:]))
            + lambda_ / (2 * len(y)) * np.sum(theta[1:] ** 2)
        )

    return logloss


def gradientDescent(X, y, theta, alpha, lambda_=0, regularization='none'):
    gradient = X.T @ (sigmoid(X @ theta) - y) / len(y)

    if regularization == 'l2':
        gradient[1:] += lambda_ / len(y) * theta[1:]

    elif regularization == 'l1':
        gradient[1:] += lambda_ / len(y) * np.sign(theta[1:])

    elif regularization == 'elastic':
        gradient[1:] += lambda_ / len(y) * (
            np.sign(theta[1:]) + theta[1:]
        )

    return theta - alpha * gradient

def plotData(X, y):
    class0 = X[y == 0]
    class1 = X[y == 1]

    plt.scatter(class0[:, 0], class0[:, 1], marker='o', color='blue', label='Исправен')
    plt.scatter(class1[:, 0], class1[:, 1], marker='+', color='red', label='Неисправен')
    plt.legend(fontsize=8)

    plt.show()


def plotDecisionBoundary(X, y, theta):
    class0 = X[y == 0]
    class1 = X[y == 1]
    plt.scatter(class0[:, 1], class0[:, 2], marker='o', color='blue', label='Исправен')
    plt.scatter(class1[:, 1], class1[:, 2], marker='+', color='red', label='Неисправен')

    x1_values = np.linspace(-2, 2, 100)
    x2_values = np.linspace(-2, 2, 100)

    xx, yy = np.meshgrid(x1_values, x2_values)
    x1 = xx.ravel()
    x2 = yy.ravel()

    features = [
        np.ones(len(x1)),
        x1, x2,
        x1**2, x1*x2, x2**2,
        x1**3, x1**2*x2, x1*x2**2, x2**3,
        x1**4, x1**3*x2, x1**2*x2**2, x1*x2**3, x2**4,
        x1**5, x1**4*x2, x1**3*x2**2, x1**2*x2**3, x1*x2**4, x2**5,
        x1**6, x1**5*x2, x1**4*x2**2, x1**3*x2**3, x1**2*x2**4, x1*x2**5, x2**6
    ]

    grid_features = np.column_stack(features)

    z = (grid_features @ theta).reshape(xx.shape)

    plt.contourf(
        xx, yy, z,
        levels=[-np.inf, 0, np.inf],
        colors=['red', 'green'],
        alpha=0.15
    )

    plt.contour(xx, yy, z, levels=[0], colors='black')
    plt.legend(fontsize=8)

def predict(X, theta, mean, std):

    X = (X - mean) / std
    x1 = X[:, 0]
    x2 = X[:, 1]

    features = [
        np.ones(len(x1)),
        x1, x2,
        x1 ** 2, x1 * x2, x2 ** 2,
        x1 ** 3, x1 ** 2 * x2, x1 * x2 ** 2, x2 ** 3,
        x1 ** 4, x1 ** 3 * x2, x1 ** 2 * x2 ** 2, x1 * x2 ** 3, x2 ** 4,
        x1 ** 5, x1 ** 4 * x2, x1 ** 3 * x2 ** 2, x1 ** 2 * x2 ** 3, x1 * x2 ** 4, x2 ** 5,
        x1 ** 6, x1 ** 5 * x2, x1 ** 4 * x2 ** 2, x1 ** 3 * x2 ** 3, x1 ** 2 * x2 ** 4, x1 * x2 ** 5, x2 ** 6
    ]

    features = np.column_stack(features)


    probabilities = sigmoid(features @ theta)
    classes = (probabilities >= 0.5).astype(int)

    return probabilities, classes




