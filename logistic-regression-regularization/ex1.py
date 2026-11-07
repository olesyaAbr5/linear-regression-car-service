import numpy as np
from matplotlib import pyplot as plt


from functions import computeCost, gradientDescent, plotData, sigmoid, plotDecisionBoundary, predict

data = np.loadtxt('ex2data2.txt', delimiter=',')

X = data[:, :2]
y = data[:, 2].astype(int)
plotData(X, y)


mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std
plotData(X, y)


x1 = X[:, 0]
x2 = X[:, 1]


features = [
    np.ones(len(x1)),
    x1, x2,
    x1**2, x1*x2, x2**2,
    x1**3, x1**2*x2, x1*x2**2, x2**3,
    x1**4, x1**3*x2, x1**2*x2**2, x1*x2**3, x2**4,
    x1**5, x1**4*x2, x1**3*x2**2, x1**2*x2**3, x1*x2**4, x2**5,
    x1**6, x1**5*x2, x1**4*x2**2, x1**3*x2**3, x1**2*x2**4, x1*x2**5, x2**6
]

X_new = np.column_stack(features)

regularizations = ['l1', 'l2', 'elastic']
lambdas_regularized = [0.001, 1, 5, 50, 100]
fig, axes = plt.subplots(
    len(regularizations), len(lambdas_regularized),
    figsize=(20, 12), sharex=True, sharey=True,
    constrained_layout=True
)
fig.suptitle('Сравнение регуляризаций')


for regularization in ['без регуляризации', 'l1', 'l2', 'elastic']:
    if regularization == 'без регуляризации':
        lambdas = [0]
    else:
        lambdas = lambdas_regularized

    for col, lambda_ in enumerate(lambdas):
        theta = np.zeros(X_new.shape[1])

        for i in range(100000):
            theta = gradientDescent(
                X_new, y, theta, 0.05,
                lambda_=lambda_,
                regularization=regularization
            )

        if regularization == 'l2' and lambda_ == 1:
            theta_best = theta.copy()

        probabilities = sigmoid(X_new @ theta)
        classes = (probabilities >= 0.5).astype(int)

        accuracy = np.mean(classes == y)
        logloss = computeCost(X_new, y, theta, regularization='none')

        print(regularization, lambda_, 'accuracy:', accuracy, 'logloss:', logloss)

        if regularization == 'без регуляризации':
            plt.figure(figsize=(7, 6))
        else:
            row = regularizations.index(regularization)
            plt.sca(axes[row, col])

        plt.title(f'{regularization}, λ={lambda_}\nAccuracy: {accuracy:.1%}')
        plotDecisionBoundary(X_new, y, theta)



chip = np.array([[0.72523, 0.44371]])
probabilities, classes = predict(chip, theta_best, mean, std)

print('Вероятность неисправности:', probabilities[0])
print('Результат:', 'неисправен' if classes[0] == 1 else 'исправен')


plt.show()