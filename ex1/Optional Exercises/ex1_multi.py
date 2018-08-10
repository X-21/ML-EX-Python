import numpy as np
from featureNormalize import feature_normalize
from gradientDescentMulti import gradient_descent_multi


def pause_func():
    print('Program paused. Press enter to continue.\n')
    while input() != '':
        pass


def load_data(filename):
    data_load = np.loadtxt(filename, delimiter=",")
    return data_load


if __name__ == '__main__':
    # ================ Part 1: Feature Normalization ================
    print('Loading data ...\n')
    # Load Data
    data = load_data('ex1data2.txt')
    data = np.split(data, [2], axis=1)
    X = data[0]
    y = data[1]
    m = len(y)
    # Print out some data points
    print('First 10 examples from the dataset: \n')
    for i in range(10):
        print(' x = [%.0f %.0f], y = %.0f \n' % (X[i][0], X[i][1], y[i]))
    # pause_func()

    # Scale features and set them to zero mean
    print('Normalizing Features ...\n')
    X, mu, sigma = feature_normalize(X)
    # Add intercept term to X
    X = np.append(np.ones((m, 1)), X, axis=1)

    # ================ Part 2: Gradient Descent ================
    print('Running gradient descent ...\n')
    # Number of iterations (loops)
    num_iters = 400
    # Try some other values of alpha
    alpha = 1
    theta = np.zeros(3, 1)
    theta, J_history_0 = gradient_descent_multi(X, y, theta, alpha, num_iters)

    a = 1
