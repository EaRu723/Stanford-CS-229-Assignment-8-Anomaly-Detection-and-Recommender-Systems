import numpy as np


def cofi_cost_function(params, Y, R, num_users, num_movies, num_features, lmd):
    # Reshape the flattened params array into X and Theta matrices
    X = params[0:num_movies * num_features].reshape((num_movies, num_features))
    theta = params[num_movies * num_features:].reshape((num_users, num_features))

    # Initialize the cost and gradient matrices
    cost = 0
    X_grad = np.zeros(X.shape)
    theta_grad = np.zeros(theta.shape)

    # Compute the predicted ratings only considering the ratings that exist (R = binary 1 or 0)
    hypothesis = (np.dot(X, theta.T) - Y) * R

    # Compute the cost function
    cost = (1/2)*np.sum(hypothesis**2) + (lmd/2)*np.sum(theta**2) + (lmd/2)*np.sum(X**2)

    # Compute the gradients for X and Theta
    X_grad = np.dot(hypothesis, theta) + lmd * X
    theta_grad = np.dot(hypothesis.T, X) + lmd * theta

    # Flatten and concatenate the graidents for X and Theta to return them as a single vector
    grad = np.concatenate((X_grad.flatten(), theta_grad.flatten()))

    return cost, grad
