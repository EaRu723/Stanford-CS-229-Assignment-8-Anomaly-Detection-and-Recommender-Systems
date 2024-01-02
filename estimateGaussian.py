import numpy as np

# X is a dataset with m # examples and n # features
def estimate_gaussian(X):
    # Get the number of examples (m) and features (n)
    m, n = X.shape

    # Initialize the vecors for mean (mu) and variance (sigma2) for each feature
    mu = np.zeros(n)
    sigma2 = np.zeros(n)

    # Calculate the mean of each feature across all examples
    mu = np.mean(X, axis=0)

    # Calculate the variance of each feature across all examples
    sigma2 = np.var(X, axis=0)

    # Return the mean and variance vectors
    return mu, sigma2
