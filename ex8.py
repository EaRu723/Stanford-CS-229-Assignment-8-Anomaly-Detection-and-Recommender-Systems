import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

import estimateGaussian as eg
import multivariateGaussian as mvg
import visualizeFit as vf
import selectThreshold as st

plt.ion()

# Load and visualize the dataset

print('Visualizing example dataset for outlier detection.')
data = scio.loadmat('ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval'].flatten()

# Visualize the example dataset
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c='b', marker='x', s=15, linewidth=1)
plt.axis([0, 30, 0, 30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s')

input('Program paused. Press ENTER to continue')

# Estimate dataset statistics

print('Visualizing Gaussian fit.')

# Estimate mu and sigma2
mu, sigma2 = eg.estimate_gaussian(X)

# Calculate the probabilities for each datapoint in the training set
p = mvg.multivariate_gaussian(X, mu, sigma2)

# Visualize the fit
vf.visualize_fit(X, mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s')

input('Program paused. Press ENTER to continue')

# Find Outliers
# Use cross-validation set to find best threshold epsilons for anomaly detection
pval = mvg.multivariate_gaussian(Xval, mu, sigma2)

epsilon, f1 = st.select_threshold(yval, pval)
print('Best epsilon found using cross-validation: {:0.4e}'.format(epsilon))
print('Best F1 on Cross Validation Set: {:0.6f}'.format(f1))
print('(you should see a value epsilon of about 8.99e-05 and F1 of about 0.875)')

# Identify and visualize outliers in the training set
outliers = np.where(p < epsilon)
plt.scatter(X[outliers, 0], X[outliers, 1], marker='o', facecolors='none', edgecolors='r')

input('Program paused. Press ENTER to continue')

# Multidimensional Outliers
# Apply the anomaly detection model to a dataset with more features

# Loads the second dataset.
data = scio.loadmat('ex8data2.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval'].flatten()

# Apply the same steps to the larger dataset
mu, sigma2 = eg.estimate_gaussian(X)

# Training set
p = mvg.multivariate_gaussian(X, mu, sigma2)

# Cross Validation set
pval = mvg.multivariate_gaussian(Xval, mu, sigma2)

# Find the best threshold
epsilon, f1 = st.select_threshold(yval, pval)

print('Best epsilon found using cross-validation: {:0.4e}'.format(epsilon))
print('Best F1 on Cross Validation Set: {:0.6f}'.format(f1))
print('# Outliers found: {}'.format(np.sum(np.less(p, epsilon))))
print('(you should see a value epsilon of about 1.38e-18, F1 of about 0.615, and 117 outliers)')

input('ex8 Finished. Press ENTER to exit')
