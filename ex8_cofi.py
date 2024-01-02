import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import scipy.optimize as opt

import cofiCostFunction as ccf
import checkCostFunction as cf
import loadMovieList as lm
import normalizeRatings as nr


plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

# ===================== Part 1: Loading movie ratings dataset =====================
# First we load a movie rating dataset
print('Loading movie ratings dataset.')

# Load data
data = scio.loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']

# Y is a 1682 x 943 2-d ndarray, containing ratings 1-5 of 1682 movies on 943 users
#
# R is a 1682 x 943 2-d ndarray, where R[i, j] = 1 if and only if user j gave a
# rating to movie i

# Show average rating score for toy story
print('Average ratings for movie 0(Toy Story): {:0.6f}/5'.format(np.mean(Y[0, np.where(R[0] == 1)])))

# Visualize the ratings matrix
plt.figure()
plt.imshow(Y)
plt.colorbar()
plt.xlabel('Users')
plt.ylabel('Movies')

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Collaborative Filtering Cost function =====================
# Compute the cost function for collaborative filtering

# Load pre-trained weights (X, theta, num_users, num_movies, num_features)
data = scio.loadmat('ex8_movieParams.mat')
X = data['X']
theta = data['Theta']
num_users = data['num_users']
num_movies = data['num_movies']
num_features = data['num_features']

# Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3
X = X[0:num_movies, 0:num_features]
theta = theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]

# Evaluate cost function
cost, grad = ccf.cofi_cost_function(np.concatenate((X.flatten(), theta.flatten())), Y, R, num_users, num_movies, num_features, 0)

print('Cost at loaded parameters: {:0.2f}\n(this value should be about 22.22)'.format(cost))

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Collaborative Filtering Gradient =====================
# Check the gradient calculation for collaborative filtering
print('Checking gradients (without regularization) ...')

# Check gradients by running check_cost_function()
cf.check_cost_function(0)

input('Program paused. Press ENTER to continue')

# ===================== Part 4: Collaborative Filtering Cost Regularization =====================
# Implement regularization within cost function

# Evaluate cost function
cost, _ = ccf.cofi_cost_function(np.concatenate((X.flatten(), theta.flatten())), Y, R, num_users, num_movies, num_features, 1.5)

print('Cost at loaded parameters (lambda = 1.5): {:0.2f}\n'
      '(this value should be about 31.34)'.format(cost))

input('Program paused. Press ENTER to continue')

# ===================== Part 5: Collaborative Filtering Gradient Regularization =====================
# Check the gradient with regularization

print('Checking Gradients (with regularization) ...')

# Check gradients by running check_cost_function
cf.check_cost_function(1.5)

input('Program paused. Press ENTER to continue')

# ===================== Part 6: Entering ratings for a new user =====================
# Allows a user to enter their own ratings
movie_list = lm.load_movie_list()

# Initialize my ratings
my_ratings = np.zeros(len(movie_list))

# Check the file movie_ids.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 0, so to rate it "4", you can set
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

# Displays a user's ratings
print('New user ratings:\n')
for i in range(my_ratings.size):
    if my_ratings[i] > 0:
        print('Rated {} for {}'.format(my_ratings[i], movie_list[i]))

input('Program paused. Press ENTER to continue')

# ===================== Part 7: Learning Movie Ratings =====================
# Traing collaborative filtering model on actual dataset

print('Training collaborative filtering ...\n'
      '(this may take 1 ~ 2 minutes)')


# Load data
data = scio.loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']

# Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by
# 943 users
#
# R is a 1682x943 matrix, where R[i,j] = 1 if and only if user j gave a
# rating to movie i

# Add our own ratings to the data matrix
Y = np.c_[my_ratings, Y]
R = np.c_[(my_ratings != 0), R]

# Normalize Ratings
Ynorm, Ymean = nr.normalize_ratings(Y, R)

# Useful values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set initial parameters (theta, X)
X = np.random.randn(num_movies, num_features)
theta = np.random.randn(num_users, num_features)

initial_params = np.concatenate([X.flatten(), theta.flatten()])

lmd = 10

# Define cost and gradient functions for optimization
def cost_func(p):
    return ccf.cofi_cost_function(p, Ynorm, R, num_users, num_movies, num_features, lmd)[0]


def grad_func(p):
    return ccf.cofi_cost_function(p, Ynorm, R, num_users, num_movies, num_features, lmd)[1]

# Perform optimization using conjugate gradient method
theta, *unused = opt.fmin_cg(cost_func, fprime=grad_func, x0=initial_params, maxiter=100, disp=False, full_output=True)

# Unfold the returned theta back into U and W
X = theta[0:num_movies * num_features].reshape((num_movies, num_features))
theta = theta[num_movies * num_features:].reshape((num_users, num_features))

print('Recommender system learning completed')
print(theta)

input('Program paused. Press ENTER to continue')

# ===================== Part 8: Recommendation for you =====================
# Here the recommendation engine makes recommendations for the new user

p = np.dot(X, theta.T)
my_predictions = p[:, 0] + Ymean

# Sort and print top movie recommendations
indices = np.argsort(my_predictions)[::-1]
print('\nTop recommendations for you:')
for i in range(10):
    j = indices[i]
    print('Predicting rating {:0.1f} for movie {}'.format(my_predictions[j], movie_list[j]))

# Print the original ratings provided by the user
print('\nOriginal ratings provided:')
for i in range(my_ratings.size):
    if my_ratings[i] > 0:
        print('Rated {} for {}'.format(my_ratings[i], movie_list[i]))

input('ex8_cofi Finished. Press ENTER to exit')
