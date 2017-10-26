# -*- coding: utf-8 -*-
"""Implementation of necessary methods for project 1"""
import csv
import numpy as np
import matplotlib.pyplot as plt


#____________________________ STANDARDIZE _____________________

def standardize(x):
    """Standardize the original data set. Ignoring NaNs"""
    mean_x = np.nanmean(x)
    x = x - mean_x
    std_x = np.nanstd(x)
    x = x / std_x
    return x, mean_x, std_x

#_______________________ FEATURE HANDLING _____________________
# Must contain all the processing performed on the features (columns deletion, PCA...)

def feature_handling(tx):
    # Pre-processing, delete columns, delete features, PCA...

    return tx


#____________________________ SPLIT DATASET _____________________

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    np.random.shuffle(x)
    np.random.shuffle(y)

    split = int(ratio*(x.shape[0]))

    trainX = x[:split,]
    testX = x[split:,]

    trainY = y[:split,]
    testY = y[split:,]

    return trainX, testX, trainY, testY


#____________________________ BUILD POLYNOMIAL _____________________
#TODO
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

    return matrix


#____________________________ COMPUTE LOSS _____________________

def compute_loss(y, tx, w):
    """TODO: implement MAE"""

    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    model = w*tx
    loss = np.mean(((y-model[:,0]-model[:,1])**2))/2
    return loss


#____________________________ COMPUTE GRADIENT _____________________

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    model = w*tx
    error = y - model[:,0] - model[:,1]
    gradient = - (np.dot(tx.transpose(),error))/len(y)
    return gradient, error


#____________________________ COMPUTE MSE _____________________

def compute_mse(e):
    """Calculate the loss (MSE)."""
    return 1/2*np.mean(e**2)

#____________________________ COMPUTE RMSE _____________________

def compute_rmse(mse):
    """Calculate the loss (MSE)."""
    return np.sqrt(2*mse)


#______________________________ SIGMOID _______________________

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0/(1+np.exp(-t))


#__________________ COMPUTE LOSS FOR LOG REG ______________

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = np.transpose(y).dot(np.log(pred)) + np.transpose(1-y).dot(np.log(1-pred))
    return -loss[0][0]


#____________________________ COMPUTE GRADIENT FOR LOG REG _____________________

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.transpose(tx).dot(sigmoid(tx.dot(w))-y)


#____________________________ CALCULATE HESSIAN _____________________


def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate hessian: TODO
    # ***************************************************
    S_diag = sigmoid(tx.dot(w))*(1-sigmoid(tx.dot(w)))
    S = np.eye(S_diag.shape[0])*S_diag
    return np.transpose(tx).dot(S).dot(tx)

#____________________________ PLOT TRAIN TEST _____________________

def plot_train_test(train_errors, test_errors, lambdas, degree):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set

    degree is just used for the title of the plot.
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("Ridge regression for polynomial degree " + str(degree))
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("ridge_regression")

#____________________________ CROSS VALIDATION _____________________

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    k_fold = list(range(k_indices.shape[0]))
    k_fold.remove(k)
    tr_indices = k_indices[k_fold].ravel()

    x_train = x[tr_indices]
    x_test = x[k_indices]
    y_train = y[tr_indices]
    y_test = y[k_indices]

    # ***************************************************
    # INSERT YOUR CODE HERE
    # form data with polynomial degree: TODO
    # ***************************************************
    train_matrix = build_poly(x_train, degree)
    test_matrix = build_poly(x_test, degree)


    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    w, loss_tr = ridge_regression(y_train, train_matrix, lambda_)

    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************
    loss_te = compute_mse(y_test, test_matrix, w)

    return loss_tr, loss_te

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_demo():
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation: TODO
    # ***************************************************
    for lambda_ in lambdas:
        tot_loss_tr = 0
        tot_loss_te = 0

        for k in range(0, k_fold):
            k_selected = k_indices[k,:]
            loss_te, loss_tr = cross_validation(y, x, k_selected, k, lambda_, degree)
            tot_loss_tr += loss_tr
            tot_loss_te += loss_te

        rmse_tr.append(np.sqrt(2*tot_loss_tr))
        rmse_te.append(np.sqrt(2*tot_loss_te))

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)

cross_validation_demo()
