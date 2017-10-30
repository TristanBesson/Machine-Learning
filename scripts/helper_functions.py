# -*- coding: utf-8 -*-
"""Implementation of necessary methods for project 1"""
import csv
import numpy as np
import matplotlib.pyplot as plt
#from implementations import ridge_regression
from proj1_helpers import predict_labels

#____________________________ STANDARDIZE _____________________

def standardize(x):
    """Standardize the original data set. Ignoring NaNs"""
    mean_x = np.nanmean(x, axis=0)
    std_x = np.nanstd(x, axis=0)
    x = (x - mean_x)/std_x
    return x, mean_x, std_x

#_______________________ FEATURE HANDLING _____________________

def nan_handling(tx,value=None,coef=None):
    #Handle NaNs in different ways:
        #by default NaNs will be replaced with the mean value of the corresponding feature unless a specific replacement 'value' is given
        #the coef option gives the possibility of deleting features with more than coef*100% of NaNs in it
        #By default no feature is removed
        #One can delete entirely features containing NaN by setting coef to 0.0
    tx_copy = tx.copy()

    if coef:
        tx_count=np.count_nonzero(np.isnan(tx_copy), axis = 0)
        tx_copy = np.delete(tx_copy,np.where(tx_count/tx_copy.shape[0] > coef),1)
        print("Features might have been deleted")
    if value:
        tx_copy[np.isnan(tx_copy)] = value
        return tx_copy
    else:
        column_mean = np.nanmean(tx_copy,axis=0)
        ind = np.where(np.isnan(tx_copy))
        tx_copy[ind] = column_mean[ind[1]]
        return tx_copy


def feat_add(y,x,new_x,new_Xtest,tx_train,tX_test):
    if feat_test(y,x,new_x):
        print("Feature is good, added")
        tx_train = np.c_[tx_train,new_x]
        tX_test = np.c_[tX_test,new_Xtest]
        return tx_train, tX_test, True
    else:
        print("Feature was worse, not added")
        return tx_train, tX_test, False

def feat_test(y,x,new_x):
    #feat_test will test if a new feature is more correlated to an other one
    if abs(np.corrcoef(y,x)[1,0]) < abs(np.corrcoef(y,new_x)[1,0]):
        return True
    else:
        return False

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

    matrix = np.ones((x.shape[0], 1))
    for j in range(1, degree+1):
        matrix = np.c_[matrix, np.power(x, j)]
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
    return -loss


#____________________________ COMPUTE GRADIENT FOR LOG REG _____________________

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.transpose(tx).dot(sigmoid(tx.dot(w))-y)


#____________________________ CALCULATE HESSIAN _____________________


def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
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






#____________________________ BATCH ITER _____________________

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
