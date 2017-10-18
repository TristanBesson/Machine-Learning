# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import math

def compute_mse(e):
    """Calculate the loss (MSE)."""
    return 1/2*np.mean(e**2)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    error = y - tx.dot(w)
    gradient = -1/len(error) * tx.T.dot(error)
    return gradient, error

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w

    for n_iter in range(max_iters):
        gradient, error = compute_gradient(y, tx, w)
        loss = compute_mse(error)
        w = w - gamma * gradient

    return w, loss

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

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    batch_size = 1

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
            gradient, error = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_mse(error)
            w = w - gamma * gradient

    return w, loss

def least_squares(y, tx):
    """Least squares algorithm."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    e = y - tx.dot(w)
    mse = (e**2).mean()

    return w, mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    a = tx.T.dot(tx) + 2*(tx.shape[0])*lambda_*np.eye(tx.shape[1])
    b = tx.T.dot(y)

    w = np.linalg.solve(a,b)
    e = y - tx.dot(w)

    mse = (e**2).mean()
    rmse = math.sqrt(2*mse)

    return w, rmse

def logistic_regression(y, tx, initial_w, max_iters, gamma):

    raise NotImplementedError

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    raise NotImplementedError