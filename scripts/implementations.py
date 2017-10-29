# -*- coding: utf-8 -*-
"""Implementation of ML methods for project 1"""
import csv
import numpy as np
from helper_functions import *


#____________________________ LEAST SQUARES _____________________


def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    e = y - tx.dot(w)
    mse = compute_mse(e)
    return w, mse


#_______________ LEAST SQUARES USING GRADIENT DESCENT __________



def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w

    for n_iter in range(max_iters):
        gradient, error = compute_gradient(y, tx, w)
        loss = compute_mse(error)
        w = w - gamma * gradient

    return w, loss

#________ LEAST SQUARES USING STOCHASTIC GRADIENT DESCENT ________

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


#____________________________ GRADIENT DESCENT _____________________

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    weights = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):

        #Compute gradient & losses
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * gradient

        # store w and loss
        weights.append(w)
        losses.append(loss)


    return losses, weights



#____________________________ RIDGE REGRESSION _____________________

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    a = tx.T.dot(tx) + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    b = tx.T.dot(y)

    w = np.linalg.solve(a,b)

    e = y - tx.dot(w)
    mse = compute_mse(e)
    rmse = compute_rmse(mse)

    return w, rmse

#____________________________ LOGISTIC REGRESSION _____________________

def logistic_regression(y, tx, initial_w, max_iters, gamma_):

    w = initial_w

    for iter_ in range(max_iters):
        loss = calculate_loss(y,tx,w)
        gradient = calculate_gradient(y,tx,w)
        w -= gamma_*gradient # Pas sûr, faut peut etre faire avec le hessian

    return w

#____________________ REGULARIZED LOGISTIC REGRESSION _____________

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return w
