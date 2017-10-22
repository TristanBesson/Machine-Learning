# -*- coding: utf-8 -*-
"""Implementation of ML methods for project 1"""
import csv
import numpy as np


def least_squares(y, tx):
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

def compute_loss(y, tx, w):
    """TODO: implement MAE"""

    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    model = w*tx
    loss = np.mean(((y-model[:,0]-model[:,1])**2))/2
    return loss


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    model = w*tx
    error = y - model[:,0] - model[:,1]
    gradient = - (np.dot(tx.transpose(),error))/len(y)
    return gradient

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
