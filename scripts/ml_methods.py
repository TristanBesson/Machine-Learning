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