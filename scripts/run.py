#!/usr/bin/env python

"""run.py: Given a train and a testing data set run.py will train a model and test it by cross-validation"""




import numpy as np
import matplotlib.pyplot as plt
import math

from helper_functions import *
from proj1_helpers import *
from implementations import *

__author__      = "Jean Gschwind, Tristan Besson and Sebastian Savidan"


#____________________________ Load datasets _____________________
print("\nLoading training data...")
DATA_TRAIN = '../data/train.csv'
y, tx, ids = load_csv_data(DATA_TRAIN)

print("Loading testing data...")
DATA_TEST = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST)

print("Pre-processing data...")
#Replace 999 of data with NaNs
#As stated in Learning to discover: the Higgs boson machine learning challenge undefined values were set to -999
tx_nan=tx.copy()
tx_nan[tx_nan ==-999] = np.nan

tX_test_nan=tX_test.copy()
tX_test_nan[tX_test_nan ==-999] = np.nan

#Standardize data
tx, _, _ = standardize(tx_nan)

tX_test, _, _ = standardize(tX_test_nan)

#Visualize data
    #scatterplot
    #boxplot
    #tableau stylé

#Handle missing data
tx = nan_handling(tx)

tX_test = nan_handling(tX_test)

#Feature engineering
#Test different features enginnering, TODO : find a way to select good featur engineering

print("Applying regression...")
#Apply regression
# tx=np.c_[np.ones((y.shape[0],1)),tx]
# w,loss = least_squares(y, tx)

#TODO : faire un truc qui check le meilleur lambda avec un cross validation
#least_squares_GD(y, tx, initial_w, max_iters, gamma)
#least_squares_SGD(y, tx, initial_w, max_iters, gamma)
w, loss = ridge_regression(y, tx, 0.00000000001)

#Create submission file
print("Creating submission file...")
#tX_test = np.c_[np.ones((tX_test.shape[0],1)),tX_test]
OUTPUT_PATH = '../data/sample-submission.csv'
y_pred = predict_labels(w, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print("Submission file created \n")
