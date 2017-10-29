#!/usr/bin/env python

"""run.py: Given a train and a testing data set run.py will train a model and test it by cross-validation"""

import numpy as np
import math

from helper_functions import *
from proj1_helpers import *
from implementations import *
from data_visualization import *

__author__      = "Jean Gschwind, Tristan Besson and Sebastian Savidan"


#____________________________ Load datasets ____________________________

print("\nLoading training data...")
DATA_TRAIN = '../data/train.csv'
y_train, tx_train, ids_train = load_csv_data(DATA_TRAIN)

print("Loading testing data...")
DATA_TEST = '../data/test.csv'
_, tx_test, ids_test = load_csv_data(DATA_TEST)

print("Pre-processing data...")
#Replace 999 of data with NaNs
#As stated in Learning to discover: the Higgs boson machine learning challenge undefined values were set to -999
tx_train_nan=tx_train.copy()
tx_train_nan[tx_train_nan ==-999] = np.nan

tx_test_nan=tx_test.copy()
tx_test_nan[tx_test_nan ==-999] = np.nan

#Standardize data
tx_train, _, _ = standardize(tx_train_nan)
tx_test, _, _ = standardize(tx_test_nan)

# ____________________________ Visualize data ____________________________

data_visualization(tx_train, y_train)

#Handle missing data
tx_train = nan_handling(tx_train)
tx_test = nan_handling(tx_test)

#Feature engineering
#Test different features enginnering, TODO : find a way to select good featur engineering

print("Applying regression...")
#Apply regression
# tx=np.c_[np.ones((y.shape[0],1)),tx]
# w,loss = least_squares(y, tx)

#TODO : faire un truc qui check le meilleur lambda avec un cross validation
#least_squares_GD(y, tx, initial_w, max_iters, gamma)
#least_squares_SGD(y, tx, initial_w, max_iters, gamma)
w, loss = ridge_regression(y_train, tx_train, 0.00000000001)

#Create submission file
print("Creating submission file...")
#tx_test = np.c_[np.ones((tx_test.shape[0],1)),tx_test]
OUTPUT_PATH = '../data/sample-submission.csv'
y_pred = predict_labels(w, tx_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print("Submission file created \n")
