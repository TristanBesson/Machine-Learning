#!/usr/bin/env python

"""run.py: Given a train and a testing data set run.py will train a model and test it by cross-validation"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

from helper_functions import *
from proj1_helpers import *
from implementations import *

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

print("Plotting data...", end=" ")

# Batch creation
idx_batch = np.random.randint(tx_train.shape[0], size=int(0.02*tx_train.shape[0]))
tx_train_batch = tx_train[idx_batch, :]
y_train_batch = y_train[idx_batch]
len_scatter = np.shape(tx_train_batch)[1]

plt.figure(figsize=(60, 60), dpi=100, facecolor='w', edgecolor='k')

index_scatter = 1
colors = ["red", "blue"]

for xfeature in range(len_scatter):
    for yfeature in range(len_scatter):
        plt.subplot(len_scatter, len_scatter, index_scatter)

        if xfeature != yfeature:
            plt.scatter(tx_train_batch[:, xfeature], tx_train_batch[:, yfeature], s=0.4,  c=y_train_batch, cmap=matplotlib.colors.ListedColormap(colors))
        else:
            feature_values = tx_train_batch[:, xfeature]
            plt.hist(feature_values[~np.isnan(feature_values)], color="red")

        plt.xticks(())
        plt.yticks(())
        index_scatter += 1

print("Saved.")

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('scatterplot_COLOR.png', dpi=100)

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
