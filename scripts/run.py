#!/usr/bin/env python

"""run.py: Given a train and a testing data set run.py will train a model and test it by cross-validation"""

import numpy as np
import math

from helper_functions import *
from proj1_helpers import *
from implementations import *
from data_visualization import *
from feature_making import *

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
#data_visualization(tx_train, y_train)

# ____________________________ Feature engineering____________________________

#Handle missing data
tx_train = nan_handling(tx_train)
tx_test = nan_handling(tx_test)


print("Engineering new features...")
#Feature engineering
#Test different features enginnering, TODO : find a way to select good featur engineering
#First part: Apply various feature engineering process, if resulting feature as higher correlation to output as before
for i in range(0,30):
    new_x = exp_feat(tx_train[:,i])
    new_Xtest = exp_feat(tx_test[:,i])
    tx_train, tx_test,_ = feat_add(y_train,tx_train[:,i],new_x,new_Xtest,tx_train,tx_test)

    new_x = cos_feat(tx_train[:,i])
    new_Xtest = cos_feat(tx_test[:,i])
    tx_train, tx_test,_ = feat_add(y_train,tx_train[:,i],new_x,new_Xtest,tx_train,tx_test)

    new_x = sqrt_feat(tx_train[:,i])
    new_Xtest = sqrt_feat(tx_test[:,i])
    tx_train, tx_test,_ = feat_add(y_train,tx_train[:,i],new_x,new_Xtest,tx_train,tx_test)

    new_x = log_feat(tx_train[:,i])
    new_Xtest = log_feat(tx_test[:,i])
    tx_train, tx_test,_ = feat_add(y_train,tx_train[:,i],new_x,new_Xtest,tx_train,tx_test)

    for j in range(2,6):
        new_x = power_feat(tx_train[:,i],j)
        new_Xtest = power_feat(tx_test[:,i],j)
        tx_train, tx_test,_ = feat_add(y_train,tx_train[:,i],new_x,new_Xtest,tx_train,tx_test)

print(np.shape(tx_train))

#Second part apply multiplication to features that seem correlated from figure scatterplot matrix
#Features couples that seem correlated from graph: (0,2:7),(2,7),(3,7:9:21:23:29),(4,5:6),(5,6),(9,21:23:29),(10,16),(18,20),(19,21:29),(21,24:29),(26,29)
# 22 features couples in total
new_x, new_Xtest =create_multiple(tx_train,tx_test,0,2)
tx_train, tx_test, token = feat_add(y_train,tx_train[:,0],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,2],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,0,7)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,0],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,7],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,2,7)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,2],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,7],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,3,7)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,3],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,7],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,3,9)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,3],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,9],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,3,21)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,3],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,21],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,3,23)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,3],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,23],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,3,29)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,3],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,29],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,4,5)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,4],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,5],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,4,6)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,4],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,6],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,5,6)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,5],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,6],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,9,21)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,9],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,21],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,9,23)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,9],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,23],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,9,29)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,9],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,29],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,10,16)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,10],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,16],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,18,20)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,18],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,20],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,19,21)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,19],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,21],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,21,24)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,21],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,24],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,21,29)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,21],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,29],new_x,new_Xtest,tx_train,tx_test)

new_x, new_Xtest =create_multiple(tx_train,tx_test,26,29)
tx_train, tx_test, token  = feat_add(y_train,tx_train[:,26],new_x,new_Xtest,tx_train,tx_test)
if token == False:
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,29],new_x,new_Xtest,tx_train,tx_test)

print(np.shape(tx_train))
degree = 4
tx_train = build_poly(tx_train, degree)
tx_test = build_poly(tx_test, degree)
print(np.shape(tx_train))
# ____________________________ Regression____________________________

print("Applying regression...")
#Apply regression
tx_train = np.c_[np.ones((y_train.shape[0],1)),tx_train]
tx_test = np.c_[np.ones((tx_test.shape[0],1)),tx_test]

initial_w = np.zeros(tx_train.shape[1])
max_iters = 150
#gamma_ = 0.035

# w,loss = least_squares(y_train, tx_train)
#w = logistic_regression(y_train, tx_train, initial_w, max_iters, gamma_)
#least_squares_GD(y, tx, initial_w, max_iters, gamma)
#least_squares_SGD(y, tx, initial_w, max_iters, gamma)

seed = 1
degree = 7
k_fold = 4
lambdas = np.logspace(-10, 0, 20)

# split data in k fold
k_indices = build_k_indices(y_train, k_fold, seed)

# define lists to store the loss of training data and test data
rmse_tr = []
rmse_te = []

best_lambda = 0
min_erreur = 100

# cross validation to find best parameters
for lambda_ in lambdas:
    print("\nlambda :", lambda_)
    rmse_tr_tmp = []
    rmse_te_tmp = []

    for k in range(k_fold):
        loss_tr, loss_te = cross_validation(y_train, tx_train, k_indices, k, lambda_, degree)
        rmse_tr_tmp = np.mean(loss_tr)
        rmse_te_tmp = np.mean(loss_te)

        if (min_erreur > rmse_te_tmp):
            print(min_erreur,">", rmse_te_tmp, "= best lambda found", )
            best_lambda = lambda_
            min_erreur = rmse_te_tmp

print("\nBest lambda =", best_lambda, "\n")
#print(rmse_tr, "\n\n")

# Calcul du model avec le meilleur lambda
w, rmse = ridge_regression(y_train, tx_train, best_lambda)





#Create submission file
print("Creating submission file...")
#tx_test = np.c_[np.ones((tx_test.shape[0],1)),tx_test]
OUTPUT_PATH = '../data/sample-submission.csv'
y_pred = predict_labels(w, tx_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print("Submission file created \n")
