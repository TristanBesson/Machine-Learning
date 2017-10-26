#!/usr/bin/env python

"""run.py: Given a train and a testing data set run.py will train a model and test it by cross-validation"""




import numpy as np
import matplotlib.pyplot as plt
import pandas
import math

from helper_functions import *
from proj1_helpers import *
from implementations import *

__author__      = "Jean Gschwind, Tristan Besson and Sebastian Savidan"


#____________________________ Load datasets _____________________

DATA_TRAIN = '../data/train.csv'
y, tx, ids = load_csv_data(DATA_TRAIN)

DATA_TEST = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST)


#Replace 999 of data with NaNs
#As stated in Learning to discover: the Higgs boson machine learning challenge undefined values were set to -999
tx_nan=tx.copy()
tx_nan[tx_nan ==-999] = np.nan

#Standardize data
tx, _, _ = standardize(tx)

#Handle missing data
