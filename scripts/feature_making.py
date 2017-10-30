#!/usr/bin/env python

"""feature_making.py: feature_making.py regroup a set of different transformation one can apply to data in order to create new features by features transformation"""




import numpy as np
import matplotlib.pyplot as plt
import math

from helper_functions import *
from proj1_helpers import *
from implementations import *

__author__      = "Jean Gschwind, Tristan Besson and Sebastian Savidan"

def exp_feat(x):
    """Apply exp on a feature x"""
    exp_x = np.exp(x)
    return exp_x

def multiple_feat(x1,x2):
    """Multiplies the elements of two features to create a new one"""
    mult_x = np.multiply(x1,x2)
    return mult_x

def create_multiple(tx_train,tx_test,ind1,ind2):
    new_x = multiple_feat(tx_train[:,ind1],tx_train[:,ind2])
    new_Xtest = multiple_feat(tx_test[:,ind1],tx_test[:,ind2])
    return new_x, new_Xtest


def cos_feat(x):
    """Apply cosinus on a feature x"""
    cos_x = np.cos(x)
    return cos_x

def sqrt_feat(x):
    """Apply sqrt on a feature x"""
    sqrt_x = np.sqrt(x)
    return sqrt_x

def log_feat(x):
    """Apply logarithm on a feature x"""
    ##Warning: May return NaNs
    log_x = np.log(x)
    return log_x

def power_feat(x,degree):
    """Powers a feature x to a given degree"""
    pow_x = np.power(x, degree)
    return pow_x

def features_couples(tx_train,tx_test,y_train):
    """Set of features combinaisions my multiplication picked by hand"""

#TODO : clean the code

    new_x, new_Xtest = create_multiple(tx_train,tx_test,0,2)
    tx_train, tx_test, token = feat_add(y_train,tx_train[:,0],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,2],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,0,7)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,0],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,7],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,2,7)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,2],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,7],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,3,7)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,3],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,7],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,3,9)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,3],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,9],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,3,21)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,3],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,21],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,3,23)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,3],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,23],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,3,29)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,3],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,29],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,4,5)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,4],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,5],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,4,6)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,4],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,6],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,5,6)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,5],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,6],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,9,21)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,9],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,21],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,9,23)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,9],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,23],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,9,29)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,9],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,29],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,10,16)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,10],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,16],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,18,20)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,18],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,20],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,19,21)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,19],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,21],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,21,24)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,21],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,24],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,21,29)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,21],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,29],new_x,new_Xtest,tx_train,tx_test)

    new_x, new_Xtest = create_multiple(tx_train,tx_test,26,29)
    tx_train, tx_test, token  = feat_add(y_train,tx_train[:,26],new_x,new_Xtest,tx_train,tx_test)
    if token == False:
        tx_train, tx_test, token = feat_add(y_train,tx_train[:,29],new_x,new_Xtest,tx_train,tx_test)
    return tx_train, tx_test
