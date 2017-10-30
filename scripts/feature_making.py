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
