# -*- coding: utf-8 -*-
"""Scatter plot for features vizualisation"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

from helper_functions import *
from proj1_helpers import *
from implementations import *

print("Extraction Data...")

DATA_TRAIN = '../data/train.csv'
y_train, tx_train, ids_train = load_csv_data(DATA_TRAIN)

DATA_TEST = '../data/test.csv'
_, tx_test, ids_test = load_csv_data(DATA_TEST)

print("Standardization...")

tx_train[tx_train == -999] = np.nan
tx_train, _, _ = standardize(tx_train)

print("Batch creation...")

# Batch creation
idx_batch = np.random.randint(tx_train.shape[0], size=int(0.02*tx_train.shape[0]))
tx_train_batch = tx_train[idx_batch, :]
y_train_batch = y_train[idx_batch]

len_scatter = np.shape(tx_train_batch)[1]

print("Plotting data...")

plt.figure(figsize=(60, 60), dpi=100, facecolor='w', edgecolor='k')

index_scatter = 1
colors = ["red", "blue"]

for xfeature in range(len_scatter):
    for yfeature in range(len_scatter):
        plt.subplot(len_scatter, len_scatter, index_scatter)

        if xfeature != yfeature:
            #plt.scatter(tx_train_batch[:, xfeature], tx_train_batch[:, yfeature], s=0.4,  c=y_train_batch, cmap=matplotlib.colors.ListedColormap(colors))
            plt.scatter(tx_train_batch[:, xfeature], tx_train_batch[:, yfeature])
        else:
            test = tx_train_batch[:, xfeature]
            #plt.hist(test[~np.isnan(test)], color="red")
            plt.hist(test[~np.isnan(test)])
# Beam_irradiance_DNI[Beam_irradiance_DNI != 0]

        plt.xticks(())
        plt.yticks(())
        index_scatter += 1

print("Save...")

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('sampleFileName.png', dpi=100)

print("FINISH BRO!")
