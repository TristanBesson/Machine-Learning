##Imports
import numpy as np
from ex_helpers import *  #checker le quel est le bon
from proj1_helpers import *
from implementations import *

print("LOL")

#Load data
DATA_TRAIN = 'data/train.csv'
class_label,features,event_id = load_csv_data(DATA_TRAIN)


#Replace 999 of data with NaNs
features_nan=features.copy()
features_nan[features_nan ==-999] = np.nan

#__________________________________________________________________________________________________________________________

#Maybe evaluate % of NaNs (if to much NaNs -> discard column)

threshold_Drop = 0.5

for i in range(0,features_nan.shape[1]):
    column_count = 0
    for j in range(0,features_nan.shape[0]):
        if np.isnan(features_nan.item((j,i))):
            column_count +=1
    percent_nan = float(column_count)/features_nan.shape[0]

    if percent_nan > threshold_Drop:
        print 'To drop column', i+1,'with ', percent_nan

## CONCLUSION: We could drop columns 5,6,7,13,27,28,29 (70% de NaN); après aussi 24,25,26 avec 40% de NaN

#__________________________________________________________________________________________________________________________

#Standardize data by substracting to each colums its mean and dividing by standard deviation

tx = features_nan.copy()
tx = np.delete(tx, 28, 1)
tx = np.delete(tx, 27, 1)
tx = np.delete(tx, 26, 1)
tx = np.delete(tx, 12, 1)
tx = np.delete(tx, 6, 1)
tx = np.delete(tx, 5, 1)
tx = np.delete(tx, 4, 1)

for i in range(0,tx.shape[1]):
    tx[:,i] = (tx[:,i] - np.nanmean(tx[:,i]))/np.nanstd(tx[:,i])


#__________________________________________________________________________________________________________________________

#Plot every feature according to an other (voir PDF lazare) to "visualize our data"

data = pandas.read_csv(DATA_TRAIN)
scatter_matrix(data)

scatter_matrix(tx)
plt.show()


#__________________________________________________________________________________________________________________________

#Do something about NaNs (a trouver: chercher dans les papers que faire quand ya des NaNs dans la data)




#__________________________________________________________________________________________________________________________

#Evaluate variance to see most explaining data




#__________________________________________________________________________________________________________________________

#Kernel trick or other tricks (polynomials)to augment the dimensions of data with best features couples (if strongly correlated: see from first figure)
