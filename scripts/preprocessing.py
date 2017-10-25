##Imports
import numpy as np
from ex_helpers import *  #checker le quel est le bon
from proj1_helpers import *
from implementations import *


#Load data
class_label,features,event_id = load_csv_data(/../data/train.csv)


#Replace 999 of data with NaNs
features_nan=features.copy()
features_nan[features_nan ==-999] = np.nan

#Maybe evaluate % of NaNs (if to much NaNs -> discard column)





#Standardize data by substracting to each colums its mean and dividing by standard deviation





#Plot every feature according to an other (voir PDF lazare) to "visualize our data"






#Do something about NaNs (a trouver: chercher dans les papers que faire quand ya des NaNs dans la data)






#Evaluate variance to see most explaining data






#Kernel trick or other tricks (polynomials)to augment the dimensions of data with best features couples (if strongly correlated: see from first figure)
