# README FILE

## Higgs Boson Machine Learning Challenge

### Submission for the group composed of Jean Gschwind, Sebastian Savidan & Tristan Besson.

Kaggle name "No Regress"


_run.py_

Function that creates the results found on the Kaggle competition. It contains all the necessary steps, from loading, processing the features and then applying the machine learning method. It will create directly a submission file for the competition.

_implementations.py_

Contains all the asked Machine Learning Methods. Please note that the regularized logistic regression is currently not working.

_cross-validation.py_

Contains the methods around the cross-validation principle. A function find_best_lambda will find the best penalty term for our system.

_helper-functions.py_

Contains most of the functions used with the ML methods.

_data-visualization.py_

Contains the code that creates the scatter plot matrix present in the report.

_feature-making.py_

Contains a set of different transformation one can apply to data in order to create new features by features transformation

_proj1-helpers.py_

Functions furnished for the competition





To run our classifier: 

-put the train and test data csv files in a folder called data next to the scripts folder

-open a terminal, navigate to the scripts directory, and type: python run.py

-The corresponding prediction csv file will be saved isinde the data folder
