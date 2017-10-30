from helper_functions import *
from implementations import *
import matplotlib.pyplot as plt

def accuracy(y_predicted,y):
    return 1 - sum(abs(y - y_predicted))/(2*len(y_predicted))

def cross_validation(y, x, indices_k, k, lambda_, degree, model_function):
    """return the loss of ridge regression."""
    indice_test = indices_k[k]
    indice_train = indices_k[~(np.arange(indices_k.shape[0]) == k)]
    indice_train = indice_train.reshape(-1)

    y_test = y[indice_test]
    y_train = y[indice_train]
    x_test = x[indice_test]
    x_train = x[indice_train]

    initial_w = np.zeros((x_train.shape[1], 1))
    max_iters = 500
    gamma = 0.01
    model = model_function.__name__

    if model == "least_squares_GD" or model == "least_squares_SGD" or model == "logistic_regression":
        w, loss = model_function(y_train, x_train, initial_w, gamma)
    elif model == "least_squares":
        w, loss = model_function(y_train, x_train)
    elif model == "ridge_regression":
        w, loss = model_function(y_train, x_train, lambda_)
    else:
        w, loss = model_function(y_train, x_train, lambda_, initial_w, max_iters, gamma)

    # calculate the loss for train and test data
    e_train = y_train - x_train.dot(w)
    loss_train = np.sqrt(2 * compute_mse(e_train))
    e_test = y_test - x_test.dot(w)
    loss_test = np.sqrt(2 * compute_mse(e_test))

    y_pred = predict_labels(w,x_test)
    acc = accuracy(y_pred,y_test)
    print("Accuracy: ",acc)
    return loss_train, loss_test

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def find_best_lambda(y,x, degree, k_fold, model):

    seed = 1
    lambdas = np.logspace(-10, 0, 15)

    k_indices = build_k_indices(y, k_fold, seed)

    rmse_train = []
    rmse_test = []

    best_lambda = 0
    best_rmse_test = 100

    # cross validation
    for lambda_ in lambdas:
        print("\nlambda :", lambda_)
        rmse_tr_tmp = []
        rmse_te_tmp = []

        for k in range(k_fold):
            loss_train, loss_test = cross_validation(y, x, k_indices, k, lambda_, degree, model)
            rmse_train_tmp = np.mean(loss_train)
            rmse_test_tmp = np.mean(loss_test)

            rmse_train.append(rmse_train_tmp)
            rmse_test.append(rmse_test_tmp)

            if (best_rmse_test > rmse_test_tmp):
                print(best_rmse_test,">", rmse_test_tmp, "= best lambda found", )
                best_lambda = lambda_
                best_rmse_test = rmse_test_tmp
                best_rmse_train = rmse_train_tmp

    print("\nBest lambda =", best_lambda, "\n")


    return best_lambda, best_rmse_train, best_rmse_test
