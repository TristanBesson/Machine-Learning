from helper_functions import *
from implementations import *
import matplotlib.pyplot as plt

def accuracy(y_predicted,y):
    # Compute the accuracy of the predictions
    return 1 - sum(abs(y - y_predicted))/(2*len(y_predicted))

def cross_validation_lambda(y, x, k_indices, k, lambda_, degree, model_function):
    """return the loss of ridge regression."""

    # Separate train and test data in function of the indice
    indice_test = k_indices[k]

    x_test = x[indice_test]
    y_test = y[indice_test]

    x_train = np.delete(x, indice_test, axis=0)
    y_train = np.delete(y, indice_test, axis=0)

    initial_w = np.zeros((x_train.shape[1], 1))
    max_iters = 500
    gamma = 0.01

    # Choose the correct model
    model = model_function.__name__

    if model == "least_squares_GD" or model == "least_squares_SGD" or model == "logistic_regression":
        w, loss = model_function(y_train, x_train, initial_w, gamma)
    elif model == "least_squares":
        w, loss = model_function(y_train, x_train)
    elif model == "ridge_regression":
        w, loss = model_function(y_train, x_train, lambda_)
    else:
        w, loss = model_function(y_train, x_train, lambda_, initial_w, max_iters, gamma)

    # Compute loss for train and test data
    e_train = y_train - x_train.dot(w)
    loss_train = np.sqrt(2*compute_mse(e_train))
    e_test = y_test - x_test.dot(w)
    loss_test = np.sqrt(2*compute_mse(e_test))

    # Compute the prediction and check the accuracy
    y_pred = predict_labels(w, x_test)
    acc = accuracy(y_pred, y_test)
    print("Accuracy: ", acc)

    return loss_train, loss_test

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def find_best_lambda(y,x, degree, k_fold, model):

    nb_lambdas = 20
    seed = 1
    best_lambda = 0
    best_rmse_test = 100

    # Create a space of lambdas to test based on a log rule
    lambdas = np.logspace(-5, 0, nb_lambdas)
    # Create different fold for the cross validation
    k_indices = build_k_indices(y, k_fold, seed)

    rmse_train = []
    rmse_test = []

    # cross validation
    for lambda_ in lambdas:
        rmse_tr_tmp = []
        rmse_te_tmp = []

        for k in range(k_fold):
            loss_train, loss_test = cross_validation_lambda(y, x, k_indices, k, lambda_, degree, model)
            rmse_train_tmp = np.mean(loss_train)
            rmse_test_tmp = np.mean(loss_test)

            # We choose the smaller RMSE (for the test) to choose lambda
            if (best_rmse_test > rmse_test_tmp):
                best_lambda = lambda_
                best_rmse_test = rmse_test_tmp
                best_rmse_train = rmse_train_tmp

        rmse_train.append(rmse_train_tmp)
        rmse_test.append(rmse_test_tmp)

    ## Plot evolution of RMSE in function of lambdas
    # plt.plot(rmse_train, label="rmse (train)")
    # plt.plot(rmse_test, label="rmse (test)")
    # plt.grid()
    # plt.legend()
    # plt.title("Root Mean Squared Error (RMSE) in function of lambda")
    # plt.xlabel("lambda")
    # plt.ylabel("RMSE")
    # plt.xticks(range(0,nb_lambdas,1), np.round(lambdas,7), rotation=70)
    # plt.show()

    print("\nBest lambda =", best_lambda, "\n")

    return best_lambda, best_rmse_train, best_rmse_test

def cross_validation(y, x, k_fold, degree, model_function):
    # Create different fold for the cross validation

    k_indices = build_k_indices(y, k_fold, seed=1)
    gamma = 0.01
    max_iters = 500

    for k in range(k_fold):

        # Separate train and test data in function of the indice
        indice_test = k_indices[k]

        x_test = x[indice_test]
        y_test = y[indice_test]

        x_train = np.delete(x, indice_test, axis=0)
        y_train = np.delete(y, indice_test, axis=0)

        initial_w = np.zeros((x_train.shape[1], 1))

        # Choose the correct model
        model = model_function.__name__
        print(model)

        if model == "least_squares_GD" or model == "least_squares_SGD" or model == "logistic_regression":
            w, loss = model_function(y_train, x_train, initial_w, max_iters, gamma)
        else:
            w, loss = model_function(y_train, x_train)

        # Compute loss for train and test data
        e_train = y_train - x_train.dot(w)
        loss_train = np.sqrt(2*compute_mse(e_train))
        e_test = y_test - x_test.dot(w)
        loss_test = np.sqrt(2*compute_mse(e_test))

        ## Compute the prediction and check the accuracy
        y_pred = predict_labels(w, x_test)
        acc = accuracy(y_pred, y_test)
        print("Accuracy: ", acc)

    return w
