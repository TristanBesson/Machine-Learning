from helper_functions import *

def cross_validation_final(y,x, degree, k_fold, model):

    seed = 1
    lambdas = np.logspace(-10, 0, 20)

    k_indices = build_k_indices(y, k_fold, seed)

    best_lambda = 0
    best_rmse_te = 100

    # cross validation
    for lambda_ in lambdas:
        print("\nlambda :", lambda_)
        rmse_tr_tmp = []
        rmse_te_tmp = []

        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree, model)
            rmse_tr_tmp = np.mean(loss_tr)
            rmse_te_tmp = np.mean(loss_te)

            if (best_rmse_te > rmse_te_tmp):
                print(best_rmse_te,">", rmse_te_tmp, "= best lambda found", )
                best_lambda = lambda_
                best_rmse_te = rmse_te_tmp
                best_rmse_tr = rmse_tr_tmp

    print("\nBest lambda =", best_lambda, "\n")

    return best_lambda, best_rmse_tr, best_rmse_te
