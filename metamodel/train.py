import numpy as np
import pandas as pd

from metamodel import build_metamodel
from sklearn.model_selection import LeaveOneOut

def run_model_using_loo(X, y, option="1nn", random_state=None):
    preds = np.empty_like(y)
    loo = LeaveOneOut()

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train = y[train_index]

        model = build_metamodel(option, random_state)

        model.fit(X_train, y_train)
        preds[test_index[0]] = model.predict(X_test)[0]

    return preds

def best_regressors_to_rmse(y_pred, rmses):
    err = []
    for model, dataset in zip(y_pred, sorted(rmses.columns)):
        err.append(rmses[dataset][model])

    return np.array(err)

def best_classifiers_to_acc(y_pred, accs):
    acc = []
    for model, dataset in zip(y_pred, sorted(accs.column)):
        acc.append(accs[dataset][model])

    return np.array(acc)