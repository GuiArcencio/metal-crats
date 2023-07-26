import numpy as np
import pandas as pd

from metamodel import build_metamodel
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

def run_model_using_loo(X, y, option="1nn", random_state=None):
    random_state = check_random_state(random_state)
    preds = np.empty_like(y)
    loo = LeaveOneOut()

    for train_index, test_index in loo.split(X):
        label_encoder = LabelEncoder()

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train = y[train_index]
        y_train = label_encoder.fit_transform(y_train)

        model = build_metamodel(option, random_state)

        model.fit(X_train, y_train)
        preds[test_index[0]] = label_encoder.inverse_transform(model.predict(X_test))[0]

    return preds

def best_regressors_to_rmse(y_pred, rmses):
    err = []
    for model, dataset in zip(y_pred, sorted(rmses.columns)):
        err.append(rmses[dataset][model])

    return np.array(err)

def best_classifiers_to_acc(y_pred, accs):
    acc = []
    for model, dataset in zip(y_pred, sorted(accs.columns)):
        acc.append(accs[dataset][model])

    return np.array(acc)
