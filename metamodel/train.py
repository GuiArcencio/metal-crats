import numpy as np
import pandas as pd

from metamodel import build_metamodel, AVAILABLE_METAMODELS
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

def get_rmse_from_metamodel_prediction(y_pred, rmses):
    err = []
    for model, dataset in zip(y_pred, rmses.columns):
        err.append(rmses[dataset][model])

    return np.array(err)

def run_all_metamodels(X, y, rmses):
    results = {}
    seeds = [
        5180022, 169332600, 174764313,
        350137309, 468180727, 516403885,
        741233569, 757999656, 812064213, 
        840293668
    ]

    for metaoption in AVAILABLE_METAMODELS:
        print(f"Running metal-rats-{metaoption}")

        if metaoption != "rf":
            metamodel_rmses = get_rmse_from_metamodel_prediction(
                run_model_using_loo(X, y, metaoption, None),
                rmses
            )
            results[f"metal-rats-{metaoption}"] = metamodel_rmses
        else:
            runs = []
            for run, seed in enumerate(seeds):
                print(f"\tRun #{run+1}")
                runs.append(get_rmse_from_metamodel_prediction(
                    run_model_using_loo(X, y, metaoption, seed),
                    rmses
                ))
            results[f"metal-rats-{metaoption}"] = np.mean(runs, axis=0)

    return pd.DataFrame(results, index=X.index)