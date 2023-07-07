import pandas as pd
import numpy as np
import os

from tsml.datasets import load_from_ts_file

def load_dataset(name, problem_type="regression", path=None):
    if path is None:
        path = f"./assets/{problem_type}/datasets"

    X_train, y_train = load_from_ts_file(f"{path}/{name}/{name}_TRAIN.ts")
    X_test, y_test = load_from_ts_file(f"{path}/{name}/{name}_TEST.ts")
    if problem_type == "regression":
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test])

    return X, y

def list_datasets(problem_type="regression", path=None):
    if path is None:
        path = f"./assets/{problem_type}/datasets"

    datasets = os.listdir(path)
    return datasets

def load_metadataset(features="efficient", problem_type="regression", path=None):
    if path is None:
        path = f"./assets/{problem_type}"

    X = pd.read_csv(f"{path}/metadatasets/{features}.csv", index_col="name")
    y = pd.read_csv("{path}/best_estimators.csv")

    X = X.sort_values(by="name")
    y = y.sort_values(by="dataset")
    y = y["best_estimator"].to_numpy()

    # TODO: remove this line when results are available
    if problem_type == "regression":
        X = X.drop("AustraliaRainfall")

    return X, y

def write_metadataset(X, features="efficient", problem_type="regression", path=None):
    if path is None:
        path = f"./assets/{problem_type}"

    X.to_csv(f"{path}/metadatasets/{features}.csv")

def load_rmses(path="./assets/regression/rmse.csv"):
    rmse = pd.read_csv(path, index_col="regressor")
    return rmse