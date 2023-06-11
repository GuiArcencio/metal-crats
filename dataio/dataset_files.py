import pandas as pd
import numpy as np
import os

from tsml.datasets import load_from_ts_file

def load_dataset(name, path="./assets/datasets"):
    X_train, y_train = load_from_ts_file(f"{path}/{name}/{name}_TRAIN.ts")
    X_test, y_test = load_from_ts_file(f"{path}/{name}/{name}_TEST.ts")
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test])

    return X, y

def list_datasets(path="./assets/datasets"):
    datasets = os.listdir(path)
    return datasets

def load_metadataset(X_filename="efficient.csv", path="./assets/metadataset"):
    X = pd.read_csv(f"{path}/{X_filename}", index_col="name")
    y = pd.read_csv("./assets/best_regressors.csv")

    X = X.sort_values(by="name")
    y = y.sort_values(by="dataset")
    y = y["best_regressor"].to_numpy()

    return X, y