import pandas as pd
import numpy as np

from tsml.datasets import load_from_ts_file

def load_dataset(name, path="./datasets"):
    X_train, y_train = load_from_ts_file(f"{path}/{name}/{name}_TRAIN.ts")
    X_test, y_test = load_from_ts_file(f"{path}/{name}/{name}_TEST.ts")
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    X = pd.concat(X_train, X_test, ignore_index=True)
    y = np.concatenate(y_train, y_test)

    return X, y
