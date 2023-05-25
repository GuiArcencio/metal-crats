from tsml.datasets import load_from_ts_file

def load_dataset(name, path="./datasets"):
    X, y = load_from_ts_file(f"{path}/{name}")
    y = y.astype(float)
    return X, y
