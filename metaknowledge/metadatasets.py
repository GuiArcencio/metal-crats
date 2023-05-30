import pandas as pd

from metaknowledge.instance_level import characterize_dataset
from dataio import load_dataset, list_datasets

def create_metadataset(feature_collection, path="./assets/datasets"):
    """
        Create meta-knowledge from datasets in `path`
    """
    datasets = list_datasets(path)
    meta_X = []
    for dataset in datasets:
        X, y = load_dataset(dataset, path)
        meta_X.append(characterize_dataset(X, y, feature_collection))
        del X, y

    meta_X = pd.concat(meta_X, ignore_index=True)
    meta_X['name'] = datasets
    meta_X = meta_X.set_index('name')

    return meta_X

def load_best_regressors(path="./assets/best_regressors.csv"):
    regressors = pd.read_csv(path)
    return regressors


