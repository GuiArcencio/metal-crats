import pandas as pd

from metaknowledge.instance_level import characterize_dataset
from dataio import load_dataset, list_datasets

def create_metadataset(feature_collection, problem_type="regression", use_label_features=True, path=None):
    """
        Create meta-knowledge from datasets in `path`
    """
    if path is None:
        path = f"./assets/{problem_type}/datasets"

    datasets = list_datasets(problem_type, path)
    meta_X = []
    for dataset in datasets:
        X, y = load_dataset(dataset, problem_type, path)
        meta_X.append(characterize_dataset(X, y, feature_collection, use_label_features, problem_type))
        del X, y

    meta_X = pd.concat(meta_X, ignore_index=True)
    meta_X['name'] = datasets
    meta_X = meta_X.set_index('name')

    return meta_X

def load_best_estimators(problem_type="regression", path=None):
    if path is None:
        path = f"./assets/{problem_type}/best_estimators.csv"

    estimators = pd.read_csv(path)
    return estimators


