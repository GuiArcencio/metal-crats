import pandas as pd
import numpy as np
import sys

from tqdm import tqdm
from tsfresh.feature_extraction import extract_features
from pycatch22 import catch22_all

def characterize_dataset(X, y, feature_collection, label_features=True, problem_type="regression"):
    """
        Extract features (given by `feature_collection`) 
        from a single dataset (X, a 3D-array).
    """
    instances, dims, length = X.shape

    if feature_collection == "catch22":
        features = list()
        for i in tqdm(range(instances), desc="Feature Extraction"):
            transformed_dims = list()
            for dim in range(dims):
                catch22_results = catch22_all(X[i,dim,:], catch24=True)
                transformed_dims.append(pd.Series(
                    catch22_results["values"],
                    index=[
                        f"{dim}__{feature_name}" 
                        for feature_name in catch22_results["names"]
                    ]
                ))
            features.append(pd.concat(transformed_dims))

        features = pd.DataFrame(features)
    else: # TSFresh
        X_df = _convert_3D_array_to_tsfresh_dataframe(X)

        features = extract_features(
            X_df, 
            default_fc_parameters=feature_collection,
            column_id="id",
            column_kind="kind",
            column_sort="time",
            column_value="value"
        )
        features = features.fillna(0.)
        features = features.replace([np.inf], 1e20)
        features = features.replace([-np.inf], -1e20)

    final_features = dict()
    for f in features.columns:
        feature_name_split = f.split("__")
        feature_name_proper = "__".join(feature_name_split[1:])
        if f"mean_{feature_name_proper}" not in final_features:
            all_dims = pd.concat(
                [features[f"{dim}__{feature_name_proper}"] 
                for dim in range(dims)], 
                ignore_index=True
            )
            if feature_name_proper == "length":
                final_features["time_series_length"] = int(all_dims[0])
            else:
                final_features[f"mean_{feature_name_proper}"] = np.mean(all_dims)
                final_features[f"std_{feature_name_proper}"] = np.std(all_dims)
                final_features[f"median_{feature_name_proper}"] = np.nanmedian(all_dims)
                final_features[f"q1_{feature_name_proper}"] = np.nanpercentile(all_dims, 25)
                final_features[f"q3_{feature_name_proper}"] = np.nanpercentile(all_dims, 75)
                final_features[f"max_{feature_name_proper}"] = np.max(all_dims)
                final_features[f"min_{feature_name_proper}"] = np.min(all_dims)
                

    
    # Label features
    if label_features:
        if problem_type == "regression":
            final_features["label_mean"] = np.mean(y)
            final_features["label_median"] = np.median(y)
            final_features["label_standard_deviation"] = np.std(y)
            final_features["label_minimum"] = np.min(y)
            final_features["label_maximum"] = np.max(y)
            final_features["label_root_mean_square"] = np.sqrt(np.mean(y**2))
        else: # classification
            classes, classes_counts = np.unique(y, return_counts=True)
            classes_counts = classes_counts / np.sum(classes_counts)

            final_features["label_class_count"] = len(classes)
            final_features["label_gini_impurity"] = 1 - np.sum(classes_counts**2)
            final_features["label_entropy"] = -np.sum(classes_counts * np.log2(classes_counts))

    # General features  
    final_features["number_examples"] = instances
    if feature_collection == "catch22":
        # tsfresh features already include ts length
        final_features["time_series_length"] = length
    if problem_type == "regression":
        # Classification datasets are all univariate
        final_features["number_dimensions"] = dims

    final_features = pd.DataFrame([final_features])
    return final_features

def _convert_3D_array_to_tsfresh_dataframe(X):
    """
        Converts a 3D numpy array to a DataFrame so
        that tsfresh can use it.
    """

    instances, dims, length = X.shape
    id = list()
    time = list()
    kind = list()
    value = list()
    for instance in range(instances):
        for dim in range(dims):
            for t in range(length):
                id.append(instance)
                time.append(t)
                kind.append(dim)
                value.append(X[instance, dim, t])

    return pd.DataFrame({
        "id": id,
        "time": time,
        "kind": kind,
        "value": value
    })
