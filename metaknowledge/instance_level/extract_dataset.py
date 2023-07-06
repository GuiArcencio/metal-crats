import pandas as pd
import numpy as np

from tsfresh.feature_extraction import extract_features

def characterize_dataset(X, y, feature_collection, label_features=True, problem_type="regression"):
    """
        Extract features (given by `feature_collection`) 
        from a single dataset (X, a 3D-array).
    """
    instances, dims, length = X.shape
    X_df = _convert_3D_array_to_tsfresh_dataframe(X)

    features = extract_features(
        X_df, 
        default_fc_parameters=feature_collection,
        column_id="id",
        column_kind="kind",
        column_sort="time",
        column_value="value"
    )
    features = features.fillna(0)

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
            final_features[f"mean_{feature_name_proper}"] = np.mean(all_dims)
            final_features[f"std_{feature_name_proper}"] = np.std(all_dims)
    
    # Label features
    if label_features:
        if problem_type == "regression":
            final_features["label_sum_values"] = np.sum(y)
            final_features["label_mean"] = np.mean(y)
            final_features["label_median"] = np.median(y)
            final_features["label_standard_deviation"] = np.std(y)
            final_features["label_variance"] = np.var(y)
            final_features["label_minimum"] = np.min(y)
            final_features["label_maximum"] = np.max(y)
            final_features["label_root_mean_square"] = np.sqrt(np.mean(y**2))
        else: # classification
            # TODO: add class features
            pass

    # General features  
    final_features["time_series_length"] = length
    final_features["number_examples"] = instances
    final_features["number_dimensions"] = dims

    return pd.DataFrame([final_features])

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
