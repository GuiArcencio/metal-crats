import pandas as pd

from tsfresh.feature_extraction import extract_features

def characterize_dataset(X, feature_collection):
    """
        Extract features (given by `feature_collection`) 
        from a single dataset (X, a 3D-array).
    """
    X_df = _convert_3D_array_to_tsfresh_dataframe(X)

    features = extract_features(
        X_df, 
        default_fc_parameters=feature_collection,
        column_id="id",
        column_kind="kind",
        column_sort="time",
        column_value="value"
    )

    return features

def _convert_3D_array_to_tsfresh_dataframe(X):
    """
        Converts a 3D numpy array to a DataFrame so
        that tsfresh can use it.
    """

    instances, dims, timepoints = X.shape
    id = list()
    time = list()
    kind = list()
    value = list()
    for instance in range(instances):
        for dim in range(dims):
            for t in range(timepoints):
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
