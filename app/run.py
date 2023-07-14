import pandas as pd

from dataio import check_metadataset_ready, load_metadataset, write_metadataset, load_rmses, load_accs

from metaknowledge import create_metadataset, FEATURES
from metamodel import run_model_using_loo, best_classifiers_to_acc, best_regressors_to_rmse

def reproduce_experiment(problem_type, features, metamodels, use_label_features):
    all_results = dict()

    for feature_collection in features:
        if check_metadataset_ready(problem_type, feature_collection):
            # Metadataset already created
            meta_X, meta_y = load_metadataset(feature_collection, problem_type)
        else:
            meta_X = create_metadataset(
                FEATURES[feature_collection],
                problem_type,
                use_label_features=True,
            )
            write_metadataset(meta_X, feature_collection, problem_type)
            meta_X, meta_y = load_metadataset(feature_collection, problem_type)

        if not use_label_features:
            to_drop = filter(lambda s: s.startswith("label_"), meta_X.columns)
            meta_X = meta_X.drop(to_drop, axis=1)

        if metamodels is None:
            continue
        
        for model in metamodels:
            meta_pred = run_model_using_loo(meta_X, meta_y, model, None)
            from sklearn.metrics import accuracy_score

            if problem_type == "regression":
                metric = "rmse"
                meta_pred_metric = best_regressors_to_rmse(meta_pred, load_rmses())
            else: # classification
                metric = "acc"
                meta_pred_metric = best_classifiers_to_acc(meta_pred, load_accs())

            results = pd.DataFrame({
                "dataset": meta_X.index,
                "best_estimator": meta_y,
                "predicted_estimator": meta_pred,
                metric: meta_pred_metric
            })
            results = results.set_index("dataset")

            label_string = "label" if use_label_features else "nolabel"
            all_results[f"{problem_type}_{label_string}_{feature_collection}_{model}"] = results

    return all_results