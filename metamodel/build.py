import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

AVAILABLE_METAMODELS = [
    "1nn",
    "5nn",
    "nb",
    "rf",
    "svm",
    "xgb"
]

def build_metamodel(option="1nn", random_state=None):
    variance_selector = VarianceThreshold((1e-5)**2)
    if option == "1nn":
        model = Pipeline([
            ("normalizer", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=1))
        ])
    elif option == "5nn":
        model = Pipeline([
            ("normalizer", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ])
    elif option == "nb":
        model = GaussianNB()
    elif option == "rf":
        model = RandomForestClassifier(random_state=random_state)
    elif option == "svm":
        model = Pipeline([
            ("normalizer", StandardScaler()),
            ("model", SVC())
        ])
    elif option == "xgb":
        model = xgb.XGBClassifier(seed=random_state)

    final_model = Pipeline([
        ("variance_selection", variance_selector),
        ("classifier", model)
    ])

    return final_model

def build_baseline(results):
    preds = []
    for dataset in sorted(results.columns):
        preds.append(np.mean(results[dataset]))

    return np.array(preds)

def build_regression_topline(rmses):
    preds = []
    for dataset in sorted(rmses.columns):
        preds.append(np.min(rmses[dataset]))

    return np.array(preds)

def build_classification_topline(accs):
    preds = []
    for dataset in sorted(accs.columns):
        preds.append(np.max(accs[dataset]))

    return np.array(preds)