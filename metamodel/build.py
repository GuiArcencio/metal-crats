import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

AVAILABLE_METAMODELS = [
    "1nn",
    "5nn",
    "nb",
    "rf",
    "svm"
]

def build_metamodel(option="1nn", random_state=None):
    assert(option in AVAILABLE_METAMODELS)

    feature_selector = VarianceThreshold((1e-5)**2)
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

    final_model = Pipeline([
        ("feature_selection", feature_selector),
        ("classifier", model)
    ])

    return final_model

def build_regression_baseline(rmses):
    preds = []
    for dataset in rmses.columns:
        preds.append(np.mean(rmses[dataset]))

    return np.array(preds)

def build_regression_topline(rmses):
    preds = []
    for dataset in rmses.columns:
        preds.append(np.min(rmses[dataset]))

    return np.array(preds)