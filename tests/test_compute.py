import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from target_permutation_importances import (
    compute,
    compute_permutation_importance_by_division,
    compute_permutation_importance_by_subtraction,
)

data = load_breast_cancer()


def test_compute_with_sklearn_by_subtraction():
    Xpd = pd.DataFrame(data.data, columns=data.feature_names)
    result_df = compute(
        model_cls=RandomForestClassifier,
        model_cls_params={
            "n_estimators": 1,
        },
        model_fit_params={},
        permutation_importance_calculator=compute_permutation_importance_by_subtraction,
        X=Xpd,
        y=data.target,
        num_actual_runs=2,
        num_random_runs=10,
    )
    assert isinstance(result_df, pd.DataFrame)


def test_compute_with_sklearn_by_division():
    Xpd = pd.DataFrame(data.data, columns=data.feature_names)
    result_df = compute(
        model_cls=RandomForestClassifier,
        model_cls_params={
            "n_estimators": 1,
        },
        model_fit_params={},
        permutation_importance_calculator=compute_permutation_importance_by_division,
        X=Xpd,
        y=data.target,
        num_actual_runs=2,
        num_random_runs=10,
    )
    assert isinstance(result_df, pd.DataFrame)


def test_compute_with_xgboost():
    pass


def test_compute_with_lightgbm():
    pass


def test_compute_with_catboost():
    pass
