import numpy as np
import pandas as pd
import pytest
from beartype import roar
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import wasserstein_distance  # type: ignore
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_wine,
    make_multilabel_classification,
    make_regression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier, XGBRegressor

from target_permutation_importances.functional import (
    compute,
    compute_permutation_importance_by_division,
    compute_permutation_importance_by_subtraction,
    compute_permutation_importance_by_wasserstein_distance,
)

IMP_FUNCS = [
    compute_permutation_importance_by_subtraction,
    compute_permutation_importance_by_division,
    compute_permutation_importance_by_wasserstein_distance,
]
CLF_MODEL_CLS = [
    (RandomForestClassifier, {"n_estimators": 2, "n_jobs": 1}),
    (XGBClassifier, {"n_estimators": 2, "n_jobs": 1}),
    (CatBoostClassifier, {"n_estimators": 2}),
    (LGBMClassifier, {"n_estimators": 2, "n_jobs": 1}),
    (Lasso, {"max_iter": 2}),
    (LinearSVC, {"max_iter": 2}),
]
REG_MODEL_CLS = [
    (RandomForestRegressor, {"n_estimators": 2, "n_jobs": 1}),
    (XGBRegressor, {"n_estimators": 2, "n_jobs": 1}),
    (CatBoostRegressor, {"n_estimators": 2}),
    (LGBMRegressor, {"n_estimators": 2, "n_jobs": 1}),
]
X_TYPES = [pd.DataFrame, np.ndarray]
test_compute_clf_scope = []
for model_cls in CLF_MODEL_CLS:
    for imp_func in IMP_FUNCS:
        for xtype in X_TYPES:
            test_compute_clf_scope.append((model_cls, imp_func, xtype))

test_compute_reg_scope = []
for model_cls in REG_MODEL_CLS:
    for imp_func in IMP_FUNCS:
        for xtype in X_TYPES:
            test_compute_reg_scope.append((model_cls, imp_func, xtype))


@pytest.mark.parametrize("model_cls,imp_func,xtype", test_compute_clf_scope)
def test_compute_binary_classification(model_cls, imp_func, xtype):
    data = load_breast_cancer()
    if xtype is pd.DataFrame:
        X = pd.DataFrame(
            data.data, columns=[f.replace(" ", "_") for f in data.feature_names]
        )
    else:
        X = data.data
    result_df = compute(
        model_cls=model_cls[0],
        model_cls_params=model_cls[1],
        model_fit_params=lambda _: {},
        permutation_importance_calculator=imp_func,
        X=X,
        y=data.target,
        num_actual_runs=5,
        num_random_runs=5,
    )
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape[0] == X.shape[1]
    assert "importance" in result_df.columns
    assert "feature" in result_df.columns

    if xtype is pd.DataFrame:
        assert set(result_df["feature"].tolist()) == set(X.columns.tolist())
    else:
        assert set(result_df["feature"].tolist()) == set(range(data.data.shape[1]))
    assert result_df["importance"].isna().sum() == 0
    assert result_df["std_random_importance"].mean() > 0


@pytest.mark.parametrize("model_cls,imp_func,xtype", test_compute_clf_scope)
def test_compute_multi_class_classification(model_cls, imp_func, xtype):
    data = load_wine()
    if xtype is pd.DataFrame:
        X = pd.DataFrame(
            data.data, columns=[f.replace(" ", "_") for f in data.feature_names]
        )
    else:
        X = data.data

    result_df = compute(
        model_cls=model_cls[0],
        model_cls_params=model_cls[1],
        model_fit_params=lambda _: {},
        permutation_importance_calculator=imp_func,
        X=X,
        y=data.target,
        num_actual_runs=5,
        num_random_runs=5,
    )
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape[0] == X.shape[1]
    assert "importance" in result_df.columns
    assert "feature" in result_df.columns

    if xtype is pd.DataFrame:
        assert set(result_df["feature"].tolist()) == set(X.columns.tolist())
    else:
        assert set(result_df["feature"].tolist()) == set(range(data.data.shape[1]))
    assert result_df["importance"].isna().sum() == 0


@pytest.mark.parametrize("model_cls,imp_func,xtype", test_compute_clf_scope)
def test_compute_multi_label_classification(model_cls, imp_func, xtype):
    X, y = make_multilabel_classification(
        n_samples=500, n_features=20, n_classes=5, n_labels=2
    )
    if xtype is pd.DataFrame:
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    model_cls_params = model_cls[1].copy()
    if "Cat" in model_cls[0].__name__:
        model_cls_params["objective"] = "MultiLogloss"
    # LGBMClassifier does not support multi-label classification
    elif (
        "LGBM" in model_cls[0].__name__
        or "Lasso" in model_cls[0].__name__
        or "LinearSVC" in model_cls[0].__name__
    ):
        return

    result_df = compute(
        model_cls=model_cls[0],
        model_cls_params=model_cls_params,
        model_fit_params=lambda _: {},
        permutation_importance_calculator=imp_func,
        X=X,
        y=y,
        num_actual_runs=5,
        num_random_runs=5,
    )
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape[0] == X.shape[1]
    assert "importance" in result_df.columns
    assert "feature" in result_df.columns

    if xtype is pd.DataFrame:
        assert set(result_df["feature"].tolist()) == set(X.columns.tolist())
    else:
        assert set(result_df["feature"].tolist()) == set(range(X.shape[1]))
    assert result_df["importance"].isna().sum() == 0


@pytest.mark.parametrize("model_cls,imp_func,xtype", test_compute_clf_scope)
def test_compute_multi_label_classification_with_MultiOutputClassifier(
    model_cls, imp_func, xtype
):
    X, y = make_multilabel_classification(
        n_samples=500, n_features=20, n_classes=3, n_labels=3
    )
    if xtype is pd.DataFrame:
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    if "Lasso" in model_cls[0].__name__ or "LinearSVC" in model_cls[0].__name__:
        return

    result_df = compute(
        model_cls=MultiOutputClassifier,
        model_cls_params={
            "estimator": model_cls[0](**model_cls[1]),
        },
        model_fit_params=lambda _: {},
        permutation_importance_calculator=imp_func,
        X=X,
        y=y,
        num_actual_runs=5,
        num_random_runs=5,
    )
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape[0] == X.shape[1]
    assert "importance" in result_df.columns
    assert "feature" in result_df.columns

    if xtype is pd.DataFrame:
        assert set(result_df["feature"].tolist()) == set(X.columns.tolist())
    else:
        assert set(result_df["feature"].tolist()) == set(range(X.shape[1]))
    assert result_df["importance"].isna().sum() == 0


@pytest.mark.parametrize("model_cls,imp_func,xtype", test_compute_reg_scope)
def test_compute_regression(model_cls, imp_func, xtype):
    data = load_diabetes()
    if xtype is pd.DataFrame:
        X = pd.DataFrame(
            data.data, columns=[f.replace(" ", "_") for f in data.feature_names]
        )
    else:
        X = data.data
    result_df = compute(
        model_cls=model_cls[0],
        model_cls_params=model_cls[1],
        model_fit_params=lambda _: {},
        permutation_importance_calculator=imp_func,
        X=X,
        y=data.target,
        num_actual_runs=5,
        num_random_runs=5,
    )
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape[0] == X.shape[1]
    assert "importance" in result_df.columns
    assert "feature" in result_df.columns
    if xtype is pd.DataFrame:
        assert set(result_df["feature"].tolist()) == set(X.columns.tolist())
    else:
        assert set(result_df["feature"].tolist()) == set(range(data.data.shape[1]))
    assert result_df["importance"].isna().sum() == 0
    assert result_df["std_actual_importance"].isna().sum() == 0
    assert result_df["std_random_importance"].mean() > 0


@pytest.mark.parametrize("model_cls,imp_func,xtype", test_compute_reg_scope)
def test_compute_multi_target_regression_with_MultiOutputRegressor(
    model_cls, imp_func, xtype
):
    X, y = make_regression(
        n_samples=100,
        n_features=20,
        n_targets=3,
    )
    if xtype is pd.DataFrame:
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    result_df = compute(
        model_cls=MultiOutputRegressor,
        model_cls_params={
            "estimator": model_cls[0](**model_cls[1]),
        },
        model_fit_params=lambda _: {},
        permutation_importance_calculator=imp_func,
        X=X,
        y=y,
        num_actual_runs=5,
        num_random_runs=5,
    )
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape[0] == X.shape[1]
    assert "importance" in result_df.columns
    assert "feature" in result_df.columns

    if xtype is pd.DataFrame:
        assert set(result_df["feature"].tolist()) == set(X.columns.tolist())
    else:
        assert set(result_df["feature"].tolist()) == set(range(X.shape[1]))
    assert result_df["importance"].isna().sum() == 0


def test_compute_with_multiple_importance_functions():
    data = load_breast_cancer()
    X = pd.DataFrame(
        data.data, columns=[f.replace(" ", "_") for f in data.feature_names]
    )
    result_dfs = compute(
        model_cls=RandomForestClassifier,
        model_cls_params={"n_estimators": 2, "n_jobs": 1},
        model_fit_params=lambda _: {},
        permutation_importance_calculator=[
            compute_permutation_importance_by_subtraction,
            compute_permutation_importance_by_division,
            compute_permutation_importance_by_wasserstein_distance,
        ],
        X=X,
        y=data.target,
        num_actual_runs=5,
        num_random_runs=5,
        shuffle_feature_order=True,
    )
    assert isinstance(result_dfs, list)
    for result_df in result_dfs:
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.shape[0] == X.shape[1]
        assert "importance" in result_df.columns
        assert "feature" in result_df.columns
        assert set(result_df["feature"].tolist()) == set(X.columns.tolist())
        assert result_df["importance"].isna().sum() == 0
        assert result_df["std_random_importance"].mean() > 0


def test_compute_permutation_importance():
    actual_importance_dfs = [
        pd.DataFrame({"feature": ["a", "b"], "importance": [1, 2]}),
        pd.DataFrame({"feature": ["b", "a"], "importance": [1, 3]}),
    ]
    random_importance_dfs = [
        pd.DataFrame({"feature": ["a", "b"], "importance": [4, 2]}),
        pd.DataFrame({"feature": ["b", "a"], "importance": [1, 2]}),
        pd.DataFrame({"feature": ["b", "a"], "importance": [5, 2]}),
    ]

    result_df = compute_permutation_importance_by_subtraction(
        actual_importance_dfs, random_importance_dfs
    )

    assert result_df["feature"].tolist() == ["a", "b"]
    assert result_df["mean_actual_importance"].tolist() == [(1 + 3) / 2, (2 + 1) / 2]
    assert result_df["mean_random_importance"].tolist() == [
        (4 + 2 + 2) / 3,
        (2 + 1 + 5) / 3,
    ]
    assert result_df["importance"].tolist() == [
        (1 + 3) / 2 - (4 + 2 + 2) / 3,
        (2 + 1) / 2 - (2 + 1 + 5) / 3,
    ]

    result_df = compute_permutation_importance_by_division(
        actual_importance_dfs, random_importance_dfs
    )

    assert result_df["feature"].tolist() == ["a", "b"]
    assert result_df["mean_actual_importance"].tolist() == [(1 + 3) / 2, (2 + 1) / 2]
    assert result_df["mean_random_importance"].tolist() == [
        (4 + 2 + 2) / 3,
        (2 + 1 + 5) / 3,
    ]
    assert result_df["importance"].tolist() == [
        (1 + 3) / 2 / ((4 + 2 + 2) / 3 + 1),
        (2 + 1) / 2 / ((2 + 1 + 5) / 3 + 1),
    ]

    result_df = compute_permutation_importance_by_wasserstein_distance(
        actual_importance_dfs, random_importance_dfs
    )
    assert result_df["feature"].tolist() == ["a", "b"]
    assert result_df["mean_actual_importance"].tolist() == [(1 + 3) / 2, (2 + 1) / 2]
    assert result_df["mean_random_importance"].tolist() == [
        (4 + 2 + 2) / 3,
        (2 + 1 + 5) / 3,
    ]
    assert result_df["wasserstein_distance"].tolist() == [
        wasserstein_distance([1, 3], [4, 2, 2]),
        wasserstein_distance([2, 1], [2, 1, 5]),
    ]
    assert (result_df["importance"] == result_df["wasserstein_distance"]).all()


def test_invalid_compute():
    data = load_diabetes()
    Xpd = pd.DataFrame(
        data.data, columns=[f.replace(" ", "_") for f in data.feature_names]
    )
    with pytest.raises(roar.BeartypeCallHintParamViolation):
        compute(
            model_cls=RandomForestClassifier,
            model_cls_params={},
            model_fit_params=lambda _: {},
            permutation_importance_calculator=compute_permutation_importance_by_subtraction,
            X=1,
            y=data.target,
            num_actual_runs=2,
            num_random_runs=10,
        )
    with pytest.raises(roar.BeartypeCallHintParamViolation):
        compute(
            model_cls=RandomForestClassifier,
            model_cls_params={},
            model_fit_params=lambda _: {},
            permutation_importance_calculator=compute_permutation_importance_by_subtraction,
            X=Xpd,
            y=1,
            num_actual_runs=2,
            num_random_runs=10,
        )
    with pytest.raises(roar.BeartypeCallHintParamViolation):
        compute(
            model_cls=RandomForestClassifier,
            model_cls_params={},
            model_fit_params=lambda _: {},
            permutation_importance_calculator=compute_permutation_importance_by_subtraction,
            X=Xpd,
            y=data.target,
            num_actual_runs=-1,
            num_random_runs=10,
        )
