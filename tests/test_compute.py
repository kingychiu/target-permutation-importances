import numpy as np
import pandas as pd
import pytest
from beartype import roar
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from target_permutation_importances import (
    compute,
    compute_permutation_importance_by_division,
    compute_permutation_importance_by_subtraction,
)

IMP_FUNCS = [
    compute_permutation_importance_by_subtraction,
    compute_permutation_importance_by_division,
]
CLF_MODEL_CLS = [
    RandomForestClassifier,
    XGBClassifier,
    CatBoostClassifier,
    LGBMClassifier,
]
REG_MODEL_CLS = [RandomForestRegressor, XGBRegressor, CatBoostRegressor, LGBMRegressor]
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
        model_cls=model_cls,
        model_cls_params={
            "n_estimators": 2,
        },
        model_fit_params={},
        permutation_importance_calculator=imp_func,
        X=X,
        y=data.target,
        num_actual_runs=5,
        num_random_runs=10,
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
        model_cls=model_cls,
        model_cls_params={
            "n_estimators": 2,
        },
        model_fit_params={},
        permutation_importance_calculator=imp_func,
        X=X,
        y=data.target,
        num_actual_runs=5,
        num_random_runs=10,
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


def test_invalid_compute():
    data = load_diabetes()
    Xpd = pd.DataFrame(
        data.data, columns=[f.replace(" ", "_") for f in data.feature_names]
    )
    with pytest.raises(roar.BeartypeCallHintParamViolation):
        compute(
            model_cls=RandomForestClassifier,
            model_cls_params={},
            model_fit_params={},
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
            model_fit_params={},
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
            model_fit_params={},
            permutation_importance_calculator=compute_permutation_importance_by_subtraction,
            X=Xpd,
            y=data.target,
            num_actual_runs=-1,
            num_random_runs=10,
        )
