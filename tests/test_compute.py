import pandas as pd
import pytest
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

test_compute_clf_scope = []
for model_cls in CLF_MODEL_CLS:
    for imp_func in IMP_FUNCS:
        test_compute_clf_scope.append((model_cls, imp_func))

test_compute_reg_scope = []
for model_cls in REG_MODEL_CLS:
    for imp_func in IMP_FUNCS:
        test_compute_reg_scope.append((model_cls, imp_func))


@pytest.mark.parametrize("model_cls,imp_func", test_compute_clf_scope)
def test_compute_binary_classification(model_cls, imp_func):
    data = load_breast_cancer()
    Xpd = pd.DataFrame(
        data.data, columns=[f.replace(" ", "_") for f in data.feature_names]
    )

    result_df = compute(
        model_cls=model_cls,
        model_cls_params={
            "n_estimators": 1,
        },
        model_fit_params={},
        permutation_importance_calculator=imp_func,
        X=Xpd,
        y=data.target,
        num_actual_runs=5,
        num_random_runs=20,
    )
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape[0] == Xpd.shape[1]
    assert "importance" in result_df.columns
    assert "feature" in result_df.columns
    assert set(result_df["feature"].tolist()) == set(Xpd.columns.tolist())
    assert result_df["importance"].isna().sum() == 0


@pytest.mark.parametrize("model_cls,imp_func", test_compute_reg_scope)
def test_compute_regression(model_cls, imp_func):
    data = load_diabetes()
    Xpd = pd.DataFrame(
        data.data, columns=[f.replace(" ", "_") for f in data.feature_names]
    )

    result_df = compute(
        model_cls=model_cls,
        model_cls_params={
            "n_estimators": 1,
        },
        model_fit_params={},
        permutation_importance_calculator=imp_func,
        X=Xpd,
        y=data.target,
        num_actual_runs=5,
        num_random_runs=20,
    )
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape[0] == Xpd.shape[1]
    assert "importance" in result_df.columns
    assert "feature" in result_df.columns
    assert set(result_df["feature"].tolist()) == set(Xpd.columns.tolist())
    assert result_df["importance"].isna().sum() == 0
