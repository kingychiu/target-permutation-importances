import pandas as pd
import pytest
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from target_permutation_importances import (
    compute,
    compute_permutation_importance_by_division,
    compute_permutation_importance_by_subtraction,
)

data = load_breast_cancer()
IMP_FUNCS = [
    compute_permutation_importance_by_subtraction,
    compute_permutation_importance_by_division,
]
MODEL_CLS = [RandomForestClassifier, XGBClassifier, CatBoostClassifier, LGBMClassifier]


test_compute_scope = []
for model_cls in MODEL_CLS:
    for imp_func in IMP_FUNCS:
        test_compute_scope.append((model_cls, imp_func))


@pytest.mark.parametrize("model_cls,imp_func", test_compute_scope)
def test_compute(model_cls, imp_func):
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
        num_actual_runs=2,
        num_random_runs=10,
    )
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape[0] == Xpd.shape[1]
    assert "importance" in result_df.columns
    assert "feature" in result_df.columns
    assert set(result_df["feature"].tolist()) == set(Xpd.columns.tolist())
    assert result_df["importance"].isna().sum() == 0
