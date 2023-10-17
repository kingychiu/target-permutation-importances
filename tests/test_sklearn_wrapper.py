import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from target_permutation_importances.functional import (
    compute_permutation_importance_by_division,
    compute_permutation_importance_by_subtraction,
    compute_permutation_importance_by_wasserstein_distance,
)
from target_permutation_importances.sklearn_wrapper import (
    TargetPermutationImportancesWrapper,
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
    (Ridge, {"max_iter": 2}),
    (LinearSVC, {"max_iter": 2}),
]
X_TYPES = [pd.DataFrame, np.ndarray]
test_compute_clf_scope = []
for model_cls in CLF_MODEL_CLS:
    for imp_func in IMP_FUNCS:
        for xtype in X_TYPES:
            test_compute_clf_scope.append((model_cls, imp_func, xtype))


@pytest.mark.parametrize("model_cls,imp_func,xtype", test_compute_clf_scope)
def test_compute_binary_classification_and_SelectFromModel(model_cls, imp_func, xtype):
    data = load_breast_cancer()
    if xtype is pd.DataFrame:
        X = pd.DataFrame(
            data.data, columns=[f.replace(" ", "_") for f in data.feature_names]
        )
    else:
        X = data.data

    wrapped_model = TargetPermutationImportancesWrapper(
        model_cls=model_cls[0],
        model_cls_params=model_cls[1],
        permutation_importance_calculator=imp_func,
        num_actual_runs=5,
        num_random_runs=5,
    )
    wrapped_model.fit(
        X=X,
        y=data.target,
    )
    result_df = wrapped_model.feature_importances_df
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

    assert (
        wrapped_model.feature_importances_ == result_df["importance"].to_numpy()
    ).all()
    assert (wrapped_model.feature_names_in_ == result_df["feature"].to_numpy()).all()
    assert wrapped_model.n_features_in_ == result_df.shape[0]

    # Test with SelectFromModel prefit=True
    selector = SelectFromModel(
        estimator=wrapped_model, prefit=True, max_features=5, threshold=-np.inf
    ).fit(X, data.target)

    assert len(selector.get_support()) == X.shape[1]

    selected_x = selector.transform(X)
    selected_features = selector.get_feature_names_out()
    assert len(selected_features) == 5  # noqa
    assert selected_x.shape[1] == len(selected_features)
    # Assert the best 5 features are selected
    best_n_features = result_df.sort_values("importance", ascending=False)["feature"][
        :5
    ]

    assert set(selected_features) == set(best_n_features)

    if xtype is pd.DataFrame:
        assert (
            X[sorted(best_n_features)] - X[sorted(selected_features)]
        ).sum().sum() == 0
        assert (X[selected_features] - selected_x).sum().sum() == 0
    else:
        assert (
            X[:, sorted(best_n_features)] - X[:, sorted(selected_features)]
        ).sum().sum() == 0
        assert (X[:, selected_features] - selected_x).sum().sum() == 0
