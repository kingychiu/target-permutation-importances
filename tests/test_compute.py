from target_permutation_importances import compute
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data = load_breast_cancer()


def test_compute():
    Xpd = pd.DataFrame(data.data, columns=data.feature_names)
    result_df = compute(
        model_cls=RandomForestClassifier,
        model_cls_params={
            "n_estimators": 1,
        },
        model_fit_params={},
        X=Xpd,
        y=data.target,
        num_actual_runs=2,
        num_random_runs=10,
    )
    assert isinstance(result_df, pd.DataFrame)
