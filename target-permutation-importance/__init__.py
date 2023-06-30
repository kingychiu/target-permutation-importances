from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

model_type = TypeVar("model_type", bound=Any)


def sklearn_rf_importance_calculator(
    model: model_type, x: pd.DataFrame, y: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame(
        {"feature": model.feature_names_in_, "importance": model.feature_importances_}
    )


def default_X_builder(
    is_random_rum: bool, run_idx: int, X: pd.DataFrame
) -> pd.DataFrame:
    return X


def default_y_builder(
    is_random_rum: bool, run_idx: int, y: pd.DataFrame
) -> pd.DataFrame:
    if is_random_rum:
        # When it is a random run, we want to shuffle the y
        return np.random.permutation(y)
    else:
        return y


def generic_compute(
    model_builder: Callable[..., model_type],
    model_fitter: Callable[[model_type, pd.DataFrame, pd.DataFrame], model_type],
    importance_calculator: Callable[
        [model_type, pd.DataFrame, pd.DataFrame], pd.DataFrame
    ],
    X_builder: Callable[[bool, int], pd.DataFrame] = default_X_builder,
    y_builder: Callable[[bool, int], pd.DataFrame] = default_y_builder,
    num_base_runs: int = 2,
    num_random_runs: int = 10,
):
    """_summary_

    Args:
        model_builder (Callable[..., model_type]):
            A function that builds a fresh model for a trial.
        model_fitter (Callable[[model_type, pd.DataFrame, pd.DataFrame], model_type]):
            A function that fits a model to the data.
        importance_calculator (
            Callable[[model_type, pd.DataFrame, pd.DataFrame], pd.DataFrame]
        ):
            A function that calculates the importance of the features.
        X_builder (
            Callable[[bool, int], pd.DataFrame], optional
        ):
            A function that builds a fresh X for a trial. Defaults to default_X_builder.
        y_builder (
            Callable[[bool, int], pd.DataFrame], optional
        ):
            A function that builds a fresh y for a trial. Defaults to default_y_builder
        num_base_runs (int, optional): Defaults to 2.
        num_random_runs (int, optional): Defaults to 10.
    """
    # Run the base runs
    base_importance_dfs = []
    for run_idx in range(num_base_runs):
        model = model_builder()
        X = X_builder(is_random_run=False, run_idx=run_idx)
        y = y_builder(is_random_run=False, run_idx=run_idx)
        model = model_fitter(model, X, y)
        importance_df = importance_calculator(model, X, y)
        base_importance_dfs.append(importance_df)

    # Run the random runs
    random_importance_dfs = []
    for run_idx in range(num_random_runs):
        model = model_builder()
        X = X_builder(is_random_run=True, run_idx=run_idx)
        y = y_builder(is_random_run=True, run_idx=run_idx)
        model = model_fitter(model, X, y)
        importance_df = importance_calculator(model, X, y)
        random_importance_dfs.append(importance_df)
