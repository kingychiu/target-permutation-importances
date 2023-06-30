from typing import Any, List, Protocol, Union

import numpy as np
import pandas as pd

XType = pd.DataFrame
YType = Union[np.ndarray, pd.Series]


class XBuilderType(Protocol):
    def __call__(self, is_random_run: bool, run_idx: int) -> XType:
        ...


class YBuilderType(Protocol):
    def __call__(self, is_random_run: bool, run_idx: int) -> YType:
        ...


class ModelBuilderType(Protocol):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        ...


class ModelFitterType(Protocol):
    def __call__(self, model: Any, X: XType, y: YType) -> Any:
        ...


class ModelImportanceCalculatorType(Protocol):
    def __call__(self, model: Any, X: XType, y: YType) -> pd.DataFrame:
        ...


class PermutationImportanceCalculatorType(Protocol):
    def __call__(
        self,
        base_importance_dfs: List[pd.DataFrame],
        random_importance_dfs: List[pd.DataFrame],
    ) -> pd.DataFrame:
        ...


def sklearn_importance_calculator(
    model: Any, x: pd.DataFrame, y: YType
) -> pd.DataFrame:
    return pd.DataFrame(
        {"feature": model.feature_names_in_, "importance": model.feature_importances_}
    )


def default_permutation_importance_calculator(
    base_importance_dfs: List[pd.DataFrame], random_importance_dfs: List[pd.DataFrame]
) -> pd.DataFrame:
    # Calculate the mean importance
    mean_base_importance_df = pd.concat(base_importance_dfs).groupby("feature").mean()
    # Calculate the mean random importance
    mean_random_importance_df = (
        pd.concat(random_importance_dfs).groupby("feature").mean()
    )
    # Add 1 to the random importance to avoid division by 0 and scaling up
    mean_base_importance_df["importance"] = mean_base_importance_df["importance"] / (
        mean_random_importance_df["importance"] + 1
    )
    return mean_base_importance_df


def generic_compute(
    model_builder: ModelBuilderType,
    model_fitter: ModelFitterType,
    model_importance_calculator: ModelImportanceCalculatorType,
    permutation_importance_calculator: PermutationImportanceCalculatorType,
    X_builder: XBuilderType,
    y_builder: YBuilderType,
    num_base_runs: int = 2,
    num_random_runs: int = 10,
):
    # Run the base runs
    base_importance_dfs = []
    for run_idx in range(num_base_runs):
        model = model_builder()
        X = X_builder(is_random_run=False, run_idx=run_idx)
        y = y_builder(is_random_run=False, run_idx=run_idx)
        model = model_fitter(model, X, y)
        importance_df = model_importance_calculator(model, X, y)
        base_importance_dfs.append(importance_df)

    # Run the random runs
    random_importance_dfs = []
    for run_idx in range(num_random_runs):
        model = model_builder()
        X = X_builder(is_random_run=True, run_idx=run_idx)
        y = y_builder(is_random_run=True, run_idx=run_idx)
        model = model_fitter(model, X, y)
        importance_df = model_importance_calculator(model, X, y)
        random_importance_dfs.append(importance_df)

    # Calculate the permutation importance
    permutation_importance_df = permutation_importance_calculator(
        base_importance_dfs, random_importance_dfs
    )

    return permutation_importance_df
