from functools import partial
from typing import Any, Dict, List, Protocol, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

XType = pd.DataFrame
YType = Union[np.ndarray, pd.Series]


class XBuilderType(Protocol):  # pragma: no cover
    def __call__(self, is_random_run: bool, run_idx: int) -> XType:
        ...


class YBuilderType(Protocol):  # pragma: no cover
    def __call__(self, is_random_run: bool, run_idx: int) -> YType:
        ...


class ModelBuilderType(Protocol):  # pragma: no cover
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        ...


class ModelFitterType(Protocol):  # pragma: no cover
    def __call__(self, model: Any, X: XType, y: YType) -> Any:
        ...


class ModelImportanceCalculatorType(Protocol):  # pragma: no cover
    def __call__(self, model: Any, X: XType, y: YType) -> pd.DataFrame:
        ...


class PermutationImportanceCalculatorType(Protocol):  # pragma: no cover
    def __call__(
        self,
        actual_importance_dfs: List[pd.DataFrame],
        random_importance_dfs: List[pd.DataFrame],
    ) -> pd.DataFrame:
        ...


def compute_permutation_importance_by_subtraction(
    actual_importance_dfs: List[pd.DataFrame], random_importance_dfs: List[pd.DataFrame]
) -> pd.DataFrame:
    # Calculate the mean importance
    mean_actual_importance_df = (
        pd.concat(actual_importance_dfs).groupby("feature").mean()
    )
    # Calculate the mean random importance
    mean_random_importance_df = (
        pd.concat(random_importance_dfs).groupby("feature").mean()
    )
    # Sort by feature name to make sure the order is the same
    mean_actual_importance_df = mean_actual_importance_df.sort_index()
    mean_random_importance_df = mean_random_importance_df.sort_index()
    assert (mean_random_importance_df.index == mean_actual_importance_df.index).all()

    # Calculate the signal to noise ratio
    mean_actual_importance_df["permutation_importance"] = mean_actual_importance_df[
        "importance"
    ] - (mean_random_importance_df["importance"])
    mean_actual_importance_df["mean_actual_importance"] = mean_actual_importance_df[
        "importance"
    ]
    mean_actual_importance_df["mean_random_importance"] = mean_random_importance_df[
        "importance"
    ]
    return mean_actual_importance_df[
        ["permutation_importance", "mean_actual_importance", "mean_random_importance"]
    ].reset_index()


def compute_permutation_importance_by_division(
    actual_importance_dfs: List[pd.DataFrame], random_importance_dfs: List[pd.DataFrame]
) -> pd.DataFrame:
    # Calculate the mean importance
    mean_actual_importance_df = (
        pd.concat(actual_importance_dfs).groupby("feature").mean()
    )
    # Calculate the mean random importance
    mean_random_importance_df = (
        pd.concat(random_importance_dfs).groupby("feature").mean()
    )
    # Sort by feature name to make sure the order is the same
    mean_actual_importance_df = mean_actual_importance_df.sort_index()
    mean_random_importance_df = mean_random_importance_df.sort_index()
    assert (mean_random_importance_df.index == mean_actual_importance_df.index).all()

    # MinMax scale the random importance + 1
    random_min = mean_random_importance_df["importance"].min()
    random_max = mean_random_importance_df["importance"].max()
    mean_random_importance_df["importance"] -= random_min
    mean_random_importance_df["importance"] /= random_max - random_min
    mean_random_importance_df["importance"] += 1

    # Calculate the signal to noise ratio
    mean_actual_importance_df["permutation_importance"] = mean_actual_importance_df[
        "importance"
    ] / (mean_random_importance_df["importance"])
    mean_actual_importance_df["mean_actual_importance"] = mean_actual_importance_df[
        "importance"
    ]
    mean_actual_importance_df["mean_random_importance"] = mean_random_importance_df[
        "importance"
    ]

    return mean_actual_importance_df[
        ["permutation_importance", "mean_actual_importance", "mean_random_importance"]
    ].reset_index()


def _compute_one_run(
    model_builder: ModelBuilderType,
    model_fitter: ModelFitterType,
    model_importance_calculator: ModelImportanceCalculatorType,
    X_builder: XBuilderType,
    y_builder: YBuilderType,
    is_random_run: bool,
    run_idx: int,
):
    model = model_builder()
    X = X_builder(is_random_run=is_random_run, run_idx=run_idx)
    y = y_builder(is_random_run=is_random_run, run_idx=run_idx)
    model = model_fitter(model, X, y)
    return model_importance_calculator(model, X, y)


def generic_compute(
    model_builder: ModelBuilderType,
    model_fitter: ModelFitterType,
    model_importance_calculator: ModelImportanceCalculatorType,
    permutation_importance_calculator: PermutationImportanceCalculatorType,
    X_builder: XBuilderType,
    y_builder: YBuilderType,
    num_actual_runs: int = 2,
    num_random_runs: int = 10,
):
    run_params = {
        "model_builder": model_builder,
        "model_fitter": model_fitter,
        "model_importance_calculator": model_importance_calculator,
        "X_builder": X_builder,
        "y_builder": y_builder,
    }
    partial_compute_one_run = partial(_compute_one_run, **run_params)
    # Run the base runs
    print(f"Running {num_actual_runs} actual runs and {num_random_runs} random runs")
    actual_importance_dfs = []
    for run_idx in tqdm(range(num_actual_runs)):
        actual_importance_dfs.append(
            partial_compute_one_run(
                is_random_run=False,
                run_idx=run_idx,
            )
        )

    # Run the random runs
    random_importance_dfs = []
    for run_idx in tqdm(range(num_random_runs)):
        random_importance_dfs.append(
            partial_compute_one_run(
                is_random_run=True,
                run_idx=run_idx,
            )
        )

    # Calculate the permutation importance
    permutation_importance_df = permutation_importance_calculator(
        actual_importance_dfs, random_importance_dfs
    )

    return permutation_importance_df


def compute(
    model_cls: Any,
    model_cls_params: Dict,
    model_fit_params: Dict,
    X: XType,
    y: YType,
    num_actual_runs: int = 2,
    num_random_runs: int = 10,
    permutation_importance_calculator: PermutationImportanceCalculatorType = compute_permutation_importance_by_subtraction,  # noqa
):
    def _x_builder(is_random_run: bool, run_idx: int) -> XType:
        return X

    def _y_builder(is_random_run: bool, run_idx: int) -> YType:
        np.random.seed(run_idx)
        if is_random_run:
            return np.random.permutation(y)
        return y

    def _model_builder() -> Any:
        return model_cls(**model_cls_params)

    def _model_fitter(model: Any, X: XType, y: YType) -> Any:
        return model.fit(X, y, **model_fit_params)

    def _model_importance_calculator(model: Any, X: XType, y: YType) -> pd.DataFrame:
        feature_attr = "feature_names_in_"
        if "LGBM" in str(model.__class__):
            feature_attr = "feature_name_"
        elif "Cat" in str(model.__class__):
            feature_attr = "feature_names_"

        return pd.DataFrame(
            {
                "feature": feature_attr,
                "importance": model.feature_importances_,
            }
        )

    return generic_compute(
        model_builder=_model_builder,
        model_fitter=_model_fitter,
        model_importance_calculator=_model_importance_calculator,
        permutation_importance_calculator=permutation_importance_calculator,
        X_builder=_x_builder,
        y_builder=_y_builder,
        num_actual_runs=num_actual_runs,
        num_random_runs=num_random_runs,
    )
