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
    """
    A function/callable that takes in a list of actual importance DataFrames
    and a list of random importance s and returns a single DataFrame

    Args:
        actual_importance_dfs (List[pd.DataFrame]): list of actual importance DataFrames with columns ["feature", "importance"]
        random_importance_dfs (List[pd.DataFrame]): list of random importance DataFrames with columns ["feature", "importance"]

    Returns:
        pd.DataFrame: The return DataFrame with columns ["feature", "importance"]
    """

    def __call__(
        self,
        actual_importance_dfs: List[pd.DataFrame],
        random_importance_dfs: List[pd.DataFrame],
    ) -> pd.DataFrame:
        ...


def compute_permutation_importance_by_subtraction(
    actual_importance_dfs: List[pd.DataFrame], random_importance_dfs: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Given a list of actual importance DataFrames and a list of random importance compute
    the permutation importance by $I_f = Avg(A_f) - Avg(R_f)$

    Args:
        actual_importance_dfs (List[pd.DataFrame]): list of random importance DataFrames with columns ["feature", "importance"]
        random_importance_dfs (List[pd.DataFrame]): list of random importance DataFrames with columns ["feature", "importance"]

    Returns:
        pd.DataFrame: The return DataFrame with columns ["feature", "importance"]
    """
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
    mean_actual_importance_df["importance"] = mean_actual_importance_df[
        "importance"
    ] - (mean_random_importance_df["importance"])
    mean_actual_importance_df["mean_actual_importance"] = mean_actual_importance_df[
        "importance"
    ]
    mean_actual_importance_df["mean_random_importance"] = mean_random_importance_df[
        "importance"
    ]
    return mean_actual_importance_df[
        ["importance", "mean_actual_importance", "mean_random_importance"]
    ].reset_index()


def compute_permutation_importance_by_division(
    actual_importance_dfs: List[pd.DataFrame], random_importance_dfs: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Given a list of actual importance DataFrames and a list of random importance compute
    the permutation importance by $I_f = Avg(A_f) / (Avg(R_f) + 1)$

    Args:
        actual_importance_dfs (List[pd.DataFrame]): list of random importance DataFrames with columns ["feature", "importance"]
        random_importance_dfs (List[pd.DataFrame]): list of random importance DataFrames with columns ["feature", "importance"]

    Returns:
        pd.DataFrame: The return DataFrame with columns ["feature", "importance"]
    """
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
    mean_actual_importance_df["importance"] = mean_actual_importance_df[
        "importance"
    ] / (mean_random_importance_df["importance"] + 1)
    mean_actual_importance_df["mean_actual_importance"] = mean_actual_importance_df[
        "importance"
    ]
    mean_actual_importance_df["mean_random_importance"] = mean_random_importance_df[
        "importance"
    ]
    return mean_actual_importance_df[
        ["importance", "mean_actual_importance", "mean_random_importance"]
    ].reset_index()


def _input_validation(X: XType, y: YType, num_actual_runs: int, num_random_runs: int):
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series) and not isinstance(y, np.ndarray):
        raise ValueError("y must be a pandas Series or a numpy array")
    if num_actual_runs <= 0 or num_random_runs <= 0:
        raise ValueError("num_actual_runs and num_random_runs must be positive")


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

    _input_validation(X, y, 1, 1)

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
) -> pd.DataFrame:
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
) -> pd.DataFrame:
    """
    Compute the permutation importance of a model given a dataset.

    Args:
        model_cls: The constructor/class of the model.
        model_cls_params: The parameters to pass to the model constructor.
        model_fit_params: The parameters to pass to the model fit method.
        X (pd.DataFrame): The input data.
        y (pd.Series, np.ndarray): The target vector.
        num_actual_runs: Number of actual runs. Defaults to 2.
        num_random_runs: Number of random runs. Defaults to 10.
        permutation_importance_calculator: The function to compute the final importance. Defaults to compute_permutation_importance_by_subtraction.

    Returns:
        pd.DataFrame: The return DataFrame contain columns ["feature", "importance"]
    """
    _input_validation(X, y, num_actual_runs, num_random_runs)

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
        if "Cat" in str(model.__class__):
            model_fit_params["verbose"] = False
        return model.fit(X, y, **model_fit_params)

    def _model_importance_calculator(model: Any, X: XType, y: YType) -> pd.DataFrame:
        feature_attr = "feature_names_in_"
        if "LGBM" in str(model.__class__):
            feature_attr = "feature_name_"
        elif "Cat" in str(model.__class__):
            feature_attr = "feature_names_"

        return pd.DataFrame(
            {
                "feature": getattr(model, feature_attr),
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
