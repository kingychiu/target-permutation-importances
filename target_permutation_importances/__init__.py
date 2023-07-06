import gc
from functools import partial

import numpy as np
import pandas as pd
from beartype import beartype, vale
from beartype.typing import Any, Dict, List, Protocol, Union, runtime_checkable
from tqdm import tqdm
from typing_extensions import Annotated

XType = Union[np.ndarray, pd.DataFrame]
YType = Union[np.ndarray, pd.Series]
PositiveInt = Annotated[int, vale.Is[lambda x: x > 0]]


@runtime_checkable
class XBuilderType(Protocol):  # pragma: no cover
    def __call__(self, is_random_run: bool, run_idx: int) -> XType:
        ...


@runtime_checkable
class YBuilderType(Protocol):  # pragma: no cover
    def __call__(self, is_random_run: bool, run_idx: int) -> YType:
        ...


@runtime_checkable
class ModelBuilderType(Protocol):  # pragma: no cover
    def __call__(self, is_random_run: bool, run_idx: int) -> Any:
        ...


@runtime_checkable
class ModelFitterType(Protocol):  # pragma: no cover
    def __call__(self, model: Any, X: XType, y: YType) -> Any:
        ...


@runtime_checkable
class ModelImportanceCalculatorType(Protocol):  # pragma: no cover
    def __call__(self, model: Any, X: XType, y: YType) -> pd.DataFrame:
        ...


@runtime_checkable
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
    actual_importance_df = pd.concat(actual_importance_dfs)
    mean_actual_importance_df = actual_importance_df.groupby("feature").mean()
    std_actual_importance_df = actual_importance_df.groupby("feature").std()

    # Calculate the mean random importance
    random_importance_df = pd.concat(random_importance_dfs)
    mean_random_importance_df = random_importance_df.groupby("feature").mean()
    std_random_importance_df = random_importance_df.groupby("feature").std()

    # Sort by feature name to make sure the order is the same
    mean_actual_importance_df = mean_actual_importance_df.sort_index()
    std_actual_importance_df = std_actual_importance_df.sort_index()
    mean_random_importance_df = mean_random_importance_df.sort_index()
    std_random_importance_df = std_random_importance_df.sort_index()
    assert (mean_random_importance_df.index == mean_actual_importance_df.index).all()

    # Calculate the signal to noise ratio
    mean_actual_importance_df["mean_actual_importance"] = mean_actual_importance_df[
        "importance"
    ]
    mean_actual_importance_df["std_actual_importance"] = std_actual_importance_df[
        "importance"
    ]
    mean_actual_importance_df["mean_random_importance"] = mean_random_importance_df[
        "importance"
    ]
    mean_actual_importance_df["importance"] = mean_actual_importance_df[
        "importance"
    ] - (mean_random_importance_df["importance"])
    mean_actual_importance_df["std_random_importance"] = std_random_importance_df[
        "importance"
    ]
    return mean_actual_importance_df[
        [
            "importance",
            "mean_actual_importance",
            "mean_random_importance",
            "std_actual_importance",
            "std_random_importance",
        ]
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
    actual_importance_df = pd.concat(actual_importance_dfs)
    mean_actual_importance_df = actual_importance_df.groupby("feature").mean()
    std_actual_importance_df = actual_importance_df.groupby("feature").std()

    # Calculate the mean random importance
    random_importance_df = pd.concat(random_importance_dfs)
    mean_random_importance_df = random_importance_df.groupby("feature").mean()
    std_random_importance_df = random_importance_df.groupby("feature").std()

    # Sort by feature name to make sure the order is the same
    mean_actual_importance_df = mean_actual_importance_df.sort_index()
    std_actual_importance_df = std_actual_importance_df.sort_index()
    mean_random_importance_df = mean_random_importance_df.sort_index()
    std_random_importance_df = std_random_importance_df.sort_index()

    assert (mean_random_importance_df.index == mean_actual_importance_df.index).all()

    # Calculate the signal to noise ratio
    mean_actual_importance_df["mean_actual_importance"] = mean_actual_importance_df[
        "importance"
    ]
    mean_actual_importance_df["std_actual_importance"] = std_actual_importance_df[
        "importance"
    ]
    mean_actual_importance_df["mean_random_importance"] = mean_random_importance_df[
        "importance"
    ]
    mean_actual_importance_df["importance"] = mean_actual_importance_df[
        "importance"
    ] / (mean_random_importance_df["importance"] + 1)
    mean_actual_importance_df["std_random_importance"] = std_random_importance_df[
        "importance"
    ]
    return mean_actual_importance_df[
        [
            "importance",
            "mean_actual_importance",
            "mean_random_importance",
            "std_actual_importance",
            "std_random_importance",
        ]
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
    model = model_builder(is_random_run=is_random_run, run_idx=run_idx)
    X = X_builder(is_random_run=is_random_run, run_idx=run_idx)
    y = y_builder(is_random_run=is_random_run, run_idx=run_idx)

    model = model_fitter(model, X, y)
    gc.collect()
    return model_importance_calculator(model, X, y)


@beartype
def generic_compute(
    model_builder: ModelBuilderType,
    model_fitter: ModelFitterType,
    model_importance_calculator: ModelImportanceCalculatorType,
    permutation_importance_calculator: PermutationImportanceCalculatorType,
    X_builder: XBuilderType,
    y_builder: YBuilderType,
    num_actual_runs: PositiveInt = 2,
    num_random_runs: PositiveInt = 10,
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
    return permutation_importance_calculator(
        actual_importance_dfs, random_importance_dfs
    )


@beartype
def compute(
    model_cls: Any,
    model_cls_params: Dict,
    model_fit_params: Dict,
    X: XType,
    y: YType,
    num_actual_runs: PositiveInt = 2,
    num_random_runs: PositiveInt = 10,
    permutation_importance_calculator: PermutationImportanceCalculatorType = compute_permutation_importance_by_subtraction,
) -> pd.DataFrame:
    """
    Compute the permutation importance of a model given a dataset.

    Args:
        model_cls: The constructor/class of the model.
        model_cls_params: The parameters to pass to the model constructor.
        model_fit_params: The parameters to pass to the model fit method.
        X (pd.DataFrame, np.ndarray): The input data.
        y (pd.Series, np.ndarray): The target vector.
        num_actual_runs (int): Number of actual runs. Defaults to 2.
        num_random_runs (int): Number of random runs. Defaults to 10.
        permutation_importance_calculator: The function to compute the final importance. Defaults to compute_permutation_importance_by_subtraction.

    Returns:
        pd.DataFrame: The return DataFrame contain columns ["feature", "importance"]
    """

    def _x_builder(is_random_run: bool, run_idx: int) -> XType:
        return X

    def _y_builder(is_random_run: bool, run_idx: int) -> YType:
        rng = np.random.default_rng(seed=run_idx)
        if is_random_run:
            # Only shuffle the target for random runs
            return rng.permutation(y)
        return y

    def _model_builder(is_random_run: bool, run_idx: int) -> Any:
        # Model random state should be different for each run for both
        # actual and random runs
        _model_cls_params = model_cls_params.copy()
        _model_cls_params["random_state"] = run_idx
        return model_cls(**_model_cls_params)

    def _model_fitter(model: Any, X: XType, y: YType) -> Any:
        _model_fit_params = model_fit_params.copy()
        if "Cat" in str(model.__class__):
            _model_fit_params["verbose"] = False
        return model.fit(X, y, **_model_fit_params)

    def _model_importance_calculator(model: Any, X: XType, y: YType) -> pd.DataFrame:
        feature_attr = "feature_names_in_"
        if "LGBM" in str(model.__class__):
            feature_attr = "feature_name_"
        elif "Cat" in str(model.__class__):
            feature_attr = "feature_names_"

        if isinstance(X, pd.DataFrame):
            features = getattr(model, feature_attr)
        else:
            features = list(range(0, X.shape[1]))

        return pd.DataFrame(
            {
                "feature": features,
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
