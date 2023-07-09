import numpy as np
import pandas as pd
from beartype import vale
from beartype.typing import Any, List, Union, runtime_checkable
from typing_extensions import Annotated, Protocol

XType = Union[np.ndarray, pd.DataFrame]
YType = Union[np.ndarray, pd.Series]
PositiveInt = Annotated[int, vale.Is[lambda x: x > 0]]


@runtime_checkable
class XBuilderType(Protocol):  # pragma: no cover
    """
    A function/callable that return X data. This function is called once per run (actual and random)

    Args:
        is_random_run (bool): Indicate if this is a random run
        run_idx (int): The run index
    Returns:
        return (XType): The X data
    """

    def __call__(self, is_random_run: bool, run_idx: int) -> XType:
        ...


@runtime_checkable
class YBuilderType(Protocol):  # pragma: no cover
    """
    A function/callable that return Y data. This function is called once per run (actual and random)

    Args:
        is_random_run (bool): Indicate if this is a random run
        run_idx (int): The run index
    Returns:
        return (YType): The y data
    """

    def __call__(self, is_random_run: bool, run_idx: int) -> YType:
        ...


@runtime_checkable
class ModelBuilderType(Protocol):  # pragma: no cover
    """
    A function/callable that return a newly created model.
    This function is called once per run (actual and random)

    Args:
        is_random_run (bool): Indicate if this is a random run
        run_idx (int): The run index
    Returns:
        return (Any): The newly created model
    """

    def __call__(self, is_random_run: bool, run_idx: int) -> Any:
        ...


@runtime_checkable
class ModelFitterType(Protocol):  # pragma: no cover
    """
    A function/callable that fit a model. This function is called once per run (actual and random)

    Args:
        model (Any): The model to fit
        X (XType): The X data
        y (YType): The y data
    Returns:
        return (Any): The fitted model
    """

    def __call__(self, model: Any, X: XType, y: YType) -> Any:
        ...


@runtime_checkable
class ModelImportanceCalculatorType(Protocol):  # pragma: no cover
    """
    A function/callable computes the feature importances of a fitted model. This function is called once per run (actual and random)

    Args:
        model (Any): The fitted model
        X (XType): The X data
        y (YType): The y data
    Returns:
        return (pd.DataFrame): The return DataFrame with columns ["feature", "importance"]
    """

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
        return (pd.DataFrame): The return DataFrame with columns ["feature", "importance"]
    """

    def __call__(
        self,
        actual_importance_dfs: List[pd.DataFrame],
        random_importance_dfs: List[pd.DataFrame],
    ) -> pd.DataFrame:
        ...
