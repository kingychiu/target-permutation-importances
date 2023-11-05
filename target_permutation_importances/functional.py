"""
The core APIs of this library.
"""

import gc
from functools import partial

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Union
from scipy.stats import wasserstein_distance  # type: ignore
from tqdm import tqdm

from target_permutation_importances.typing import (
    ModelBuilderType,
    ModelFitterType,
    ModelImportanceGetter,
    PermutationImportanceCalculatorType,
    PositiveInt,
    XBuilderType,
    XType,
    YBuilderType,
    YRandomizationType,
    YType,
)


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


def compute_permutation_importance_by_wasserstein_distance(
    actual_importance_dfs: List[pd.DataFrame], random_importance_dfs: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Given a list of actual importance DataFrames and a list of random importance compute
    the permutation importance by $I_f = wasserstein_distance(A_f, R_f)$

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

    # Calculate the wasserstein_distance
    distances = {}
    for f in random_importance_df["feature"].unique():
        distances[f] = wasserstein_distance(
            actual_importance_df[actual_importance_df["feature"] == f][
                "importance"
            ].to_numpy(),
            random_importance_df[random_importance_df["feature"] == f][
                "importance"
            ].to_numpy(),
        )
    mean_actual_importance_df[
        "wasserstein_distance"
    ] = mean_actual_importance_df.index.map(distances)

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
    mean_actual_importance_df["std_random_importance"] = std_random_importance_df[
        "importance"
    ]
    mean_actual_importance_df["importance"] = mean_actual_importance_df[
        "wasserstein_distance"
    ]
    return mean_actual_importance_df[
        [
            "importance",
            "mean_actual_importance",
            "mean_random_importance",
            "std_actual_importance",
            "std_random_importance",
            "wasserstein_distance",
        ]
    ].reset_index()


def _compute_one_run(
    model_builder: ModelBuilderType,
    model_fitter: ModelFitterType,
    importance_getter: ModelImportanceGetter,
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
    return importance_getter(model, X, y)


@beartype
def generic_compute(
    model_builder: ModelBuilderType,
    model_fitter: ModelFitterType,
    importance_getter: ModelImportanceGetter,
    permutation_importance_calculator: Union[
        PermutationImportanceCalculatorType, List[PermutationImportanceCalculatorType]
    ],
    X_builder: XBuilderType,
    y_builder: YBuilderType,
    num_actual_runs: PositiveInt = 2,
    num_random_runs: PositiveInt = 10,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    The generic compute function allows customization of the computation. It is used by the `compute` function.

    Args:
        model_builder (ModelBuilderType): A function that return a model.
        model_fitter (ModelFitterType): A function that fit a model.
        importance_getter (ModelImportanceGetter): A function that compute the importance of a model.
        permutation_importance_calculator (Union[ PermutationImportanceCalculatorType, List[PermutationImportanceCalculatorType] ]):
            A function or list of functions that compute the final permutation importance.
        X_builder (XBuilderType): A function that return the X data.
        y_builder (YBuilderType): A function that return the y data.
        num_actual_runs (PositiveInt, optional): Number of actual runs. Defaults to 2.
        num_random_runs (PositiveInt, optional): Number of random runs. Defaults to 10.

    Returns:
        The return DataFrame(s) contain columns ["feature", "importance"]
    """
    run_params = {
        "model_builder": model_builder,
        "model_fitter": model_fitter,
        "importance_getter": importance_getter,
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
    if isinstance(permutation_importance_calculator, list):
        return [
            calc(actual_importance_dfs, random_importance_dfs)
            for calc in permutation_importance_calculator
        ]
    return permutation_importance_calculator(
        actual_importance_dfs, random_importance_dfs
    )


def _get_feature_names_attr(model: Any):
    feature_attr = "feature_names_in_"
    if "LGBM" in str(model.__class__):
        feature_attr = "feature_name_"
    elif "Cat" in str(model.__class__):
        feature_attr = "feature_names_"
    return feature_attr


def _get_model_importances_attr(model: Any):
    if hasattr(model, "feature_importances_"):
        return "feature_importances_"
    if hasattr(model, "coef_"):
        return "coef_"
    raise NotImplementedError(  # pragma: no cover
        "Model does not have feature importances method"
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
    shuffle_feature_order: bool = False,
    permutation_importance_calculator: Union[
        PermutationImportanceCalculatorType, List[PermutationImportanceCalculatorType]
    ] = compute_permutation_importance_by_subtraction,
    y_randomizations: Optional[List[YRandomizationType]] = None,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Compute the permutation importance of a model given a dataset.

    Args:
        model_cls: The constructor/class of the model.
        model_cls_params: The parameters to pass to the model constructor.
        model_fit_params: The parameters to pass to the model fit method.
        X: The input data.
        y: The target vector.
        num_actual_runs: Number of actual runs. Defaults to 2.
        num_random_runs: Number of random runs. Defaults to 10.
        shuffle_feature_order: Whether to shuffle the feature order for each run (only for X being pd.DataFrame). Defaults to False.
        permutation_importance_calculator: The function to compute the final importance. Defaults to compute_permutation_importance_by_subtraction.
        y_randomizations: The randomization methods to use for the random runs. Defaults to [YRandomizationType.SHUFFLE_TARGET].
    Returns:
        The return DataFrame(s) contain columns ["feature", "importance"]

    Example:
        ```python
        # import the package
        import target_permutation_importances as tpi

        # Prepare a dataset
        import pandas as pd
        from sklearn.datasets import load_breast_cancer

        # Models
        from sklearn.ensemble import RandomForestClassifier

        data = load_breast_cancer()

        # Convert to a pandas dataframe
        Xpd = pd.DataFrame(data.data, columns=data.feature_names)

        # Compute permutation importances with default settings
        result_df = tpi.compute(
            model_cls=RandomForestClassifier, # The constructor/class of the model.
            model_cls_params={ # The parameters to pass to the model constructor. Update this based on your needs.
                "n_jobs": -1,
            },
            model_fit_params={}, # The parameters to pass to the model fit method. Update this based on your needs.
            X=Xpd, # pd.DataFrame, np.ndarray
            y=data.target, # pd.Series, np.ndarray
            num_actual_runs=2,
            num_random_runs=10,
            # Options: {compute_permutation_importance_by_subtraction, compute_permutation_importance_by_division}
            # Or use your own function to calculate.
            permutation_importance_calculator=tpi.compute_permutation_importance_by_subtraction,
        )

        print(result_df[["feature", "importance"]].sort_values("importance", ascending=False).head())
        ```
    """
    if y_randomizations is None:
        y_randomizations = [YRandomizationType.SHUFFLE_TARGET]

    def _x_builder(is_random_run: bool, run_idx: int) -> XType:
        if shuffle_feature_order:
            if isinstance(X, pd.DataFrame):
                # Shuffle the columns
                rng = np.random.default_rng(seed=run_idx)
                shuffled_columns = rng.permutation(X.columns)
                return X[shuffled_columns]
            raise NotImplementedError(  # pragma: no cover
                "Only support pd.DataFrame when shuffle_feature_order=True"
            )
        return X

    def _y_builder(is_random_run: bool, run_idx: int) -> YType:
        rng = np.random.default_rng(seed=run_idx)
        if is_random_run:
            random_method_idx = run_idx % len(y_randomizations)
            y_randomization = y_randomizations[random_method_idx]
            if y_randomization == YRandomizationType.SHUFFLE_TARGET:
                # Only shuffle the target for random runs
                return rng.permutation(y)
            if y_randomization == YRandomizationType.RANDOM_NORMAL:
                # Only shuffle the target for random runs
                return rng.normal(y.mean(), y.std(), y.shape)
            if y_randomization == YRandomizationType.RANDOM_UNIFORM:
                # Only shuffle the target for random runs
                return rng.uniform(y.min(), y.max(), y.shape)

            raise NotImplementedError(  # pragma: no cover
                f"Selected y_randomization_type {y_randomization} is not supported"
            )
        return y

    def _model_builder(is_random_run: bool, run_idx: int) -> Any:
        # Model random state should be different for each run for both
        # actual and random runs
        _model_cls_params = model_cls_params.copy()
        if "MultiOutput" not in model_cls.__name__:
            _model_cls_params["random_state"] = run_idx
        else:
            _model_cls_params["estimator"].random_state = run_idx

        return model_cls(**_model_cls_params)

    def _model_fitter(model: Any, X: XType, y: YType) -> Any:
        _model_fit_params = model_fit_params.copy()
        if "Cat" in str(model.__class__):
            _model_fit_params["verbose"] = False
        return model.fit(X, y, **_model_fit_params)

    def _importance_getter(model: Any, X: XType, y: YType) -> pd.DataFrame:
        feature_names_attr = _get_feature_names_attr(model)
        is_pd = isinstance(X, pd.DataFrame)

        if "MultiOutput" not in str(model.__class__):
            if is_pd:
                features = getattr(model, feature_names_attr)
            else:
                features = list(range(0, X.shape[1]))

            model_importances_attr = _get_model_importances_attr(model)
            importances = np.abs(getattr(model, model_importances_attr))
            if len(importances.shape) > 1:
                importances = importances.mean(axis=0)
            return pd.DataFrame(
                {
                    "feature": features,
                    "importance": importances,
                }
            )

        features = []
        feature_importances = np.zeros(X.shape[1])
        for est in model.estimators_:
            if is_pd:
                feature_names_attr = _get_feature_names_attr(est)
                features = getattr(est, feature_names_attr)
            else:
                features = list(range(0, X.shape[1]))

            model_importances_attr = _get_model_importances_attr(est)
            importances = np.abs(getattr(est, model_importances_attr))
            if len(importances.shape) > 1:  # pragma: no cover
                importances = importances.mean(axis=0)
            feature_importances += importances
        return pd.DataFrame(
            {
                "feature": features,
                "importance": feature_importances / len(model.estimators_),
            }
        )

    return generic_compute(
        model_builder=_model_builder,
        model_fitter=_model_fitter,
        importance_getter=_importance_getter,
        permutation_importance_calculator=permutation_importance_calculator,
        X_builder=_x_builder,
        y_builder=_y_builder,
        num_actual_runs=num_actual_runs,
        num_random_runs=num_random_runs,
    )
