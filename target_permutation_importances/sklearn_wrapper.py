"""
The Sklearn Class Wrappers
"""
import numpy as np
import pandas as pd
from beartype.typing import Any, Dict

from target_permutation_importances.functional import (
    compute,
    compute_permutation_importance_by_subtraction,
)
from target_permutation_importances.typing import (
    PermutationImportanceCalculatorType,
    PositiveInt,
    XType,
    YType,
)


class TargetPermutationImportances:
    def __init__(
        self,
        model_cls: Any,
        model_cls_params: Dict,
        num_actual_runs: PositiveInt = 2,
        num_random_runs: PositiveInt = 10,
        shuffle_feature_order: bool = False,
        permutation_importance_calculator: PermutationImportanceCalculatorType = compute_permutation_importance_by_subtraction,
    ):
        """
        Compute the permutation importance of a model given a dataset.

        Args:
            model_cls: The constructor/class of the model.
            model_cls_params: The parameters to pass to the model constructor.
            model_fit_params: The parameters to pass to the model fit method.
            num_actual_runs: Number of actual runs. Defaults to 2.
            num_random_runs: Number of random runs. Defaults to 10.
            shuffle_feature_order: Whether to shuffle the feature order for each run (only for X being pd.DataFrame). Defaults to False.
            permutation_importance_calculator: The function to compute the final importance. Defaults to compute_permutation_importance_by_subtraction.

        Example:
            ```python
            # Import the function
            import target_permutation_importances as tpi

            # Prepare a dataset
            import pandas as pd
            from sklearn.datasets import load_breast_cancer

            # Models
            from sklearn.feature_selection import SelectFromModel
            from sklearn.ensemble import RandomForestClassifier

            data = load_breast_cancer()

            # Convert to a pandas dataframe
            Xpd = pd.DataFrame(data.data, columns=data.feature_names)

            # Compute permutation importances with default settings
            ranker = tpi.TargetPermutationImportances(
                model_cls=RandomForestClassifier, # The constructor/class of the model.
                model_cls_params={ # The parameters to pass to the model constructor. Update this based on your needs.
                    "n_jobs": -1,
                },
                num_actual_runs=2,
                num_random_runs=10,
                shuffle_feature_order=False,
                # Options: {compute_permutation_importance_by_subtraction, compute_permutation_importance_by_division}
                # Or use your own function to calculate.
                permutation_importance_calculator=tpi.compute_permutation_importance_by_subtraction,
            )
            ranker.fit(
                X=Xpd, # pd.DataFrame, np.ndarray
                y=data.target, # pd.Series, np.ndarray
                # And other fit parameters for the model.
                n_jobs=-1,
            )
            # Get the feature importances as a pandas dataframe
            result_df = ranker.feature_importances_df_
            print(result_df[["feature", "importance"]].sort_values("importance", ascending=False).head())


            # Select features with sklearn feature selectors
            selector = SelectFromModel(
                estimator=ranker, prefit=True, threshold=result_df["importance"].max()
            ).fit(Xpd, data.target)
            selected_x = selector.transform(X)
            print(selected_x.shape)
            ```
        """
        self.model_cls = model_cls
        self.model_cls_params = model_cls_params
        self.num_actual_runs = num_actual_runs
        self.num_random_runs = num_random_runs
        self.shuffle_feature_order = shuffle_feature_order
        self.permutation_importance_calculator = permutation_importance_calculator

        self.n_features_in_ = 0
        self.feature_names_in_ = np.zeros(0)
        self.feature_importances_ = np.zeros(0)
        self.feature_importances_df_ = pd.DataFrame()

    def fit(
        self,
        X: XType,
        y: YType,
        **fit_params,
    ) -> "TargetPermutationImportances":
        """
        Compute the permutation importance of a model given a dataset.

        Args:
            X: The input data.
            y: The target vector.
            fit_params: The parameters to pass to the model fit method.
        """
        result = compute(
            model_cls=self.model_cls,
            model_cls_params=self.model_cls_params,
            model_fit_params=fit_params,
            X=X,
            y=y,
            num_actual_runs=self.num_actual_runs,
            num_random_runs=self.num_random_runs,
            shuffle_feature_order=self.shuffle_feature_order,
            permutation_importance_calculator=self.permutation_importance_calculator,
        )
        if isinstance(result, list):  # pragma: no cover
            result = result[0]

        if isinstance(X, pd.DataFrame):
            # Sort back feature order of the importance dataframe
            result = (
                result.set_index("feature")
                .loc[X.columns]
                .reset_index()
                .rename(columns={"index": "feature"})
            )

        self.feature_importances_df_ = result
        self.n_features_in_ = result.shape[0]
        self.feature_names_in_ = result["feature"].to_numpy()
        self.feature_importances_ = result["importance"].to_numpy()

        return self
