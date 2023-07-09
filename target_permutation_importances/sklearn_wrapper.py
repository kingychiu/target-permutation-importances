"""
The Sklearn Class Wrappers
"""
import numpy as np
import pandas as pd
from beartype.typing import Any, Dict, List, Union

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


class TargetPermutationImportancesWrapper:
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
            # Import the package
            import target_permutation_importances as tpi

            # Prepare a dataset
            import pandas as pd
            import numpy as np
            from sklearn.datasets import load_breast_cancer

            # Models
            from sklearn.feature_selection import SelectFromModel
            from sklearn.ensemble import RandomForestClassifier

            data = load_breast_cancer()

            # Convert to a pandas dataframe
            Xpd = pd.DataFrame(data.data, columns=data.feature_names)

            # Compute permutation importances with default settings
            wrapped_model = tpi.TargetPermutationImportancesWrapper(
                model_cls=RandomForestClassifier, # The constructor/class of the model.
                model_cls_params={ # The parameters to pass to the model constructor. Update this based on your needs.
                    "n_jobs": -1,
                },
                num_actual_runs=2,
                num_random_runs=10,
                # Options: {compute_permutation_importance_by_subtraction, compute_permutation_importance_by_division}
                # Or use your own function to calculate.
                permutation_importance_calculator=tpi.compute_permutation_importance_by_subtraction,
            )
            wrapped_model.fit(
                X=Xpd, # pd.DataFrame, np.ndarray
                y=data.target, # pd.Series, np.ndarray
                # And other fit parameters for the model.
            )
            # Get the feature importances as a pandas dataframe
            result_df = wrapped_model.feature_importances_df
            print(result_df[["feature", "importance"]].sort_values("importance", ascending=False).head())


            # Select top-5 features with sklearn `SelectFromModel`
            selector = SelectFromModel(
                estimator=wrapped_model, prefit=True, max_features=5, threshold=-np.inf
            ).fit(Xpd, data.target)
            selected_x = selector.transform(Xpd)
            print(selected_x.shape)
            print(selector.get_feature_names_out())
            ```
        """
        self.model_cls = model_cls
        self.model_cls_params = model_cls_params
        self.model = self.model_cls(**self.model_cls_params)
        self.num_actual_runs = num_actual_runs
        self.num_random_runs = num_random_runs
        self.shuffle_feature_order = shuffle_feature_order
        self.permutation_importance_calculator = permutation_importance_calculator

    def _process_feature_importances_df(
        self,
        feature_importances_df: pd.DataFrame,
        feature_order: Union[List, np.ndarray],
    ):
        feature_importances_df = (
            feature_importances_df.set_index("feature")
            .loc[feature_order]
            .reset_index()
            .rename(columns={"index": "feature"})
        )

        # Make sure the importance is positive
        feature_importances_df["raw_importance"] = feature_importances_df["importance"]
        if feature_importances_df["importance"].min() < 0:
            feature_importances_df["importance"] -= feature_importances_df[
                "importance"
            ].min()
        assert feature_importances_df["importance"].min() >= 0

        self.feature_importances_df = feature_importances_df
        self.n_features_in_ = feature_importances_df.shape[0]
        self.feature_names_in_ = feature_importances_df["feature"].to_numpy()
        self.feature_importances_ = feature_importances_df["importance"].to_numpy()

    def fit(
        self,
        X: XType,
        y: YType,
        **fit_params,
    ) -> "TargetPermutationImportancesWrapper":
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
            self._process_feature_importances_df(result, X.columns.to_list())
        else:
            self._process_feature_importances_df(result, list(range(X.shape[1])))
        return self
