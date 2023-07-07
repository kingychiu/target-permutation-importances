import cudf

# import pytest
from cuml.cluster import DBSCAN

from target_permutation_importances import (  # compute,
    compute_permutation_importance_by_division,
    compute_permutation_importance_by_subtraction,
)

# from cuml.datasets.classification import make_classification
# from cuml.ensemble import RandomForestClassifier as cuRF


IMP_FUNCS = [
    compute_permutation_importance_by_subtraction,
    compute_permutation_importance_by_division,
]

test_compute_clf_scope = []
for imp_func in IMP_FUNCS:
    test_compute_clf_scope.append(imp_func)


def test_install():
    # Create and populate a GPU DataFrame
    gdf_float = cudf.DataFrame()
    gdf_float["0"] = [1.0, 2.0, 5.0]
    gdf_float["1"] = [4.0, 2.0, 1.0]
    gdf_float["2"] = [4.0, 2.0, 1.0]

    # Setup and fit clusters
    dbscan_float = DBSCAN(eps=1.0, min_samples=1)
    dbscan_float.fit(gdf_float)

    assert dbscan_float.labels_.any()


# @pytest.mark.parametrize("imp_func", test_compute_clf_scope)
# def test_compute_binary_classification(imp_func):
#     X, y = make_classification(
#         n_classes=2, n_features=20, n_samples=100, random_state=0
#     )
#     print(type(X))
#     print(type(y))
#     result_df = compute(
#         model_cls=cuRF,
#         model_cls_params={"n_estimators": 2},
#         model_fit_params={},
#         permutation_importance_calculator=imp_func,
#         X=X,
#         y=y,
#         num_actual_runs=5,
#         num_random_runs=5,
#     )
