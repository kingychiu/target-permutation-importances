from target_permutation_importances.functional import (
    compute,
    compute_permutation_importance_by_division,
    compute_permutation_importance_by_subtraction,
    compute_permutation_importance_by_wasserstein_distance,
    generic_compute,
)
from target_permutation_importances.sklearn_wrapper import (  # type: ignore
    TargetPermutationImportancesWrapper,
)
