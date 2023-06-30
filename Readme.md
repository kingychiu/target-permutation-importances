# Target Permutation Importances

Keywords: Permutation Importances, Null Importances

References:
[Feature Selection with Null Importances
](https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances/notebook)


## Basic Usage

### With Scikit Learn Models

```python
# Import the function
from target_permutation_importances import compute

# Prepare a dataset
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data = load_breast_cancer()

# Compute permutation importances with default settings
result_df = compute(
    model_cls=RandomForestClassifier,
    model_cls_params={ # The params for the model class construction
        "n_estimators": 1,
    },
    model_fit_params={}, # The params for model.fit
    X=Xpd,
    y=data.target,
    num_actual_runs=2,
    num_random_runs=10,
)
```