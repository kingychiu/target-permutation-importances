# Target Permutation Importances

Keywords: Permutation Importances, Null Importances

References:
- [Permutation importance: a corrected feature importance measure](https://academic.oup.com/bioinformatics/article/26/10/1340/193348)
- [Feature Selection with Null Importances
](https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances/notebook)

## Overview
This method aims at lower the feature attribution due to the variance of a feature.
If a feature is important after the target vector is shuffled, it is fitting to noise.

By default, this package 

1. Fit the given model class on the given dataset M times to compute the mean actual feature importances ($A$).
2. Fit the given model class on the given dataset with shuffled targets N times to compute mean random feature importances ($R$).
3. Compute the final importances by $A / (R + 1)$


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

### With XGBoost

### With LightGBM

### With CatBoost


## Advance Usage

## Feature Selection Example

## Feature Selection Result Benchmark