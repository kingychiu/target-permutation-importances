# Target Permutation Importances

[![image](https://img.shields.io/pypi/v/target-permutation-importances.svg)](https://pypi.python.org/pypi/target-permutation-importances)
[![Downloads](https://static.pepy.tech/badge/target-permutation-importances)](https://pepy.tech/project/target-permutation-importances)
[![image](https://img.shields.io/pypi/pyversions/target-permutation-importances.svg)](https://pypi.python.org/pypi/target-permutation-importances)
[![Actions status](https://github.com/kingychiu/target-permutation-importances/workflows/CI/badge.svg)](https://github.com/kingychiu/target-permutation-importances/actions/workflows/main.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/kingychiu/target-permutation-importances)

[[Source]](https://github.com/kingychiu/target-permutation-importances/)
[[Bug Report]](https://github.com/kingychiu/target-permutation-importances/issues/)
[[Documentation]](https://target-permutation-importances.readthedocs.io/en/latest/)
[[API Reference]](https://target-permutation-importances.readthedocs.io/en/latest/reference/)

## Overview
This method aims to lower the feature attribution due to a feature's variance.
If a feature shows high importance to a model after the target vector is shuffled, it fits the noise.

Overall, this package 

1. Fit the given model class $M$ times with different model's `random_state` to get $M$ actual feature importances of feature f: $A_f = [a_{f_1},a_{f_2}...a_{f_M}]$.
2. Fit the given model class with different model's `random_state` and **shuffled targets** for $N$ times to get $N$ feature random importances: $R_f = [r_{f_1},r_{f_2}...r_{f_N}]$.
3. Compute the final importances of a feature $f$ by various methods, such as:
    - $I_f = Avg(A_f) - Avg(R_f)$
    - $I_f = Avg(A_f) / (Avg(R_f) + 1)$

We want $M \ge 1$ and $N \gg 1$. Having $M=1$ means the actual importances depends on only 1 model's `random_state` (Which is also fine).

Not to be confused with [sklearn.inspection.permutation_importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance),
this sklearn method is about feature permutation instead of target permutation.

This method were originally proposed/implemented by:
- [Permutation importance: a corrected feature importance measure](https://academic.oup.com/bioinformatics/article/26/10/1340/193348)
- [Feature Selection with Null Importances
](https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances/notebook)


## Install

```
pip install target-permutation-importances
```
or with poetry:
```
poetry add target-permutation-importances
```

Although this package is tested on models from `sklearn`, `xgboost`, `catboost`, `lightgbm`, they are not
a hard requirement for the installation, you can use this package for any model if it implements the `sklearn` interface.
For models that don't follow `sklearn` interface, you can use the exposed `generic_compute` method as discussed in the 
Advance Usage / Customization section.

Dependencies:
```
[tool.poetry.dependencies]
python = "^3.8"
numpy = "^1.21.0"
pandas = "^1.5.3"
tqdm = "^4.48.2"
beartype = "^0.14.1"
```


## Basic Usage

```python
# Import the function
import target_permutation_importances as tpi

# Prepare a dataset
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Models
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

data = load_breast_cancer()

# Convert to a pandas dataframe
Xpd = pd.DataFrame(data.data, columns=data.feature_names)

# Compute permutation importances with default settings
result_df = tpi.compute(
    model_cls=RandomForestClassifier, # The constructor/class of the model.
    model_cls_params={ # The parameters to pass to the model constructor. Update this based on your needs.
        "n_estimators": 1,
        "n_jobs": -1,
    },
    model_fit_params={}, # The parameters to pass to the model fit method. Update this based on your needs.
    X=Xpd, # pd.DataFrame, np.ndarray
    y=data.target, # pd.Series, np.ndarray
    num_actual_runs=2,
    num_random_runs=10,
)

print(result_df[["feature", "importance"]].sort_values("importance", ascending=False).head())
```
Fork above code from [Kaggle](https://www.kaggle.com/code/kingychiu/target-permutation-importances-basic-usage/notebook).

Outputs:
```
Running 2 actual runs and 10 random runs
100%|██████████| 2/2 [00:00<00:00, 167.35it/s]
100%|██████████| 10/10 [00:00<00:00, 163.71it/s]
                feature  importance
7   mean concave points    0.343365
8        mean concavity    0.291501
25      worst perimeter    0.021797
10       mean perimeter    0.021520
26         worst radius    0.008913
```

You can find more detailed examples in the "Feature Selection Examples" section.

## Advance Usage / Customization
This package exposes `generic_compute` to allow customization.
Read [`target_permutation_importances.__init__.py`](https://github.com/kingychiu/target-permutation-importances/blob/main/target_permutation_importances/__init__.py) for details.


## Feature Selection Examples
- [Feature Selection for Binary Classification](https://www.kaggle.com/code/kingychiu/feature-selection-for-binary-classification-task)

## Benchmarks

Benchmark has been done with some tabular datasets from the [Tabular data learning benchmark](https://github.com/LeoGrin/tabular-benchmark/tree/main). It is also
hosted on [Hugging Face](https://huggingface.co/datasets/inria-soda/tabular-benchmark).

For the binary classification task, `sklearn.metrics.f1_score` is used for evaluation. For the regression task, `sklearn.metrics.mean_squared_error` is used for evaluation.

The downloaded datasets are divided into 3 sections: `train`: 50%, `val`: 10%, `test`: 40%.
Feature importance is calculated from the `train` set. Feature selection is done on the `val` set. 
The final benchmark is evaluated on the `test` set. Therefore the `test` set is unseen to both the feature importance and selection process.


Raw result data are in [`benchmarks/results`](https://github.com/kingychiu/target-permutation-importances/tree/main/benchmarks/results).

<details>
    <summary>Binary Classification Results with RandomForest</summary>


built-in: The baseline, it is the built-in importances from the model.


| dataset                                         | importances  | feature_reduction | test_f1          |
|-------------------------------------------------|--------------|-------------------|---------------------|
| clf_cat/electricity.csv                         | built-in     | 8->2              | 0.894               |
| clf_cat/electricity.csv                         | **A-R**      | 8->4              | **0.9034**          |
| clf_cat/electricity.csv                         | A/(R+1)      | 8->2              | 0.894               |
| clf_cat/eye_movements.csv                       | built-in     | 23->22            | 0.6169              |
| clf_cat/eye_movements.csv                       | **A-R**      | 23->10            | **0.6772**          |
| clf_cat/eye_movements.csv                       | A/(R+1)      | 23->22            | 0.6212              |
| clf_cat/covertype.csv                           | built-in     | 54->26            | 0.9558              |
| clf_cat/covertype.csv                           | **A-R**      | 54->52            | **0.9586**          |
| clf_cat/covertype.csv                           | A/(R+1)      | 54->30            | 0.9547              |
| clf_cat/albert.csv                              | built-in     | 31->22            | 0.6518              |
| clf_cat/albert.csv                              | **A-R**      | 31->24            | **0.6587**          |
| clf_cat/albert.csv                              | A/(R+1)      | 31->22            | 0.6527              |
| clf_cat/compas-two-years.csv                    | built-in     | 11->10            | 0.6316              |
| clf_cat/compas-two-years.csv                    | **A-R**      | 11->2             | **0.6589**          |
| clf_cat/compas-two-years.csv                    | A/(R+1)      | 11->6             | 0.6335              |
| clf_cat/default-of-credit-card-clients.csv      | built-in     | 21->18            | 0.671               |
| clf_cat/default-of-credit-card-clients.csv      | **A-R**      | 21->17            | **0.6826**          |
| clf_cat/default-of-credit-card-clients.csv      | A/(R+1)      | 21->20            | 0.6797              |
| clf_cat/road-safety.csv                         | **built-in** | 32->31            | **0.7895**          |
| clf_cat/road-safety.csv                         | A-R          | 32->30            | 0.7886              |
| clf_cat/road-safety.csv                         | A/(R+1)      | 32->29            | 0.7893              |
| clf_num/Bioresponse.csv                         | built-in     | 419->295          | 0.7686              |
| clf_num/Bioresponse.csv                         | A-R          | 419->214          | 0.7692              |
| clf_num/Bioresponse.csv                         | **A/(R+1)**  | 419->403          | **0.775**           |
| clf_num/jannis.csv                              | built-in     | 54->22            | 0.7958              |
| clf_num/jannis.csv                              | A-R          | 54->28            | 0.7988              |
| clf_num/jannis.csv                              | **A/(R+1)**  | 54->26            | **0.7998**          |
| clf_num/MiniBooNE.csv                           | built-in     | 50->33            | 0.9306              |
| clf_num/MiniBooNE.csv                           | A-R          | 50->47            | 0.93                |
| clf_num/MiniBooNE.csv                           | **A/(R+1)**  | 50->49            | **0.9316**          |
</details>

<details>
    <summary>Regression Results with RandomForest</summary>

built-in: The baseline, it is the built-in importances from the model.

| dataset                                         | importances  | feature_reduction | test_mse          |
|-------------------------------------------------|--------------|-------------------|---------------------|
| reg_num/cpu_act.csv                             | built-in     | 21->20            | 6.0055              |
| reg_num/cpu_act.csv                             | A-R          | 21->20            | 6.0099              |
| reg_num/cpu_act.csv                             | **A/(R+1)**  | 21->19            | **5.9768**          |
| reg_num/pol.csv                                 | **built-in** | 26->16            | **0.2734**          |
| reg_num/pol.csv                                 | A-R          | 26->26            | 0.278               |
| reg_num/pol.csv                                 | A/(R+1)      | 26->12            | 0.2786              |
| reg_num/elevators.csv                           | built-in     | 16->7             | 8.0447              |
| reg_num/elevators.csv                           | A-R          | 16->15            | 8.3465              |
| reg_num/elevators.csv                           | **A/(R+1)**  | 16->6             | **7.8848**          |
| reg_num/wine_quality.csv                        | built-in     | 11->11            | 0.4109              |
| reg_num/wine_quality.csv                        | **A-R**      | 11->10            | **0.4089**          |
| reg_num/wine_quality.csv                        | A/(R+1)      | 11->11            | 0.4122              |
| reg_num/Ailerons.csv                            | built-in     | 33->12            | 2.8274              |
| reg_num/Ailerons.csv                            | **A-R**      | 33->29            | **2.8125**          |
| reg_num/Ailerons.csv                            | A/(R+1)      | 33->12            | 2.8304              |
| reg_num/yprop_4_1.csv                           | built-in     | 42->26            | 75403.6496          |
| reg_num/yprop_4_1.csv                           | A-R          | 42->41            | 75081.8961          |
| reg_num/yprop_4_1.csv                           | **A/(R+1)**  | 42->32            | **74671.0854**      |
| reg_num/superconduct.csv                        | built-in     | 79->53            | 54470.4924          |
| reg_num/superconduct.csv                        | **A-R**      | 79->63            | **54011.8479**      |
| reg_num/superconduct.csv                        | A/(R+1)      | 79->60            | 54454.3817          |
| reg_cat/topo_2_1.csv                            | built-in     | 255->217          | 76175.864           |
| reg_cat/topo_2_1.csv                            | A-R          | 255->254          | 76206.9714          |
| reg_cat/topo_2_1.csv                            | **A/(R+1)**  | 255->226          | **76140.8313**      |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | **built-in** | 359->6            | **177937.9184**     |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | A-R          | 359->194          | 183405.9763         |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | **A/(R+1)**  | 359->6            | **177937.9184**     |
| reg_cat/house_sales.csv                         | **built-in** | 17->16            | **110072.8755**     |
| reg_cat/house_sales.csv                         | A-R          | 17->17            | 110141.2913         |
| reg_cat/house_sales.csv                         | A/(R+1)      | 17->17            | 110404.0862         |
| reg_cat/nyc-taxi-green-dec-2016.csv             | **built-in** | 16->15            | **10585.6377**      |
| reg_cat/nyc-taxi-green-dec-2016.csv             | A-R          | 16->4             | 10758.4811          |
| reg_cat/nyc-taxi-green-dec-2016.csv             | A/(R+1)      | 16->15            | 10589.5054          |
| reg_cat/Allstate_Claims_Severity.csv            | **built-in** | 124->113          | **1002055785.0415** |
| reg_cat/Allstate_Claims_Severity.csv            | A-R          | 124->124          | 1003019739.9178     |
| reg_cat/Allstate_Claims_Severity.csv            | A/(R+1)      | 124->102          | 1003113924.3013     |


</details>


## Kaggle Competitions
Many Kaggle Competition top solutions involve this method, here are some examples

| Year | Competition                                                                                                                  | Medal | Link                                                                                                                                        |
| ---- | ---------------------------------------------------------------------------------------------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023 | [Predict Student Performance from Game Play](https://www.kaggle.com/competitions/predict-student-performance-from-game-play) | Gold  | [3rd place solution](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/420235)                      |
| 2019 | [Elo Merchant Category Recommendation](https://www.kaggle.com/competitions/elo-merchant-category-recommendation/overview)    | Gold  | [16th place solution]([-play/discussion/420235](https://www.kaggle.com/competitions/elo-merchant-category-recommendation/discussion/82166)) |
| 2018 | [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview)                            | Gold  | [10th place solution](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64598)                                        |


## Development Setup and Contribution Guide
**Python Version**

You can find the suggested development Python version in `.python-version`.
You might consider setting up `Pyenv` if you want to have multiple Python versions on your machine.

**Python packages**

This repository is setup with `Poetry`. If you are not familiar with Poetry, you can find package requirements listed in `pyproject.toml`. 
Otherwise, you can just set it up with `poetry install`

**Run Benchmarks**

To run the benchmark locally on your machine, run `make run_tabular_benchmark` or `python -m benchmarks.run_tabular_benchmark`

**Make Changes**

Following the [Make Changes Guide from Github](https://github.com/github/docs/blob/main/CONTRIBUTING.md#make-changes)
Before committing or merging, please run the linters defined in `make lint` and the tests defined in `make test`

