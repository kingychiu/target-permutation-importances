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

1. Fit the given model class $M$ times with shuffled feature order and different model's `random_state` to get $M$ actual feature importances of feature f: $A_f = [a_{f_1},a_{f_2}...a_{f_M}]$.
2. Fit the given model class with shuffled feature order and different model's `random_state` and **shuffled targets** for $N$ times to get $N$ feature random importances: $R_f = [r_{f_1},r_{f_2}...r_{f_N}]$.
3. Compute the final importances of a feature $f$ by various methods, such as:
    - $I_f = Avg(A_f) - Avg(R_f)$
    - $I_f = Avg(A_f) / (Avg(R_f) + 1)$

We want $M \ge 1$ and $N \gg 1$. Having $M=1$ means the actual importances depends on only 1 set of feature order and model's `random_state`.

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
or
```
poetry add target-permutation-importances
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
    model_cls_params={ # The parameters to pass to the model constructor.
        "n_estimators": 1,
    },
    model_fit_params={}, # The parameters to pass to the model fit method.
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

**Classification**

built_in: feature selection with models' built in importances method.

|dataset|n_samples|model|importances|feature_reduction|test_f1|improve_from_built_in|
|--|--|--|--|--|--|--|
|clf_cat/electricity|38474|RandomForestClassifier|built_in|8->2|0.894|-|
|clf_cat/electricity|38474|RandomForestClassifier|**A-R**|**8->4**|**0.9034**|**+1.0515%**|
|clf_cat/electricity|38474|RandomForestClassifier|A/(R+1)|8->2|0.894|0%|
|clf_cat/eye_movements|7608|RandomForestClassifier|built_in|23->22|0.6169|-|
|clf_cat/eye_movements|7608|RandomForestClassifier|**A-R**|**23->8**|**0.6499**|**+5.3544%**|
|clf_cat/eye_movements|7608|RandomForestClassifier|A/(R+1)|23->19|0.5927|-3.9164%|
|clf_cat/covertype|423680|RandomForestClassifier|built_in|54->26|0.9558|-|
|clf_cat/covertype|423680|RandomForestClassifier|**A-R**|**54->52**|**0.9593**|**+0.3634%**|
|clf_cat/covertype|423680|RandomForestClassifier|A/(R+1)|54->25|0.9561|+0.0311%|
|clf_cat/albert|58252|RandomForestClassifier|built_in|31->22|0.6518|-|
|clf_cat/albert|58252|RandomForestClassifier|A-R|31->31|0.6572|+0.828%|
|clf_cat/albert|58252|RandomForestClassifier|**A/(R+1)**|**31->27**|**0.6605**|**+1.3315%**|
|clf_cat/compas-two-years|4966|RandomForestClassifier|built_in|11->10|0.6316|-|
|clf_cat/compas-two-years|4966|RandomForestClassifier|A-R|11->11|0.6296|-0.3284%|
|clf_cat/compas-two-years|4966|RandomForestClassifier|**A/(R+1)**|**11->8**|**0.6346**|**+0.474%**|
|clf_cat/default-of-credit-card-clients|13272|RandomForestClassifier|built_in|21->18|0.671|-|
|clf_cat/default-of-credit-card-clients|13272|RandomForestClassifier|**A-R**|**21->17**|**0.6826**|**+1.7301%**|
|clf_cat/default-of-credit-card-clients|13272|RandomForestClassifier|A/(R+1)|21->19|0.6733|+0.3455%|
|clf_cat/road-safety|111762|RandomForestClassifier|built_in|32->31|0.7895|-|
|clf_cat/road-safety|111762|RandomForestClassifier|**A-R**|**32->32**|**0.791**|**+0.192%**|
|clf_cat/road-safety|111762|RandomForestClassifier|A/(R+1)|32->24|0.7889|-0.0786%|
|clf_num/Bioresponse|3434|RandomForestClassifier|built_in|419->295|0.7686|-|
|clf_num/Bioresponse|3434|RandomForestClassifier|**A-R**|**419->344**|**0.7703**|**+0.2265%**|
|clf_num/Bioresponse|3434|RandomForestClassifier|A/(R+1)|419->311|0.7686|+0.0008%|
|clf_num/jannis|57580|RandomForestClassifier|built_in|54->22|0.7958|-|
|clf_num/jannis|57580|RandomForestClassifier|A-R|54->26|0.7981|+0.286%|
|clf_num/jannis|57580|RandomForestClassifier|**A/(R+1)**|**54->27**|**0.799**|**+0.4019%**|
|clf_num/MiniBooNE|72998|RandomForestClassifier|built_in||||
|clf_num/MiniBooNE|72998|RandomForestClassifier|A-R||||
|clf_num/MiniBooNE|72998|RandomForestClassifier|A/(R+1)||||



**Regression**

built_in: feature selection with models' built in importances method.

|dataset|n_samples|model|importances|feature_reduction|test_rmse1|improve_from_built_in|
|--|--|--|--|--|--|--|
|reg_num/cpu_act|8192|RandomForestRegressor|built_in|21->20|6.0055|-|
|reg_num/cpu_act|8192|RandomForestRegressor|**A-R**|**21->21**|**5.9896**|**-2.2648%**|
|reg_num/cpu_act|8192|RandomForestRegressor|A/(R+1)|21->20|5.9999|-0.0932%|
|reg_num/pol|15000|RandomForestRegressor|**built_in**|**26->16**|**0.2734**|**-**|
|reg_num/pol|15000|RandomForestRegressor|A-R|26->26|0.2794|+2.1946%|
|reg_num/pol|15000|RandomForestRegressor|A/(R+1)|26->11|0.2827|+3.4016%|
|reg_num/elevators|16599|RandomForestRegressor|**built_in**|**16->7**|**8.0447**|**-**|
|reg_num/elevators|16599|RandomForestRegressor|A-R|16->14|8.3434|+3.713%|
|reg_num/elevators|16599|RandomForestRegressor|A/(R+1)|16->16|8.3232|+3.4619%|
|reg_num/wine_quality|6497|RandomForestRegressor|**built_in**|**11->11**|**0.4109**|**-**|
|reg_num/wine_quality|6497|RandomForestRegressor|A-R|11->11|0.4124|+0.3651%|
|reg_num/wine_quality|6497|RandomForestRegressor|A/(R+1)|11->9|0.421|+2.458%|
|reg_num/Ailerons|13750|RandomForestRegressor|built_in|33->12|2.8274|-|
|reg_num/Ailerons|13750|RandomForestRegressor|**A-R**|**33->25**|**2.7965**|**-1.0929%**|
|reg_num/Ailerons|13750|RandomForestRegressor|A/(R+1)|33->31|2.8438|+0.58%|
|reg_num/yprop_4_1|8885|RandomForestRegressor|built_in|42->26|75403.6496|-|
|reg_num/yprop_4_1|8885|RandomForestRegressor|A-R|42->42|74824.051|**-0.7687%**|
|reg_num/yprop_4_1|8885|RandomForestRegressor|**A/(R+1)**|**42->34**|**74466.2487**|**-1.2432%**|
|reg_num/superconduct|21263|RandomForestRegressor|built_in|79->53|54470.4924|-|
|reg_num/superconduct|21263|RandomForestRegressor|**A-R**|**79->74**|**54374.393**|**-0.1764%**|
|reg_num/superconduct|21263|RandomForestRegressor|A/(R+1)|79->75|55654.4327|+2.1735%|
|reg_cat/topo_2_1|8885|RandomForestRegressor|**built_in**|**255->217**|**76175.864**|**-**|
|reg_cat/topo_2_1|8885|RandomForestRegressor|A-R|255->255|76400.4006|+0.2948%|
|reg_cat/topo_2_1|8885|RandomForestRegressor|A/(R+1)|255->249|76432.8836|+0.3374%|


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

