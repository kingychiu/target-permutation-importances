# Benchmarks

Benchmark has been done with some tabular datasets from the [Tabular data learning benchmark](https://github.com/LeoGrin/tabular-benchmark/tree/main). It is also
hosted on [Hugging Face](https://huggingface.co/datasets/inria-soda/tabular-benchmark).

For the binary classification task, `sklearn.metrics.f1_score` is used for evaluation. For the regression task, `sklearn.metrics.mean_squared_error` is used for evaluation.

The downloaded datasets are divided into 3 sections: `train`: 50%, `val`: 10%, `test`: 40%.
Feature importance is calculated from the `train` set. Feature selection is done on the `val` set. 
The final benchmark is evaluated on the `test` set. Therefore the `test` set is unseen to both the feature importance and selection process.

The exact model params are the following as show in [run_tabular_benchmark.py](https://github.com/kingychiu/target-permutation-importances/blob/main/benchmarks/run_tabular_benchmark.py) as well.

```python
seed = 2023
num_actual_runs = 10
num_random_runs = 50
shuffle_feature_order = True

model_cls_dicts = {
    "binary_classification": {
        "RandomForest": (RandomForestClassifier, {"n_jobs": -1}),
        "XGBoost": (
            XGBClassifier,
            {"n_jobs": -1, "importance_type": "gain"},
        ),
        "LGBM": (
            LGBMClassifier,
            {"n_jobs": -1, "importance_type": "gain"},
        ),
        "CatBoost": (
            CatBoostClassifier,
            {"verbose": False},
        ),
    },
    "regression": {
        "RandomForest": (RandomForestRegressor, {"n_jobs": -1}),
        "XGBoost": (
            XGBRegressor,
            {"n_jobs": -1, "importance_type": "gain"},
        ),
        "LGBM": (
            LGBMRegressor,
            {"n_jobs": -1, "importance_type": "gain"},
        ),
        "CatBoost": (
            CatBoostRegressor,
            {"verbose": False},
        ),
    },
}
```

Here is the summary of running null-importances with feature selection on multiple models and datasets. "better" means it is better than running feature selection with the model's built-in feature importances. We can see even with **with default models' parameters** it shows its effectiveness.

| model                  | n_dataset | n_better | better % |
|------------------------|-----------|----------|----------|
| RandomForestClassifier | 10        | 10       | 100.0    |
| RandomForestRegressor  | 12        | 8        | 66.67    |
| XGBClassifier          | 10        | 7        | 70.0     |
| XGBRegressor           | 12        | 7        | 58.33    |
| LGBMClassifier         | 10        | 8        | 80.0     |
| LGBMRegressor          | 12        | 8        | 66.67    |
| CatBoostClassifier     | 10        | 6        | 60.0     |
| CatBoostRegressor      | 12        | 8        | 66.67    |


The table below also shows the effectiveness of different importances calculation methods for each task over all datasets:

| task                  | n_dataset | importances | the best % |
| --------------------- | --------- | ----------- | ---------- |
| binary_classification | 10        | A-R         | 32.69      |
| binary_classification | 10        | A/(R+1)     | 26.92      |
| binary_classification | 10        | Wasserstein | 23.08      |
| binary_classification | 10        | built-in    | 17.31      |
| regression            | 12        | built-in    | 29.82      |
| regression            | 12        | A/(R+1)     | 28.07      |
| regression            | 12        | A-R         | 26.32      |
| regression            | 12        | Wasserstein | 15.79      |



built-in: The baseline, it is the built-in importances from the model.

A-R: `compute_permutation_importance_by_subtraction`

A/(R+1): `compute_permutation_importance_by_division`

Wasserstein: `compute_permutation_importance_by_wasserstein_distance`

Below tables shows the raw results and the raw csv data are in [`benchmarks/results`](https://github.com/kingychiu/target-permutation-importances/tree/main/benchmarks/results).

## Binary Classification Results with RandomForest
| dataset                                    | importances     | feature_reduction | test_score   |
|--------------------------------------------|-----------------|-------------------|--------------|
| clf_cat/electricity.csv                    | built-in        | 8->2              | 0.894008     |
| clf_cat/electricity.csv                    | **A-R**         | **8->4**          | **0.903448** |
| clf_cat/electricity.csv                    | A/(R+1)         | 8->2              | 0.894008     |
| clf_cat/electricity.csv                    | Wasserstein     | 8->8              | 0.886164     |
| clf_cat/eye_movements.csv                  | built-in        | 23->22            | 0.616902     |
| clf_cat/eye_movements.csv                  | **A-R**         | **23->11**        | **0.663573** |
| clf_cat/eye_movements.csv                  | A/(R+1)         | 23->23            | 0.613378     |
| clf_cat/eye_movements.csv                  | Wasserstein     | 23->22            | 0.616098     |
| clf_cat/covertype.csv                      | built-in        | 54->26            | 0.955826     |
| clf_cat/covertype.csv                      | **A-R**         | **54->52**        | **0.958779** |
| clf_cat/covertype.csv                      | A/(R+1)         | 54->25            | 0.956147     |
| clf_cat/covertype.csv                      | Wasserstein     | 54->28            | 0.954931     |
| clf_cat/albert.csv                         | built-in        | 31->22            | 0.65181      |
| clf_cat/albert.csv                         | A-R             | 31->22            | 0.651038     |
| clf_cat/albert.csv                         | A/(R+1)         | 31->28            | 0.656057     |
| clf_cat/albert.csv                         | **Wasserstein** | **31->28**        | **0.656968** |
| clf_cat/compas-two-years.csv               | built-in        | 11->10            | 0.631631     |
| clf_cat/compas-two-years.csv               | **A-R**         | **11->2**         | **0.658924** |
| clf_cat/compas-two-years.csv               | A/(R+1)         | 11->8             | 0.63761      |
| clf_cat/compas-two-years.csv               | Wasserstein     | 11->8             | 0.630445     |
| clf_cat/default-of-credit-card-clients.csv | built-in        | 21->18            | 0.670973     |
| clf_cat/default-of-credit-card-clients.csv | **A-R**         | **21->17**        | **0.682581** |
| clf_cat/default-of-credit-card-clients.csv | A/(R+1)         | 21->21            | 0.676039     |
| clf_cat/default-of-credit-card-clients.csv | Wasserstein     | 21->21            | 0.676729     |
| clf_cat/road-safety.csv                    | built-in        | 32->31            | 0.789492     |
| clf_cat/road-safety.csv                    | A-R             | 32->30            | 0.788183     |
| clf_cat/road-safety.csv                    | A/(R+1)         | 32->32            | 0.790758     |
| clf_cat/road-safety.csv                    | **Wasserstein** | **32->32**        | **0.790862** |
| clf_num/Bioresponse.csv                    | built-in        | 419->295          | 0.768577     |
| clf_num/Bioresponse.csv                    | A-R             | 419->80           | 0.764124     |
| clf_num/Bioresponse.csv                    | A/(R+1)         | 419->300          | 0.767705     |
| clf_num/Bioresponse.csv                    | **Wasserstein** | **419->60**       | **0.769556** |
| clf_num/jannis.csv                         | built-in        | 54->22            | 0.795777     |
| clf_num/jannis.csv                         | **A-R**         | **54->28**        | **0.798353** |
| clf_num/jannis.csv                         | A/(R+1)         | 54->27            | 0.797567     |
| clf_num/jannis.csv                         | Wasserstein     | 54->51            | 0.78659      |
| clf_num/MiniBooNE.csv                      | built-in        | 50->33            | 0.930573     |
| clf_num/MiniBooNE.csv                      | A-R             | 50->42            | 0.930107     |
| clf_num/MiniBooNE.csv                      | A/(R+1)         | 50->45            | 0.930059     |
| clf_num/MiniBooNE.csv                      | **Wasserstein** | **50->50**        | **0.931065** |

---

## Regression Results with RandomForest
| dataset                                         | importances     | feature_reduction | test_score            |
|-------------------------------------------------|-----------------|-------------------|-----------------------|
| reg_num/cpu_act.csv                             | built-in        | 21->20            | 6.005464              |
| reg_num/cpu_act.csv                             | A-R             | 21->20            | 6.009862              |
| reg_num/cpu_act.csv                             | **A/(R+1)**     | **21->19**        | **5.976787**          |
| reg_num/cpu_act.csv                             | Wasserstein     | 21->18            | 6.044395              |
| reg_num/pol.csv                                 | built-in        | 26->16            | 0.273401              |
| reg_num/pol.csv                                 | A-R             | 26->26            | 0.277991              |
| reg_num/pol.csv                                 | A/(R+1)         | 26->12            | 0.278584              |
| reg_num/pol.csv                                 | **Wasserstein** | **26->14**        | **0.271883**          |
| reg_num/elevators.csv                           | built-in        | 16->7             | 8.044735              |
| reg_num/elevators.csv                           | A-R             | 16->15            | 8.34646               |
| reg_num/elevators.csv                           | **A/(R+1)**     | **16->6**         | **7.884783**          |
| reg_num/elevators.csv                           | Wasserstein     | 16->14            | 8.37675               |
| reg_num/wine_quality.csv                        | built-in        | 11->11            | 0.410926              |
| reg_num/wine_quality.csv                        | **A-R**         | **11->10**        | **0.40895**           |
| reg_num/wine_quality.csv                        | A/(R+1)         | 11->11            | 0.411228              |
| reg_num/wine_quality.csv                        | Wasserstein     | 11->11            | 0.41206               |
| reg_num/Ailerons.csv                            | built-in        | 33->12            | 2.827377              |
| reg_num/Ailerons.csv                            | **A-R**         | **33->29**        | **2.810824**          |
| reg_num/Ailerons.csv                            | A/(R+1)         | 33->12            | 2.821326              |
| reg_num/Ailerons.csv                            | Wasserstein     | 33->32            | 2.841721              |
| reg_num/yprop_4_1.csv                           | built-in        | 42->26            | 75403.649647          |
| reg_num/yprop_4_1.csv                           | **A-R**         | **42->41**        | **74837.199659**      |
| reg_num/yprop_4_1.csv                           | A/(R+1)         | 42->30            | 75175.03063           |
| reg_num/yprop_4_1.csv                           | Wasserstein     | 42->29            | 75796.896781          |
| reg_num/superconduct.csv                        | built-in        | 79->53            | 54470.492359          |
| reg_num/superconduct.csv                        | A-R             | 79->68            | 54068.62626           |
| reg_num/superconduct.csv                        | A/(R+1)         | 79->56            | 54584.570517          |
| reg_num/superconduct.csv                        | **Wasserstein** | **79->69**        | **54009.203279**      |
| reg_cat/topo_2_1.csv                            | built-in        | 255->217          | 76175.864005          |
| reg_cat/topo_2_1.csv                            | A-R             | 255->254          | 76311.964795          |
| reg_cat/topo_2_1.csv                            | A/(R+1)         | 255->79           | 76059.172223          |
| reg_cat/topo_2_1.csv                            | **Wasserstein** | **255->165**      | **75797.079183**      |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | **built-in**    | **359->6**        | **177937.918395**     |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | A-R             | 359->195          | 183867.243148         |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | **A/(R+1)**     | **359->6**        | **177937.918395**     |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | Wasserstein     | 359->96           | 195247.864875         |
| reg_cat/house_sales.csv                         | **built-in**    | **17->16**        | **110072.875485**     |
| reg_cat/house_sales.csv                         | A-R             | 17->17            | 110141.291334         |
| reg_cat/house_sales.csv                         | A/(R+1)         | 17->17            | 110404.08618          |
| reg_cat/house_sales.csv                         | Wasserstein     | 17->17            | 110078.623402         |
| reg_cat/nyc-taxi-green-dec-2016.csv             | **built-in**    | **16->15**        | **10585.637729**      |
| reg_cat/nyc-taxi-green-dec-2016.csv             | A-R             | 16->4             | 10758.481089          |
| reg_cat/nyc-taxi-green-dec-2016.csv             | A/(R+1)         | 16->16            | 10590.967684          |
| reg_cat/nyc-taxi-green-dec-2016.csv             | Wasserstein     | 16->16            | 10600.876808          |
| reg_cat/Allstate_Claims_Severity.csv            | **built-in**    | **124->113**      | **1002055785.041467** |
| reg_cat/Allstate_Claims_Severity.csv            | A-R             | 124->124          | 1003062488.299886     |
| reg_cat/Allstate_Claims_Severity.csv            | A/(R+1)         | 124->93           | 1003238483.814182     |
| reg_cat/Allstate_Claims_Severity.csv            | Wasserstein     | 124->84           | 1002670828.551967     |

---


## Binary Classification Results with XGBoost
| dataset                                    | importances     | feature_reduction | test_score   |
|--------------------------------------------|-----------------|-------------------|--------------|
| clf_cat/electricity.csv                    | **built-in**    | **8->4**          | **0.910815** |
| clf_cat/electricity.csv                    | **A-R**         | **8->4**          | **0.910815** |
| clf_cat/electricity.csv                    | **A/(R+1)**     | **8->4**          | **0.910815** |
| clf_cat/electricity.csv                    | Wasserstein     | 8->8              | 0.901493     |
| clf_cat/eye_movements.csv                  | built-in        | 23->2             | 0.647699     |
| clf_cat/eye_movements.csv                  | A-R             | 23->1             | 0.647699     |
| clf_cat/eye_movements.csv                  | **A/(R+1)**     | **23->10**        | **0.685209** |
| clf_cat/eye_movements.csv                  | Wasserstein     | 23->1             | 0.647699     |
| clf_cat/covertype.csv                      | **built-in**    | **54->40**        | **0.891799** |
| clf_cat/covertype.csv                      | A-R             | 54->53            | 0.890788     |
| clf_cat/covertype.csv                      | **A/(R+1)**     | **54->40**        | **0.891799** |
| clf_cat/covertype.csv                      | Wasserstein     | 54->54            | 0.887196     |
| clf_cat/albert.csv                         | built-in        | 31->11            | 0.651735     |
| clf_cat/albert.csv                         | **A-R**         | **31->20**        | **0.653106** |
| clf_cat/albert.csv                         | A/(R+1)         | 31->11            | 0.651735     |
| clf_cat/albert.csv                         | Wasserstein     | 31->25            | 0.643385     |
| clf_cat/compas-two-years.csv               | built-in        | 11->4             | 0.66599      |
| clf_cat/compas-two-years.csv               | **A-R**         | **11->2**         | **0.690587** |
| clf_cat/compas-two-years.csv               | A/(R+1)         | 11->5             | 0.66599      |
| clf_cat/compas-two-years.csv               | Wasserstein     | 11->6             | 0.66599      |
| clf_cat/default-of-credit-card-clients.csv | built-in        | 21->17            | 0.677356     |
| clf_cat/default-of-credit-card-clients.csv | A-R             | 21->15            | 0.676268     |
| clf_cat/default-of-credit-card-clients.csv | A/(R+1)         | 21->16            | 0.680071     |
| clf_cat/default-of-credit-card-clients.csv | **Wasserstein** | **21->19**        | **0.685076** |
| clf_cat/road-safety.csv                    | **built-in**    | **32->20**        | **0.792338** |
| clf_cat/road-safety.csv                    | A-R             | 32->30            | 0.789321     |
| clf_cat/road-safety.csv                    | A/(R+1)         | 32->27            | 0.790174     |
| clf_cat/road-safety.csv                    | Wasserstein     | 32->31            | 0.78271      |
| clf_num/Bioresponse.csv                    | built-in        | 419->82           | 0.740794     |
| clf_num/Bioresponse.csv                    | **A-R**         | **419->67**       | **0.763121** |
| clf_num/Bioresponse.csv                    | A/(R+1)         | 419->68           | 0.745468     |
| clf_num/Bioresponse.csv                    | Wasserstein     | 419->130          | 0.754958     |
| clf_num/jannis.csv                         | built-in        | 54->17            | 0.793181     |
| clf_num/jannis.csv                         | **A-R**         | **54->26**        | **0.796462** |
| clf_num/jannis.csv                         | **A/(R+1)**     | **54->26**        | **0.796462** |
| clf_num/jannis.csv                         | Wasserstein     | 54->52            | 0.784297     |
| clf_num/MiniBooNE.csv                      | built-in        | 50->45            | 0.937386     |
| clf_num/MiniBooNE.csv                      | A-R             | 50->38            | 0.937817     |
| clf_num/MiniBooNE.csv                      | **A/(R+1)**     | **50->47**        | **0.93797**  |
| clf_num/MiniBooNE.csv                      | Wasserstein     | 50->48            | 0.936908     |

---

## Regression Results with XGBoost
| dataset                                         | importances     | feature_reduction | test_score           |
|-------------------------------------------------|-----------------|-------------------|----------------------|
| reg_num/cpu_act.csv                             | built-in        | 21->17            | 5.946372             |
| reg_num/cpu_act.csv                             | **A-R**         | **21->18**        | **5.654754**         |
| reg_num/cpu_act.csv                             | A/(R+1)         | 21->18            | 5.679894             |
| reg_num/cpu_act.csv                             | Wasserstein     | 21->21            | 5.752541             |
| reg_num/pol.csv                                 | built-in        | 26->15            | 0.305836             |
| reg_num/pol.csv                                 | A-R             | 26->25            | 0.300197             |
| reg_num/pol.csv                                 | A/(R+1)         | 26->15            | 0.305665             |
| reg_num/pol.csv                                 | **Wasserstein** | **26->26**        | **0.296534**         |
| reg_num/elevators.csv                           | built-in        | 16->15            | 5.715875             |
| reg_num/elevators.csv                           | A-R             | 16->16            | 5.799774             |
| reg_num/elevators.csv                           | **A/(R+1)**     | **16->13**        | **5.598146**         |
| reg_num/elevators.csv                           | Wasserstein     | 16->16            | 5.76182              |
| reg_num/wine_quality.csv                        | built-in        | 11->10            | 0.463778             |
| reg_num/wine_quality.csv                        | A-R             | 11->11            | 0.45338              |
| reg_num/wine_quality.csv                        | A/(R+1)         | 11->11            | 0.453376             |
| reg_num/wine_quality.csv                        | **Wasserstein** | **11->11**        | **0.449631**         |
| reg_num/Ailerons.csv                            | **built-in**    | **33->21**        | **2.771169**         |
| reg_num/Ailerons.csv                            | A-R             | 33->24            | 2.779444             |
| reg_num/Ailerons.csv                            | A/(R+1)         | 33->22            | 2.807877             |
| reg_num/Ailerons.csv                            | Wasserstein     | 33->26            | 2.880596             |
| reg_num/yprop_4_1.csv                           | **built-in**    | **42->2**         | **78997.96633**      |
| reg_num/yprop_4_1.csv                           | **A-R**         | **42->2**         | **78997.96633**      |
| reg_num/yprop_4_1.csv                           | **A/(R+1)**     | **42->2**         | **78997.96633**      |
| reg_num/yprop_4_1.csv                           | Wasserstein     | 42->1             | 80189.002714         |
| reg_num/superconduct.csv                        | built-in        | 79->40            | 58619.061023         |
| reg_num/superconduct.csv                        | A-R             | 79->36            | 58742.531663         |
| reg_num/superconduct.csv                        | **A/(R+1)**     | **79->39**        | **58312.179595**     |
| reg_num/superconduct.csv                        | Wasserstein     | 79->73            | 58888.372081         |
| reg_cat/topo_2_1.csv                            | built-in        | 255->169          | 85043.995838         |
| reg_cat/topo_2_1.csv                            | **A-R**         | **255->115**      | **84843.36509**      |
| reg_cat/topo_2_1.csv                            | A/(R+1)         | 255->180          | 85421.986381         |
| reg_cat/topo_2_1.csv                            | Wasserstein     | 255->176          | 87021.271777         |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | **built-in**    | **359->7**        | **175536.434647**    |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | A-R             | 359->82           | 179004.576184        |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | **A/(R+1)**     | **359->13**       | **175536.434647**    |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | Wasserstein     | 359->14           | 175717.216092        |
| reg_cat/house_sales.csv                         | **built-in**    | **17->17**        | **105445.807214**    |
| reg_cat/house_sales.csv                         | A-R             | 17->16            | 107369.173579        |
| reg_cat/house_sales.csv                         | A/(R+1)         | 17->17            | 105474.801274        |
| reg_cat/house_sales.csv                         | Wasserstein     | 17->17            | 105647.831015        |
| reg_cat/nyc-taxi-green-dec-2016.csv             | built-in        | 16->7             | 11361.646796         |
| reg_cat/nyc-taxi-green-dec-2016.csv             | **A-R**         | **16->7**         | **11361.5967**       |
| reg_cat/nyc-taxi-green-dec-2016.csv             | A/(R+1)         | 16->7             | 11361.944294         |
| reg_cat/nyc-taxi-green-dec-2016.csv             | Wasserstein     | 16->4             | 11586.532828         |
| reg_cat/Allstate_Claims_Severity.csv            | **built-in**    | **124->69**       | **928207411.901454** |
| reg_cat/Allstate_Claims_Severity.csv            | A-R             | 124->112          | 928486398.650217     |
| reg_cat/Allstate_Claims_Severity.csv            | A/(R+1)         | 124->73           | 930825046.195666     |
| reg_cat/Allstate_Claims_Severity.csv            | Wasserstein     | 124->124          | 929893696.608902     |

---

## Binary Classification Results with LightGBM
| dataset                                    | importances     | feature_reduction | test_score   |
|--------------------------------------------|-----------------|-------------------|--------------|
| clf_cat/electricity.csv                    | **built-in**    | **8->5**          | **0.877175** |
| clf_cat/electricity.csv                    | **A-R**         | **8->5**          | **0.877175** |
| clf_cat/electricity.csv                    | **A/(R+1)**     | **8->5**          | **0.877175** |
| clf_cat/electricity.csv                    | **Wasserstein** | **8->5**          | **0.877175** |
| clf_cat/eye_movements.csv                  | built-in        | 23->23            | 0.632579     |
| clf_cat/eye_movements.csv                  | A-R             | 23->12            | 0.668197     |
| clf_cat/eye_movements.csv                  | A/(R+1)         | 23->13            | 0.666667     |
| clf_cat/eye_movements.csv                  | **Wasserstein** | **23->6**         | **0.668401** |
| clf_cat/covertype.csv                      | **built-in**    | **54->19**        | **0.851571** |
| clf_cat/covertype.csv                      | A-R             | 54->25            | 0.85109      |
| clf_cat/covertype.csv                      | A/(R+1)         | 54->43            | 0.847426     |
| clf_cat/covertype.csv                      | Wasserstein     | 54->25            | 0.85109      |
| clf_cat/albert.csv                         | built-in        | 31->14            | 0.665288     |
| clf_cat/albert.csv                         | A-R             | 31->19            | 0.666948     |
| clf_cat/albert.csv                         | **A/(R+1)**     | **31->17**        | **0.667958** |
| clf_cat/albert.csv                         | Wasserstein     | 31->22            | 0.663828     |
| clf_cat/compas-two-years.csv               | built-in        | 11->3             | 0.655332     |
| clf_cat/compas-two-years.csv               | **A-R**         | **11->4**         | **0.66972**  |
| clf_cat/compas-two-years.csv               | **A/(R+1)**     | **11->4**         | **0.66972**  |
| clf_cat/compas-two-years.csv               | **Wasserstein** | **11->4**         | **0.66972**  |
| clf_cat/default-of-credit-card-clients.csv | built-in        | 21->20            | 0.689504     |
| clf_cat/default-of-credit-card-clients.csv | **A-R**         | **21->14**        | **0.689751** |
| clf_cat/default-of-credit-card-clients.csv | **A/(R+1)**     | **21->14**        | **0.689751** |
| clf_cat/default-of-credit-card-clients.csv | Wasserstein     | 21->21            | 0.684882     |
| clf_cat/road-safety.csv                    | built-in        | 32->23            | 0.792133     |
| clf_cat/road-safety.csv                    | A-R             | 32->21            | 0.791898     |
| clf_cat/road-safety.csv                    | A/(R+1)         | 32->27            | 0.792048     |
| clf_cat/road-safety.csv                    | **Wasserstein** | **32->19**        | **0.792245** |
| clf_num/Bioresponse.csv                    | built-in        | 419->22           | 0.752857     |
| clf_num/Bioresponse.csv                    | **A-R**         | **419->416**      | **0.762108** |
| clf_num/Bioresponse.csv                    | A/(R+1)         | 419->260          | 0.757295     |
| clf_num/Bioresponse.csv                    | Wasserstein     | 419->60           | 0.753395     |
| clf_num/jannis.csv                         | built-in        | 54->24            | 0.797305     |
| clf_num/jannis.csv                         | A-R             | 54->25            | 0.798759     |
| clf_num/jannis.csv                         | **A/(R+1)**     | **54->24**        | **0.800101** |
| clf_num/jannis.csv                         | Wasserstein     | 54->32            | 0.797517     |
| clf_num/MiniBooNE.csv                      | built-in        | 50->40            | 0.936987     |
| clf_num/MiniBooNE.csv                      | A-R             | 50->37            | 0.937139     |
| clf_num/MiniBooNE.csv                      | A/(R+1)         | 50->37            | 0.937139     |
| clf_num/MiniBooNE.csv                      | **Wasserstein** | **50->45**        | **0.93793**  |

---
## Regression Results with LightGBM
| dataset                                         | importances     | feature_reduction | test_score           |
|-------------------------------------------------|-----------------|-------------------|----------------------|
| reg_num/cpu_act.csv                             | **built-in**    | **21->15**        | **5.125908**         |
| reg_num/cpu_act.csv                             | A-R             | 21->21            | 5.188476             |
| reg_num/cpu_act.csv                             | A/(R+1)         | 21->21            | 5.188049             |
| reg_num/cpu_act.csv                             | Wasserstein     | 21->18            | 5.158656             |
| reg_num/pol.csv                                 | built-in        | 26->13            | 0.274706             |
| reg_num/pol.csv                                 | A-R             | 26->25            | 0.278663             |
| reg_num/pol.csv                                 | A/(R+1)         | 26->14            | 0.278948             |
| reg_num/pol.csv                                 | **Wasserstein** | **26->14**        | **0.26098**          |
| reg_num/elevators.csv                           | built-in        | 16->16            | 5.510428             |
| reg_num/elevators.csv                           | **A-R**         | **16->16**        | **5.510406**         |
| reg_num/elevators.csv                           | **A/(R+1)**     | **16->16**        | **5.510406**         |
| reg_num/elevators.csv                           | **Wasserstein** | **16->16**        | **5.510406**         |
| reg_num/wine_quality.csv                        | **built-in**    | **11->11**        | **0.437608**         |
| reg_num/wine_quality.csv                        | **A-R**         | **11->11**        | **0.437608**         |
| reg_num/wine_quality.csv                        | **A/(R+1)**     | **11->11**        | **0.437608**         |
| reg_num/wine_quality.csv                        | **Wasserstein** | **11->11**        | **0.437608**         |
| reg_num/Ailerons.csv                            | built-in        | 33->22            | 2.600292             |
| reg_num/Ailerons.csv                            | A-R             | 33->29            | 2.57817              |
| reg_num/Ailerons.csv                            | **A/(R+1)**     | **33->27**        | **2.560278**         |
| reg_num/Ailerons.csv                            | Wasserstein     | 33->28            | 2.595686             |
| reg_num/yprop_4_1.csv                           | **built-in**    | **42->29**        | **75930.237995**     |
| reg_num/yprop_4_1.csv                           | A-R             | 42->34            | 76201.666859         |
| reg_num/yprop_4_1.csv                           | A/(R+1)         | 42->31            | 76186.560793         |
| reg_num/yprop_4_1.csv                           | Wasserstein     | 42->19            | 76494.373796         |
| reg_num/superconduct.csv                        | built-in        | 79->55            | 63228.223091         |
| reg_num/superconduct.csv                        | **A-R**         | **79->76**        | **62647.41587**      |
| reg_num/superconduct.csv                        | A/(R+1)         | 79->70            | 62806.379162         |
| reg_num/superconduct.csv                        | Wasserstein     | 79->70            | 63367.083612         |
| reg_cat/topo_2_1.csv                            | built-in        | 255->84           | 77167.225165         |
| reg_cat/topo_2_1.csv                            | A-R             | 255->32           | 78130.633542         |
| reg_cat/topo_2_1.csv                            | **A/(R+1)**     | **255->153**      | **77161.555463**     |
| reg_cat/topo_2_1.csv                            | Wasserstein     | 255->232          | 77395.701091         |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | built-in        | 359->87           | 189609.18945         |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | **A-R**         | **359->14**       | **174764.764814**    |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | A/(R+1)         | 359->69           | 191855.328702        |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | Wasserstein     | 359->183          | 191374.726273        |
| reg_cat/house_sales.csv                         | built-in        | 17->14            | 97213.085621         |
| reg_cat/house_sales.csv                         | **A-R**         | **17->14**        | **96600.951963**     |
| reg_cat/house_sales.csv                         | A/(R+1)         | 17->11            | 96625.662916         |
| reg_cat/house_sales.csv                         | Wasserstein     | 17->12            | 100009.945837        |
| reg_cat/nyc-taxi-green-dec-2016.csv             | built-in        | 16->5             | 13655.984718         |
| reg_cat/nyc-taxi-green-dec-2016.csv             | A-R             | 16->6             | 13632.35428          |
| reg_cat/nyc-taxi-green-dec-2016.csv             | **A/(R+1)**     | **16->8**         | **13623.652139**     |
| reg_cat/nyc-taxi-green-dec-2016.csv             | Wasserstein     | 16->6             | 13632.35428          |
| reg_cat/Allstate_Claims_Severity.csv            | **built-in**    | **124->82**       | **918539118.574563** |
| reg_cat/Allstate_Claims_Severity.csv            | A-R             | 124->78           | 919651444.584081     |
| reg_cat/Allstate_Claims_Severity.csv            | A/(R+1)         | 124->102          | 919303774.163119     |
| reg_cat/Allstate_Claims_Severity.csv            | Wasserstein     | 124->95           | 918926264.810142     |

---

## Binary Classification Results with CatBoost
| dataset                                    | importances     | feature_reduction | test_score   |
|--------------------------------------------|-----------------|-------------------|--------------|
| clf_cat/electricity.csv                    | **built-in**    | **8->3**          | **0.883178** |
| clf_cat/electricity.csv                    | **A-R**         | **8->3**          | **0.883178** |
| clf_cat/electricity.csv                    | **A/(R+1)**     | **8->3**          | **0.883178** |
| clf_cat/electricity.csv                    | Wasserstein     | 8->8              | 0.883155     |
| clf_cat/eye_movements.csv                  | built-in        | 23->23            | 0.628159     |
| clf_cat/eye_movements.csv                  | A-R             | 23->14            | 0.64964      |
| clf_cat/eye_movements.csv                  | **A/(R+1)**     | **23->5**         | **0.656049** |
| clf_cat/eye_movements.csv                  | Wasserstein     | 23->22            | 0.625205     |
| clf_cat/covertype.csv                      | **built-in**    | **54->24**        | **0.913111** |
| clf_cat/covertype.csv                      | A-R             | 54->54            | 0.910959     |
| clf_cat/covertype.csv                      | A/(R+1)         | 54->35            | 0.912809     |
| clf_cat/covertype.csv                      | Wasserstein     | 54->30            | 0.913034     |
| clf_cat/albert.csv                         | built-in        | 31->15            | 0.665236     |
| clf_cat/albert.csv                         | A-R             | 31->24            | 0.66661      |
| clf_cat/albert.csv                         | **A/(R+1)**     | **31->18**        | **0.6679**   |
| clf_cat/albert.csv                         | Wasserstein     | 31->31            | 0.666891     |
| clf_cat/compas-two-years.csv               | **built-in**    | **11->7**         | **0.675926** |
| clf_cat/compas-two-years.csv               | A-R             | 11->10            | 0.674274     |
| clf_cat/compas-two-years.csv               | A/(R+1)         | 11->9             | 0.675244     |
| clf_cat/compas-two-years.csv               | Wasserstein     | 11->7             | 0.674286     |
| clf_cat/default-of-credit-card-clients.csv | built-in        | 21->21            | 0.690341     |
| clf_cat/default-of-credit-card-clients.csv | A-R             | 21->21            | 0.689137     |
| clf_cat/default-of-credit-card-clients.csv | A/(R+1)         | 21->21            | 0.689978     |
| clf_cat/default-of-credit-card-clients.csv | **Wasserstein** | **21->20**        | **0.69052**  |
| clf_cat/road-safety.csv                    | **built-in**    | **32->27**        | **0.794**    |
| clf_cat/road-safety.csv                    | A-R             | 32->28            | 0.793317     |
| clf_cat/road-safety.csv                    | A/(R+1)         | 32->22            | 0.791271     |
| clf_cat/road-safety.csv                    | Wasserstein     | 32->30            | 0.79264      |
| clf_num/Bioresponse.csv                    | built-in        | 419->290          | 0.781006     |
| clf_num/Bioresponse.csv                    | A-R             | 419->419          | 0.786373     |
| clf_num/Bioresponse.csv                    | A/(R+1)         | 419->206          | 0.784507     |
| clf_num/Bioresponse.csv                    | **Wasserstein** | **419->386**      | **0.787368** |
| clf_num/jannis.csv                         | built-in        | 54->18            | 0.806811     |
| clf_num/jannis.csv                         | A-R             | 54->26            | 0.80771      |
| clf_num/jannis.csv                         | **A/(R+1)**     | **54->22**        | **0.810063** |
| clf_num/jannis.csv                         | Wasserstein     | 54->49            | 0.801057     |
| clf_num/MiniBooNE.csv                      | built-in        | 50->48            | 0.942751     |
| clf_num/MiniBooNE.csv                      | **A-R**         | **50->46**        | **0.943224** |
| clf_num/MiniBooNE.csv                      | A/(R+1)         | 50->45            | 0.942883     |
| clf_num/MiniBooNE.csv                      | Wasserstein     | 50->50            | 0.943038     |

---

## Regression Results with CatBoost
| dataset                                         | importances     | feature_reduction | test_score           |
|-------------------------------------------------|-----------------|-------------------|----------------------|
| reg_num/cpu_act.csv                             | **built-in**    | **21->16**        | **5.104869**         |
| reg_num/cpu_act.csv                             | A-R             | 21->21            | 5.13585              |
| reg_num/cpu_act.csv                             | A/(R+1)         | 21->20            | 5.1915               |
| reg_num/cpu_act.csv                             | Wasserstein     | 21->21            | 5.127099             |
| reg_num/pol.csv                                 | built-in        | 26->15            | 0.271039             |
| reg_num/pol.csv                                 | A-R             | 26->25            | 0.271706             |
| reg_num/pol.csv                                 | A/(R+1)         | 26->20            | 0.26218              |
| reg_num/pol.csv                                 | **Wasserstein** | **26->20**        | **0.260988**         |
| reg_num/elevators.csv                           | built-in        | 16->15            | 4.324409             |
| reg_num/elevators.csv                           | A-R             | 16->15            | 4.34998              |
| reg_num/elevators.csv                           | **A/(R+1)**     | **16->16**        | **4.321757**         |
| reg_num/elevators.csv                           | Wasserstein     | 16->16            | 4.35382              |
| reg_num/wine_quality.csv                        | built-in        | 11->11            | 0.433982             |
| reg_num/wine_quality.csv                        | **A-R**         | **11->11**        | **0.428414**         |
| reg_num/wine_quality.csv                        | A/(R+1)         | 11->10            | 0.433384             |
| reg_num/wine_quality.csv                        | Wasserstein     | 11->11            | 0.433185             |
| reg_num/Ailerons.csv                            | built-in        | 33->28            | 2.42977              |
| reg_num/Ailerons.csv                            | **A-R**         | **33->28**        | **2.41639**          |
| reg_num/Ailerons.csv                            | A/(R+1)         | 33->24            | 2.440884             |
| reg_num/Ailerons.csv                            | Wasserstein     | 33->19            | 2.45879              |
| reg_num/yprop_4_1.csv                           | built-in        | 42->29            | 75892.248828         |
| reg_num/yprop_4_1.csv                           | A-R             | 42->41            | 75664.082657         |
| reg_num/yprop_4_1.csv                           | **A/(R+1)**     | **42->28**        | **75348.084082**     |
| reg_num/yprop_4_1.csv                           | Wasserstein     | 42->32            | 75485.168192         |
| reg_num/superconduct.csv                        | **built-in**    | **79->67**        | **57399.093773**     |
| reg_num/superconduct.csv                        | A-R             | 79->73            | 57487.193807         |
| reg_num/superconduct.csv                        | A/(R+1)         | 79->63            | 57478.518356         |
| reg_num/superconduct.csv                        | Wasserstein     | 79->75            | 57710.629917         |
| reg_cat/topo_2_1.csv                            | **built-in**    | **255->176**      | **75530.315952**     |
| reg_cat/topo_2_1.csv                            | A-R             | 255->240          | 76035.257684         |
| reg_cat/topo_2_1.csv                            | A/(R+1)         | 255->226          | 76258.199946         |
| reg_cat/topo_2_1.csv                            | Wasserstein     | 255->210          | 76661.935034         |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | built-in        | 359->338          | 189052.696626        |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | **A-R**         | **359->10**       | **174728.12932**     |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | A/(R+1)         | 359->9            | 174846.635809        |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | Wasserstein     | 359->318          | 189125.782494        |
| reg_cat/house_sales.csv                         | built-in        | 17->16            | 91482.325955         |
| reg_cat/house_sales.csv                         | A-R             | 17->16            | 91295.138498         |
| reg_cat/house_sales.csv                         | **A/(R+1)**     | **17->16**        | **91173.085911**     |
| reg_cat/house_sales.csv                         | Wasserstein     | 17->17            | 91505.900609         |
| reg_cat/nyc-taxi-green-dec-2016.csv             | **built-in**    | **16->12**        | **12548.48941**      |
| reg_cat/nyc-taxi-green-dec-2016.csv             | A-R             | 16->16            | 12596.921199         |
| reg_cat/nyc-taxi-green-dec-2016.csv             | A/(R+1)         | 16->16            | 12577.487796         |
| reg_cat/nyc-taxi-green-dec-2016.csv             | Wasserstein     | 16->16            | 12633.541154         |
| reg_cat/Allstate_Claims_Severity.csv            | built-in        | 124->106          | 905710139.82245      |
| reg_cat/Allstate_Claims_Severity.csv            | A-R             | 124->124          | 906265053.899284     |
| reg_cat/Allstate_Claims_Severity.csv            | **A/(R+1)**     | **124->106**      | **905170209.556065** |
| reg_cat/Allstate_Claims_Severity.csv            | Wasserstein     | 124->120          | 905989468.489141     |

---