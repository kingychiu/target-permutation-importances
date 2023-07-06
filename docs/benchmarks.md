# Benchmarks

Benchmark has been done with some tabular datasets from the [Tabular data learning benchmark](https://github.com/LeoGrin/tabular-benchmark/tree/main). It is also
hosted on [Hugging Face](https://huggingface.co/datasets/inria-soda/tabular-benchmark).

For the binary classification task, `sklearn.metrics.f1_score` is used for evaluation. For the regression task, `sklearn.metrics.mean_squared_error` is used for evaluation.

The downloaded datasets are divided into 3 sections: `train`: 50%, `val`: 10%, `test`: 40%.
Feature importance is calculated from the `train` set. Feature selection is done on the `val` set. 
The final benchmark is evaluated on the `test` set. Therefore the `test` set is unseen to both the feature importance and selection process.

Here is the summary of running null-importances with feature selection on multiple models and datasets. "better" means it is better than running feature selection with the model's built-in feature importances. We can see even with **with default models' parameters** it shows its effectiveness.

| model                  | n_dataset | n_better | better % |
|------------------------|-----------|----------|----------|
| CatBoostClassifier     | 10        | 6        | 60.0     |
| CatBoostRegressor      | 12        | 8        | 66.67    |
| LGBMClassifier         | 10        | 7        | 70.0     |
| LGBMRegressor          | 12        | 6        | 50.0     |
| RandomForestClassifier | 10        | 9        | 90.0     |
| RandomForestRegressor  | 12        | 7        | 58.33    |
| XGBClassifier          | 10        | 5        | 50.0     |
| XGBRegressor           | 12        | 5        | 41.67    |


Below tables shows the raw results and the raw csv data are in [`benchmarks/results`](https://github.com/kingychiu/target-permutation-importances/tree/main/benchmarks/results).


built-in: The baseline, it is the built-in importances from the model.
A-R: compute_permutation_importance_by_subtraction
A/(R+1): compute_permutation_importance_by_division

## Binary Classification Results with RandomForest

| dataset                                    | importances  | feature_reduction | test_f1    |
| ------------------------------------------ | ------------ | ----------------- | ---------- |
| clf_cat/electricity.csv                    | built-in     | 8->2              | 0.894      |
| clf_cat/electricity.csv                    | **A-R**      | 8->4              | **0.9034** |
| clf_cat/electricity.csv                    | A/(R+1)      | 8->2              | 0.894      |
| clf_cat/eye_movements.csv                  | built-in     | 23->22            | 0.6169     |
| clf_cat/eye_movements.csv                  | **A-R**      | 23->10            | **0.6772** |
| clf_cat/eye_movements.csv                  | A/(R+1)      | 23->22            | 0.6212     |
| clf_cat/covertype.csv                      | built-in     | 54->26            | 0.9558     |
| clf_cat/covertype.csv                      | **A-R**      | 54->52            | **0.9586** |
| clf_cat/covertype.csv                      | A/(R+1)      | 54->30            | 0.9547     |
| clf_cat/albert.csv                         | built-in     | 31->22            | 0.6518     |
| clf_cat/albert.csv                         | **A-R**      | 31->24            | **0.6587** |
| clf_cat/albert.csv                         | A/(R+1)      | 31->22            | 0.6527     |
| clf_cat/compas-two-years.csv               | built-in     | 11->10            | 0.6316     |
| clf_cat/compas-two-years.csv               | **A-R**      | 11->2             | **0.6589** |
| clf_cat/compas-two-years.csv               | A/(R+1)      | 11->6             | 0.6335     |
| clf_cat/default-of-credit-card-clients.csv | built-in     | 21->18            | 0.671      |
| clf_cat/default-of-credit-card-clients.csv | **A-R**      | 21->17            | **0.6826** |
| clf_cat/default-of-credit-card-clients.csv | A/(R+1)      | 21->20            | 0.6797     |
| clf_cat/road-safety.csv                    | **built-in** | 32->31            | **0.7895** |
| clf_cat/road-safety.csv                    | A-R          | 32->30            | 0.7886     |
| clf_cat/road-safety.csv                    | A/(R+1)      | 32->29            | 0.7893     |
| clf_num/Bioresponse.csv                    | built-in     | 419->295          | 0.7686     |
| clf_num/Bioresponse.csv                    | A-R          | 419->214          | 0.7692     |
| clf_num/Bioresponse.csv                    | **A/(R+1)**  | 419->403          | **0.775**  |
| clf_num/jannis.csv                         | built-in     | 54->22            | 0.7958     |
| clf_num/jannis.csv                         | A-R          | 54->28            | 0.7988     |
| clf_num/jannis.csv                         | **A/(R+1)**  | 54->26            | **0.7998** |
| clf_num/MiniBooNE.csv                      | built-in     | 50->33            | 0.9306     |
| clf_num/MiniBooNE.csv                      | A-R          | 50->47            | 0.93       |
| clf_num/MiniBooNE.csv                      | **A/(R+1)**  | 50->49            | **0.9316** |

## Regression Results with RandomForest

| dataset                                         | importances  | feature_reduction | test_mse            |
| ----------------------------------------------- | ------------ | ----------------- | ------------------- |
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

---


## Binary Classification Results with XGBoost
| dataset                                    | importances         | feature_reduction | test_f1   |
|--------------------------------------------|---------------------|-------------------|--------------|
| clf_cat/electricity.csv                    | **built-in (gain)** | **8->4**          | **0.910815** |
| clf_cat/electricity.csv                    | **A-R**             | **8->4**          | **0.910815** |
| clf_cat/electricity.csv                    | **A/(R+1)**         | **8->4**          | **0.910815** |
| clf_cat/eye_movements.csv                  | built-in (gain)     | 23->2             | 0.647699     |
| clf_cat/eye_movements.csv                  | **A-R**             | **23->5**         | **0.647721** |
| clf_cat/eye_movements.csv                  | A/(R+1)             | 23->2             | 0.647699     |
| clf_cat/covertype.csv                      | **built-in (gain)** | **54->40**        | **0.891799** |
| clf_cat/covertype.csv                      | A-R                 | 54->42            | 0.890554     |
| clf_cat/covertype.csv                      | **A/(R+1)**         | **54->40**        | **0.891799** |
| clf_cat/albert.csv                         | **built-in (gain)** | **31->11**        | **0.651735** |
| clf_cat/albert.csv                         | A-R                 | 31->28            | 0.651434     |
| clf_cat/albert.csv                         | **A/(R+1)**         | **31->11**        | **0.651735** |
| clf_cat/compas-two-years.csv               | **built-in (gain)** | **11->4**         | **0.66599**  |
| clf_cat/compas-two-years.csv               | A-R                 | 11->7             | 0.63963      |
| clf_cat/compas-two-years.csv               | A/(R+1)             | 11->5             | 0.660931     |
| clf_cat/default-of-credit-card-clients.csv | built-in (gain)     | 21->17            | 0.677356     |
| clf_cat/default-of-credit-card-clients.csv | **A-R**             | **21->20**        | **0.679452** |
| clf_cat/default-of-credit-card-clients.csv | A/(R+1)             | 21->17            | 0.677356     |
| clf_cat/road-safety.csv                    | **built-in (gain)** | **32->20**        | **0.792338** |
| clf_cat/road-safety.csv                    | A-R                 | 32->27            | 0.791672     |
| clf_cat/road-safety.csv                    | **A/(R+1)**         | **32->20**        | **0.792338** |
| clf_num/Bioresponse.csv                    | built-in (gain)     | 419->82           | 0.740794     |
| clf_num/Bioresponse.csv                    | **A-R**             | **419->83**       | **0.749458** |
| clf_num/Bioresponse.csv                    | A/(R+1)             | 419->82           | 0.740794     |
| clf_num/jannis.csv                         | built-in (gain)     | 54->17            | 0.793181     |
| clf_num/jannis.csv                         | **A-R**             | **54->26**        | **0.796462** |
| clf_num/jannis.csv                         | A/(R+1)             | 54->17            | 0.793181     |
| clf_num/MiniBooNE.csv                      | built-in (gain)     | 50->45            | 0.937386     |
| clf_num/MiniBooNE.csv                      | **A-R**             | **50->41**        | **0.938551** |
| clf_num/MiniBooNE.csv                      | A/(R+1)             | 50->45            | 0.937386     |

## Regression Results with XGBoost
| dataset                                         | importances         | feature_reduction | test_mse           |
|-------------------------------------------------|---------------------|-------------------|----------------------|
| reg_num/cpu_act.csv                             | built-in (gain)     | 21->17            | 5.946372             |
| reg_num/cpu_act.csv                             | **A-R**             | **21->19**        | **5.82137**          |
| reg_num/cpu_act.csv                             | A/(R+1)             | 21->17            | 5.946372             |
| reg_num/pol.csv                                 | **built-in (gain)** | **26->15**        | **0.305836**         |
| reg_num/pol.csv                                 | A-R                 | 26->24            | 0.308747             |
| reg_num/pol.csv                                 | **A/(R+1)**         | **26->15**        | **0.305836**         |
| reg_num/elevators.csv                           | built-in (gain)     | 16->15            | 5.715875             |
| reg_num/elevators.csv                           | **A-R**             | **16->13**        | **5.495379**         |
| reg_num/elevators.csv                           | A/(R+1)             | 16->15            | 5.715875             |
| reg_num/wine_quality.csv                        | built-in (gain)     | 11->10            | 0.463778             |
| reg_num/wine_quality.csv                        | **A-R**             | **11->11**        | **0.453146**         |
| reg_num/wine_quality.csv                        | A/(R+1)             | 11->10            | 0.463778             |
| reg_num/Ailerons.csv                            | **built-in (gain)** | **33->21**        | **2.771169**         |
| reg_num/Ailerons.csv                            | A-R                 | 33->26            | 2.782902             |
| reg_num/Ailerons.csv                            | **A/(R+1)**         | **33->21**        | **2.771169**         |
| reg_num/yprop_4_1.csv                           | **built-in (gain)** | **42->2**         | **78997.96633**      |
| reg_num/yprop_4_1.csv                           | **A-R**             | **42->2**         | **78997.96633**      |
| reg_num/yprop_4_1.csv                           | **A/(R+1)**         | **42->2**         | **78997.96633**      |
| reg_num/superconduct.csv                        | **built-in (gain)** | **79->40**        | **58619.061023**     |
| reg_num/superconduct.csv                        | A-R                 | 79->61            | 59136.747286         |
| reg_num/superconduct.csv                        | A/(R+1)             | 79->40            | 58669.737431         |
| reg_cat/topo_2_1.csv                            | **built-in (gain)** | **255->169**      | **85043.995838**     |
| reg_cat/topo_2_1.csv                            | A-R                 | 255->198          | 85098.089041         |
| reg_cat/topo_2_1.csv                            | **A/(R+1)**         | **255->169**      | **85043.995838**     |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | built-in (gain)     | 359->7            | 175536.434647        |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | **A-R**             | **359->10**       | **174892.772843**    |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | A/(R+1)             | 359->7            | 175536.434647        |
| reg_cat/house_sales.csv                         | **built-in (gain)** | **17->17**        | **105445.807214**    |
| reg_cat/house_sales.csv                         | A-R                 | 17->16            | 107314.46122         |
| reg_cat/house_sales.csv                         | **A/(R+1)**         | **17->17**        | **105445.807214**    |
| reg_cat/nyc-taxi-green-dec-2016.csv             | built-in (gain)     | 16->7             | 11361.646796         |
| reg_cat/nyc-taxi-green-dec-2016.csv             | **A-R**             | **16->7**         | **11361.5967**       |
| reg_cat/nyc-taxi-green-dec-2016.csv             | A/(R+1)             | 16->7             | 11361.646796         |
| reg_cat/Allstate_Claims_Severity.csv            | **built-in (gain)** | **124->69**       | **928207411.901454** |
| reg_cat/Allstate_Claims_Severity.csv            | A-R                 | 124->115          | 930074469.019719     |
| reg_cat/Allstate_Claims_Severity.csv            | **A/(R+1)**         | **124->69**       | **928207411.901454** |


## Binary Classification Results with LightGBM
| dataset                                    | importances         | feature_reduction | test_f1   |
|--------------------------------------------|---------------------|-------------------|--------------|
| clf_cat/electricity.csv                    | **built-in (gain)** | **8->5**          | **0.877175** |
| clf_cat/electricity.csv                    | **A-R**             | **8->5**          | **0.877175** |
| clf_cat/electricity.csv                    | **A/(R+1)**         | **8->5**          | **0.877175** |
| clf_cat/eye_movements.csv                  | built-in (gain)     | 23->23            | 0.632579     |
| clf_cat/eye_movements.csv                  | **A-R**             | **23->7**         | **0.675835** |
| clf_cat/eye_movements.csv                  | A/(R+1)             | 23->13            | 0.666667     |
| clf_cat/covertype.csv                      | **built-in (gain)** | **54->19**        | **0.851571** |
| clf_cat/covertype.csv                      | A-R                 | 54->25            | 0.85109      |
| clf_cat/covertype.csv                      | A/(R+1)             | 54->43            | 0.847426     |
| clf_cat/albert.csv                         | built-in (gain)     | 31->14            | 0.665288     |
| clf_cat/albert.csv                         | A-R                 | 31->19            | 0.666948     |
| clf_cat/albert.csv                         | **A/(R+1)**         | **31->17**        | **0.667958** |
| clf_cat/compas-two-years.csv               | built-in (gain)     | 11->3             | 0.655332     |
| clf_cat/compas-two-years.csv               | A-R                 | 11->3             | 0.655332     |
| clf_cat/compas-two-years.csv               | **A/(R+1)**         | **11->3**         | **0.66972**  |
| clf_cat/default-of-credit-card-clients.csv | built-in (gain)     | 21->20            | 0.689504     |
| clf_cat/default-of-credit-card-clients.csv | **A-R**             | **21->14**        | **0.689751** |
| clf_cat/default-of-credit-card-clients.csv | **A/(R+1)**         | **21->14**        | **0.689751** |
| clf_cat/road-safety.csv                    | **built-in (gain)** | **32->23**        | **0.792133** |
| clf_cat/road-safety.csv                    | A-R                 | 32->25            | 0.791463     |
| clf_cat/road-safety.csv                    | A/(R+1)             | 32->24            | 0.791463     |
| clf_num/Bioresponse.csv                    | built-in (gain)     | 419->22           | 0.752857     |
| clf_num/Bioresponse.csv                    | **A-R**             | **419->80**       | **0.76824**  |
| clf_num/Bioresponse.csv                    | A/(R+1)             | 419->61           | 0.761429     |
| clf_num/jannis.csv                         | built-in (gain)     | 54->24            | 0.797305     |
| clf_num/jannis.csv                         | A-R                 | 54->25            | 0.798759     |
| clf_num/jannis.csv                         | **A/(R+1)**         | **54->24**        | **0.800101** |
| clf_num/MiniBooNE.csv                      | built-in (gain)     | 50->40            | 0.936987     |
| clf_num/MiniBooNE.csv                      | **A-R**             | **50->37**        | **0.937139** |
| clf_num/MiniBooNE.csv                      | **A/(R+1)**         | **50->37**        | **0.937139** |

## Regression Results with LightGBM
| dataset                                         | importances         | feature_reduction | test_mse           |
|-------------------------------------------------|---------------------|-------------------|----------------------|
| reg_num/cpu_act.csv                             | built-in (gain)     | 21->17            | 5.946372             |
| reg_num/cpu_act.csv                             | **A-R**             | **21->19**        | **5.82137**          |
| reg_num/cpu_act.csv                             | A/(R+1)             | 21->17            | 5.946372             |
| reg_num/pol.csv                                 | **built-in (gain)** | **26->15**        | **0.305836**         |
| reg_num/pol.csv                                 | A-R                 | 26->24            | 0.308747             |
| reg_num/pol.csv                                 | **A/(R+1)**         | **26->15**        | **0.305836**         |
| reg_num/elevators.csv                           | built-in (gain)     | 16->15            | 5.715875             |
| reg_num/elevators.csv                           | **A-R**             | **16->13**        | **5.495379**         |
| reg_num/elevators.csv                           | A/(R+1)             | 16->15            | 5.715875             |
| reg_num/wine_quality.csv                        | built-in (gain)     | 11->10            | 0.463778             |
| reg_num/wine_quality.csv                        | **A-R**             | **11->11**        | **0.453146**         |
| reg_num/wine_quality.csv                        | A/(R+1)             | 11->10            | 0.463778             |
| reg_num/Ailerons.csv                            | **built-in (gain)** | **33->21**        | **2.771169**         |
| reg_num/Ailerons.csv                            | A-R                 | 33->26            | 2.782902             |
| reg_num/Ailerons.csv                            | **A/(R+1)**         | **33->21**        | **2.771169**         |
| reg_num/yprop_4_1.csv                           | **built-in (gain)** | **42->2**         | **78997.96633**      |
| reg_num/yprop_4_1.csv                           | **A-R**             | **42->2**         | **78997.96633**      |
| reg_num/yprop_4_1.csv                           | **A/(R+1)**         | **42->2**         | **78997.96633**      |
| reg_num/superconduct.csv                        | **built-in (gain)** | **79->40**        | **58619.061023**     |
| reg_num/superconduct.csv                        | A-R                 | 79->61            | 59136.747286         |
| reg_num/superconduct.csv                        | A/(R+1)             | 79->40            | 58669.737431         |
| reg_cat/topo_2_1.csv                            | **built-in (gain)** | **255->169**      | **85043.995838**     |
| reg_cat/topo_2_1.csv                            | A-R                 | 255->198          | 85098.089041         |
| reg_cat/topo_2_1.csv                            | **A/(R+1)**         | **255->169**      | **85043.995838**     |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | built-in (gain)     | 359->7            | 175536.434647        |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | **A-R**             | **359->10**       | **174892.772843**    |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | A/(R+1)             | 359->7            | 175536.434647        |
| reg_cat/house_sales.csv                         | **built-in (gain)** | **17->17**        | **105445.807214**    |
| reg_cat/house_sales.csv                         | A-R                 | 17->16            | 107314.46122         |
| reg_cat/house_sales.csv                         | **A/(R+1)**         | **17->17**        | **105445.807214**    |
| reg_cat/nyc-taxi-green-dec-2016.csv             | built-in (gain)     | 16->7             | 11361.646796         |
| reg_cat/nyc-taxi-green-dec-2016.csv             | **A-R**             | **16->7**         | **11361.5967**       |
| reg_cat/nyc-taxi-green-dec-2016.csv             | A/(R+1)             | 16->7             | 11361.646796         |
| reg_cat/Allstate_Claims_Severity.csv            | **built-in (gain)** | **124->69**       | **928207411.901454** |
| reg_cat/Allstate_Claims_Severity.csv            | A-R                 | 124->115          | 930074469.019719     |
| reg_cat/Allstate_Claims_Severity.csv            | **A/(R+1)**         | **124->69**       | **928207411.901454** |


## Binary Classification Results with CatBoost
| dataset                                    | importances         | feature_reduction | test_f1   |
|--------------------------------------------|---------------------|-------------------|--------------|
| clf_cat/electricity.csv                    | **built-in** | **8->3**          | **0.883178** |
| clf_cat/electricity.csv                    | **A-R**             | **8->3**          | **0.883178** |
| clf_cat/electricity.csv                    | **A/(R+1)**         | **8->3**          | **0.883178** |
| clf_cat/eye_movements.csv                  | built-in     | 23->23            | 0.628159     |
| clf_cat/eye_movements.csv                  | A-R                 | 23->14            | 0.65817      |
| clf_cat/eye_movements.csv                  | **A/(R+1)**         | **23->11**        | **0.668376** |
| clf_cat/covertype.csv                      | **built-in** | **54->24**        | **0.913111** |
| clf_cat/covertype.csv                      | A-R                 | 54->54            | 0.91068      |
| clf_cat/covertype.csv                      | A/(R+1)             | 54->38            | 0.912124     |
| clf_cat/albert.csv                         | built-in     | 31->15            | 0.665236     |
| clf_cat/albert.csv                         | A-R                 | 31->20            | 0.664977     |
| clf_cat/albert.csv                         | **A/(R+1)**         | **31->16**        | **0.666892** |
| clf_cat/compas-two-years.csv               | **built-in** | **11->7**         | **0.675926** |
| clf_cat/compas-two-years.csv               | A-R                 | 11->9             | 0.674587     |
| clf_cat/compas-two-years.csv               | A/(R+1)             | 11->11            | 0.673913     |
| clf_cat/default-of-credit-card-clients.csv | **built-in** | **21->21**        | **0.690341** |
| clf_cat/default-of-credit-card-clients.csv | A-R                 | 21->20            | 0.688871     |
| clf_cat/default-of-credit-card-clients.csv | A/(R+1)             | 21->15            | 0.687753     |
| clf_cat/road-safety.csv                    | built-in     | 32->27            | 0.794        |
| clf_cat/road-safety.csv                    | A-R                 | 32->30            | 0.794475     |
| clf_cat/road-safety.csv                    | **A/(R+1)**         | **32->25**        | **0.794574** |
| clf_num/Bioresponse.csv                    | built-in     | 419->290          | 0.781006     |
| clf_num/Bioresponse.csv                    | A-R                 | 419->65           | 0.776184     |
| clf_num/Bioresponse.csv                    | **A/(R+1)**         | **419->224**      | **0.78903**  |
| clf_num/jannis.csv                         | built-in     | 54->18            | 0.806811     |
| clf_num/jannis.csv                         | **A-R**             | **54->28**        | **0.808289** |
| clf_num/jannis.csv                         | A/(R+1)             | 54->19            | 0.807781     |
| clf_num/MiniBooNE.csv                      | built-in     | 50->48            | 0.942751     |
| clf_num/MiniBooNE.csv                      | **A-R**             | **50->46**        | **0.943387** |
| clf_num/MiniBooNE.csv                      | A/(R+1)             | 50->44            | 0.943256     |


## Regression Results with CatBoost
| dataset                                         | importances         | feature_reduction | test_mse          |
|-------------------------------------------------|---------------------|-------------------|---------------------|
| reg_num/cpu_act.csv                             | built-in     | 21->16            | 5.104869            |
| reg_num/cpu_act.csv                             | A-R                 | 21->21            | 5.137032            |
| reg_num/cpu_act.csv                             | **A/(R+1)**         | **21->21**        | **5.049887**        |
| reg_num/pol.csv                                 | built-in     | 26->15            | 0.271039            |
| reg_num/pol.csv                                 | A-R                 | 26->26            | 0.272401            |
| reg_num/pol.csv                                 | **A/(R+1)**         | **26->21**        | **0.257804**        |
| reg_num/elevators.csv                           | built-in     | 16->15            | 4.324409            |
| reg_num/elevators.csv                           | A-R                 | 16->15            | 4.382167            |
| reg_num/elevators.csv                           | **A/(R+1)**         | **16->16**        | **4.274312**        |
| reg_num/wine_quality.csv                        | built-in     | 11->11            | 0.433982            |
| reg_num/wine_quality.csv                        | A-R                 | 11->9             | 0.439486            |
| reg_num/wine_quality.csv                        | **A/(R+1)**         | **11->11**        | **0.431356**        |
| reg_num/Ailerons.csv                            | **built-in** | **33->28**        | **2.42977**         |
| reg_num/Ailerons.csv                            | A-R                 | 33->28            | 2.44583             |
| reg_num/Ailerons.csv                            | A/(R+1)             | 33->24            | 2.452458            |
| reg_num/yprop_4_1.csv                           | built-in     | 42->29            | 75892.248828        |
| reg_num/yprop_4_1.csv                           | **A-R**             | **42->34**        | **75584.5429**      |
| reg_num/yprop_4_1.csv                           | A/(R+1)             | 42->21            | 76843.555154        |
| reg_num/superconduct.csv                        | **built-in** | **79->67**        | **57399.093773**    |
| reg_num/superconduct.csv                        | A-R                 | 79->77            | 57584.094315        |
| reg_num/superconduct.csv                        | A/(R+1)             | 79->79            | 57573.828608        |
| reg_cat/topo_2_1.csv                            | **built-in** | **255->176**      | **75530.315952**    |
| reg_cat/topo_2_1.csv                            | A-R                 | 255->243          | 76398.785336        |
| reg_cat/topo_2_1.csv                            | A/(R+1)             | 255->148          | 76156.769251        |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | built-in     | 359->338          | 189052.696626       |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | **A-R**             | **359->8**        | **174809.329095**   |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | A/(R+1)             | 359->335          | 189991.04453        |
| reg_cat/house_sales.csv                         | built-in     | 17->16            | 91482.325955        |
| reg_cat/house_sales.csv                         | A-R                 | 17->16            | 91295.138498        |
| reg_cat/house_sales.csv                         | **A/(R+1)**         | **17->16**        | **90574.74304**     |
| reg_cat/nyc-taxi-green-dec-2016.csv             | **built-in** | **16->12**        | **12548.48941**     |
| reg_cat/nyc-taxi-green-dec-2016.csv             | A-R                 | 16->16            | 12596.921199        |
| reg_cat/nyc-taxi-green-dec-2016.csv             | A/(R+1)             | 16->16            | 12588.86233         |
| reg_cat/Allstate_Claims_Severity.csv            | built-in     | 124->106          | 905710139.82245     |
| reg_cat/Allstate_Claims_Severity.csv            | **A-R**             | **124->124**      | **905476424.70167** |
| reg_cat/Allstate_Claims_Severity.csv            | A/(R+1)             | 124->89           | 906173532.364731    |
