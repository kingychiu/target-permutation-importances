import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from target_permutation_importances import (
    compute,
    compute_permutation_importance_by_division,
    compute_permutation_importance_by_subtraction,
)

from . import tabular_benchmark

# Configurations
seed = 2023
num_actual_runs = 5
num_random_runs = 10
train_ratio = 0.5
valid_ratio = 0.1
test_ratio = 0.4

score_funcs = {
    "binary_classification": [f1_score, True],
}


def run_selection(model_cls, importance_df, score_func=f1_score, higher_is_better=True):
    original_features = importance_df.feature.tolist()
    best_score = float("-inf")
    best_features = None
    for num_drop in range(len(original_features), 0, -1):
        selected_features = original_features[0:num_drop]

        clf = model_cls(random_state=seed)
        clf.fit(X_train[selected_features], y_train)
        val_preds = clf.predict(X_val[selected_features])
        score = score_func(y_val, val_preds, average="macro")
        if score > best_score:
            best_score = score
            best_features = selected_features

    clf = model_cls(random_state=seed)
    clf = clf.fit(X_train[best_features], y_train)
    val_preds = clf.predict(X_val[best_features])
    test_preds = clf.predict(X_test[best_features])

    return len(best_features), best_score, score_func(y_test, test_preds)


reports = []
for model_cls in [RandomForestClassifier]:
    for (
        name,
        task,
        features,
        target,
        df,
    ) in tabular_benchmark.get_classification_tabular_datasets():
        score_func, higher_is_better = score_funcs[task]

        print(f"==== Dataset: {name}, Num. of Features: {len(features)} ====")
        X = df[features]
        y = df[target].to_numpy().reshape(-1)

        # 0.6 / 0.4
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=seed
        )
        # 0.5 / 0.1
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=valid_ratio / (1 - test_ratio),
            random_state=seed,
        )

        # Fit default random forest with X_train and y_train
        clf = model_cls(random_state=seed)
        clf.fit(X_train, y_train)
        importance_df = pd.DataFrame(
            {"feature": clf.feature_names_in_, "importance": clf.feature_importances_}
        ).sort_values("importance", ascending=False, ignore_index=True)

        num_selected, val_score, test_score = run_selection(
            model_cls, importance_df, score_func, higher_is_better
        )
        reports.append(
            {
                "model": model_cls.__name__,
                "dataset": name,
                "task": "binary_classification",
                "importances": "default importances",
                "num_features": len(features),
                "selected_num": num_selected,
                "score": score_func.__name__,
                "val_score": val_score,
                "test_score": test_score,
            }
        )
        print(reports[-1])

        compute_variants = [
            ("actual - random", compute_permutation_importance_by_subtraction),
            ("actual / random", compute_permutation_importance_by_division),
        ]

        for variant_name, func in compute_variants:
            importance_df = compute(
                model_cls=RandomForestClassifier,
                model_cls_params={},
                model_fit_params={},
                X=X_train,
                y=y_train,
                num_actual_runs=num_actual_runs,
                num_random_runs=num_random_runs,
                permutation_importance_calculator=func,
            ).sort_values("permutation_importance", ascending=False, ignore_index=True)

            num_selected, val_score, test_score = run_selection(
                model_cls, importance_df, score_func, higher_is_better
            )
            reports.append(
                {
                    "model": model_cls.__name__,
                    "dataset": name,
                    "task": "binary_classification",
                    "importances": variant_name,
                    "num_features": len(features),
                    "selected_num": num_selected,
                    "score": score_func.__name__,
                    "val_score": val_score,
                    "test_score": test_score,
                }
            )
            print(reports[-1])
        pd.DataFrame(reports).to_csv("./benchmarks/results/results.csv", index=False)

pd.DataFrame(reports).to_csv("./benchmarks/results/results.csv", index=False)
