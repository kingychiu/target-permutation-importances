import os
from typing import List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from target_permutation_importances import (
    compute,
    compute_permutation_importance_by_division,
    compute_permutation_importance_by_subtraction,
)

from . import tabular_benchmark

# Configurations
seed = 2023
num_actual_runs = 5
num_random_runs = 50
train_ratio = 0.5
valid_ratio = 0.1
test_ratio = 0.4

score_funcs = {
    "binary_classification": [f1_score, True],
    "regression": [mean_squared_error, False],
}
result_path = "./benchmarks/results/xgboost_tabular_benchmark.csv"


def run_selection(
    model_cls,
    importance_df,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    score_func=f1_score,
    higher_is_better=True,
):
    original_features = importance_df.feature.tolist()
    best_score = float("-inf") if higher_is_better else float("inf")
    best_features = None
    for num_drop in range(len(original_features), 0, -1):
        selected_features = original_features[0:num_drop]

        clf = model_cls(random_state=seed, n_jobs=-1)
        clf.fit(X_train[selected_features], y_train)
        val_preds = clf.predict(X_val[selected_features])
        score = score_func(y_val, val_preds)

        if (higher_is_better and score > best_score) or (
            not higher_is_better and score < best_score
        ):
            best_score = score
            best_features = selected_features
            print("Update best", best_score, len(selected_features))

    clf = model_cls(random_state=seed)
    clf = clf.fit(X_train[best_features], y_train)
    val_preds = clf.predict(X_val[best_features])
    test_preds = clf.predict(X_test[best_features])

    return len(best_features), best_score, score_func(y_test, test_preds)


def get_model_cls(name: str, task: str):
    if task == "binary_classification":
        if name == "RandomForest":
            return RandomForestClassifier
        elif name == "XGBoost":  # noqa
            return XGBClassifier
        raise NotImplementedError
    elif task == "regression":  # noqa
        if name == "RandomForest":
            return RandomForestRegressor
        elif name == "XGBoost":  # noqa
            return XGBRegressor
        raise NotImplementedError
    raise NotImplementedError


def write_report(reports: List):
    if os.path.exists(result_path):
        existing_df = pd.read_csv(result_path)
    report_df = pd.DataFrame(reports)
    if os.path.exists(result_path):
        report_df = pd.concat([existing_df, report_df])
    report_df = report_df.drop_duplicates(
        subset=["model", "dataset", "importances"], keep="last"
    ).reset_index(drop=True)
    # Update best method for each dataset
    report_df["higher_is_better"] = False
    report_df.loc[report_df["score"] == "f1_score", "higher_is_better"] = True

    report_df["max_score"] = report_df.groupby(["model", "dataset"])[
        "test_score"
    ].transform("max")
    report_df["min_score"] = report_df.groupby(["model", "dataset"])[
        "test_score"
    ].transform("min")
    report_df["base_score"] = report_df.groupby(["model", "dataset"])[
        "test_score"
    ].transform("first")

    report_df.loc[report_df["higher_is_better"], "is_best"] = (
        report_df["test_score"] >= report_df["max_score"]
    )
    report_df.loc[~report_df["higher_is_better"], "is_best"] = (
        report_df["test_score"] <= report_df["min_score"]
    )

    report_df["% Change from baseline"] = (
        100
        * (report_df["test_score"] - report_df["base_score"])
        / (report_df["base_score"])
    ).round(6)
    report_df["val_score"] = report_df["val_score"].round(6)
    report_df["test_score"] = report_df["test_score"].round(6)
    report_df = report_df.drop(
        columns=["max_score", "min_score", "base_score", "higher_is_better"],
    )
    report_df.to_csv(result_path, index=False)


reports = []
for model_name in ["XGBoost"]:
    for (
        name,
        task,
        features,
        target,
        df,
    ) in tabular_benchmark.get_classification_tabular_datasets():
        score_func, higher_is_better = score_funcs[task]
        model_cls = get_model_cls(model_name, task)

        print(f"==== Dataset: {name}, Num. of Features: {len(features)} ====")
        X = df[features]
        y = df[target].to_numpy().reshape(-1)

        # 0.6 / 0.4
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=seed, shuffle=True
        )
        # 0.5 / 0.1
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=valid_ratio / (1 - test_ratio),
            random_state=seed,
            shuffle=True,
        )

        # Fit default random forest with X_train and y_train
        clf = model_cls(random_state=seed, n_jobs=-1)
        clf.fit(X_train, y_train)
        importance_df = pd.DataFrame(
            {
                "feature": clf.feature_names_in_,
                "importance": clf.feature_importances_,
            }
        ).sort_values("importance", ascending=False, ignore_index=True)

        num_selected, val_score, test_score = run_selection(
            model_cls,
            importance_df,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            score_func,
            higher_is_better,
        )
        reports.append(
            {
                "model": model_cls.__name__,
                "dataset": name,
                "task": task,
                "importances": "built-in",
                "num_features": len(features),
                "selected_num": num_selected,
                "score": score_func.__name__,
                "val_score": val_score,
                "test_score": test_score,
            }
        )
        print(reports[-1])
        write_report(reports)
        compute_variants = [
            ("A-R", compute_permutation_importance_by_subtraction),
            ("A/(R+1)", compute_permutation_importance_by_division),
        ]

        for variant_name, func in compute_variants:
            print("X_train shape", X_train.shape, "y_train shape", y_train.shape)
            importance_df = compute(
                model_cls=model_cls,
                model_cls_params={"n_jobs": -1},
                model_fit_params={},
                X=X_train,
                y=y_train,
                num_actual_runs=num_actual_runs,
                num_random_runs=num_random_runs,
                permutation_importance_calculator=func,
            )
            importance_df = importance_df.sort_values(
                "importance", ascending=False, ignore_index=True
            )
            num_selected, val_score, test_score = run_selection(
                model_cls,
                importance_df,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                score_func,
                higher_is_better=higher_is_better,
            )
            reports.append(
                {
                    "model": model_cls.__name__,
                    "dataset": name,
                    "task": task,
                    "importances": variant_name,
                    "num_features": len(features),
                    "selected_num": num_selected,
                    "score": score_func.__name__,
                    "val_score": val_score,
                    "test_score": test_score,
                }
            )
            print(reports[-1])
            write_report(reports)
