import pandas as pd

df = pd.read_csv("./benchmarks/results/tabular_benchmark.csv")


def _h(_df: pd.DataFrame) -> bool:
    # For each model, dataset group
    # Check is the null importances method is the best
    for imp in _df[_df["is_best"]]["importances"].tolist():
        if "built-in" in imp:
            return False
    return True


stat_df = df.groupby(["model", "dataset"]).apply(_h)
stat_df = (
    stat_df.groupby(["model"])
    .agg(["count", "sum", "mean"])
    .rename(columns={"count": "n_dataset", "sum": "n_better", "mean": "better %"})
)
stat_df["better %"] = (100 * stat_df["better %"]).round(2)

stat_df.to_csv("./benchmarks/results/stat.csv", index=True)

stat_df = (
    df[df["is_best"]]
    .groupby(["task"])["importances"]
    .value_counts(normalize=True)
    .reset_index(name="the best %")
)
stat_df["the best %"] = (100 * stat_df["the best %"]).round(2)
stat_df.to_csv("./benchmarks/results/task_stat.csv", index=False)

df["feature_reduction"] = (
    df["num_features"].astype(str) + "->" + df["selected_num"].astype(str)
)
df["% Change from baseline"] = df["% Change from baseline"].astype(str) + "%"
for col in ["importances", "test_score", "% Change from baseline", "feature_reduction"]:
    df.loc[df["is_best"], col] = df.loc[df["is_best"], col].apply(lambda x: f"**{x}**")


models = df["model"].unique()

for model in models:
    _df = df[df["model"] == model][
        [
            "dataset",
            "importances",
            "feature_reduction",
            "test_score",
        ]
    ]
    _df.to_csv(
        f"./benchmarks/results/processed_{model}_tabular_benchmark.csv",
        index=False,
    )
