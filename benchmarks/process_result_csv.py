import pandas as pd

df = pd.read_csv("./benchmarks/results/random_forest_tabular_benchmark.csv")
df["feature_reduction"] = (
    df["num_features"].astype(str) + "->" + df["selected_num"].astype(str)
)
df["% Change from baseline"] = df["% Change from baseline"].astype(str) + "%"
for col in ["importances", "test_score", "% Change from baseline"]:
    df.loc[df["is_best"], col] = df.loc[df["is_best"], col].apply(lambda x: f"**{x}**")

df[
    [
        "dataset",
        "task",
        "importances",
        "feature_reduction",
        "test_score",
        "% Change from baseline",
    ]
].to_csv(
    "./benchmarks/results/random_forest_tabular_benchmark_processed.csv", index=False
)
