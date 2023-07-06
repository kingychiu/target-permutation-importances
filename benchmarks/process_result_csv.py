import pandas as pd

df = pd.read_csv("./benchmarks/results/tabular_benchmark.csv")
df["feature_reduction"] = (
    df["num_features"].astype(str) + "->" + df["selected_num"].astype(str)
)
df["% Change from baseline"] = df["% Change from baseline"].astype(str) + "%"
for col in ["importances", "test_score", "% Change from baseline", "feature_reduction"]:
    df.loc[df["is_best"], col] = df.loc[df["is_best"], col].apply(lambda x: f"**{x}**")

models = df["model"].unique()

for model in models:
    df[df["model"] == model][
        [
            "dataset",
            "importances",
            "feature_reduction",
            "test_score",
        ]
    ].to_csv(
        f"./benchmarks/results/processed_{model}_tabular_benchmark.csv",
        index=False,
    )
