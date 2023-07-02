from typing import Iterator, Tuple

import pandas as pd
from datasets import load_dataset
from sklearn import preprocessing

dataset_configs = [
    {
        "name": "clf_cat/electricity.csv",
        "task": "binary_classification",
        "target_column": "class",
    },
    {
        "name": "clf_cat/eye_movements.csv",
        "task": "binary_classification",
        "target_column": "label",
    },
    {
        "name": "clf_cat/covertype.csv",
        "task": "binary_classification",
        "target_column": "class",
    },
    {
        "name": "clf_cat/albert.csv",
        "task": "binary_classification",
        "target_column": "class",
    },
    {
        "name": "clf_cat/compas-two-years.csv",
        "task": "binary_classification",
        "target_column": "twoyearrecid",
    },
    {
        "name": "clf_cat/default-of-credit-card-clients.csv",
        "task": "binary_classification",
        "target_column": "y",
    },
    {
        "name": "clf_cat/road-safety.csv",
        "task": "binary_classification",
        "target_column": "SexofDriver",
    },
    {
        "name": "clf_num/Bioresponse.csv",
        "task": "binary_classification",
        "target_column": "target",
    },
    {
        "name": "clf_num/jannis.csv",
        "task": "binary_classification",
        "target_column": "class",
    },
    {
        "name": "clf_num/MiniBooNE.csv",
        "task": "binary_classification",
        "target_column": "signal",
    },
]


def get_tabular_dataset(
    name: str, target_column: str, task: str
) -> Tuple[str, pd.DataFrame]:
    # https://huggingface.co/datasets/inria-soda/tabular-benchmark
    dataset_dict = load_dataset("inria-soda/tabular-benchmark", data_files=name)
    assert list(dataset_dict.keys()) == ["train"]

    dataset = dataset_dict["train"]

    df = dataset.to_pandas()
    assert target_column in df.columns

    le = preprocessing.LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
    features = [col for col in df.columns if col != target_column]

    return name, task, features, [target_column], df


def get_classification_tabular_datasets() -> Iterator[pd.DataFrame]:
    for dataset_config in dataset_configs:
        yield get_tabular_dataset(
            name=dataset_config["name"],
            task=dataset_config["task"],
            target_column=dataset_config["target_column"],
        )


for df in get_classification_tabular_datasets():
    break
