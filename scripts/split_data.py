import hashlib
import json
from pathlib import Path
from typing import TypedDict

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/listings.csv")
SEED = 42


class DataSplit(TypedDict):
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_data(data: pd.DataFrame) -> DataSplit:
    data = data.sort_values("id").reset_index(drop=True)
    train, remainder = train_test_split(data, test_size=0.4, random_state=SEED)
    validation, test = train_test_split(remainder, test_size=0.5, random_state=SEED)
    return {"train": train, "validation": validation, "test": test}


def main() -> None:
    parts = split_data(pd.read_csv(DATA_PATH))
    partitions = (
        ("train", parts["train"]),
        ("validation", parts["validation"]),
        ("test", parts["test"]),
    )
    output = Path("runs")
    output.mkdir(exist_ok=True)
    pd.concat([part[["id"]].assign(split=name) for name, part in partitions]).to_csv(
        output / "split.csv", index=False
    )
    summary = {
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "random_state": SEED,
        "rows": {name: len(part) for name, part in partitions},
    }
    (output / "split.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
