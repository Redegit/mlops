from typing import Literal

import pandas as pd

RoomsEncoding = Literal["legacy", "numeric", "categorical"]


def make_features(
    data: pd.DataFrame,
    rooms_encoding: RoomsEncoding = "legacy",
    include_floor: bool = False,
) -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "total_meters": data["area"].astype(float),
            "floors_count": data["total_floors"],
        }
    )
    if rooms_encoding == "numeric":
        features["rooms"] = data["rooms"]
    else:
        categories = range(4) if rooms_encoding == "categorical" else range(1, 4)
        for rooms in categories:
            features[f"rooms_{rooms}"] = data["rooms"] == rooms
        if rooms_encoding == "categorical":
            features["rooms_4plus"] = data["rooms"] >= 4
    if include_floor:
        features["floor"] = data["floor"]
    features["first_floor"] = data["floor"] == 1
    features["last_floor"] = data["floor"] == data["total_floors"]
    return features
