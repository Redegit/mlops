import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.split_data import DATA_PATH
from service.features import RoomsEncoding, make_features
from service.predict import EXPERIMENT_NAME, TRACKING_URI

ModelName = Literal["baseline", "linear", "ridge"]
Model = DummyRegressor | TransformedTargetRegressor


@dataclass
class TrainingData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    data_sha256: str
    split_sha256: str


class Metrics(TypedDict):
    mae: float
    rmse: float
    r2: float
    mape_pct: float


def load_data() -> TrainingData:
    data_sha = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    metadata = json.loads(Path("runs/split.json").read_text())
    if data_sha != metadata["data_sha256"]:
        raise ValueError("Данные изменились. Повторите python scripts/split_data.py")
    data = pd.read_csv(DATA_PATH)
    split_path = Path("runs/split.csv")
    split = pd.read_csv(split_path)
    if (
        split["id"].duplicated().any()
        or set(split["id"]) != set(data["id"])
        or set(split["split"]) != {"train", "validation", "test"}
    ):
        raise ValueError("Разбиение должно содержать каждый ID ровно один раз")
    data = data.merge(split, on="id", validate="one_to_one")
    return TrainingData(
        train=data[data["split"] == "train"],
        validation=data[data["split"] == "validation"],
        test=data[data["split"] == "test"],
        data_sha256=data_sha,
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
    )


def build_model(name: ModelName, alpha: float) -> Model:
    if name == "baseline":
        return DummyRegressor(strategy="median")
    regressor = LinearRegression() if name == "linear" else Ridge(alpha=alpha)
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), regressor),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def evaluate(y_true: pd.Series, y_pred: NDArray[np.float64]) -> Metrics:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def train(
    name: ModelName,
    alpha: float,
    rooms_encoding: RoomsEncoding = "legacy",
    include_floor: bool = False,
) -> None:
    data = load_data()
    X_train = make_features(data.train, rooms_encoding, include_floor)
    X_validation = make_features(data.validation, rooms_encoding, include_floor)
    model = build_model(name, alpha)

    Path(".mlflow").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=Path(".mlflow/artifacts").resolve().as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"ridge_alpha={alpha:g}" if name == "ridge" else name
    if rooms_encoding != "legacy":
        run_name += f"_rooms={rooms_encoding}"
    if include_floor:
        run_name += "_floor=numeric"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "model": name,
                "target": "price_rub",
                "log_target": name != "baseline",
                "n_train": len(data.train),
                "n_validation": len(data.validation),
                "features": ",".join(X_train.columns),
                "rooms_encoding": rooms_encoding,
                "include_floor": include_floor,
            }
        )
        if name == "ridge":
            mlflow.log_param("alpha", alpha)
        mlflow.set_tags(
            {
                "data_sha256": data.data_sha256,
                "split_sha256": data.split_sha256,
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
                "git_dirty": bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"],
                        text=True,
                    ).strip()
                ),
            }
        )
        for path in (
            "runs/split.csv",
            "runs/split.json",
            "dvc.lock",
            "data/raw.dvc",
            "train.py",
            "scripts/split_data.py",
            "service/features.py",
            "service/predict.py",
        ):
            mlflow.log_artifact(path)
        model.fit(X_train, data.train["price"])
        if isinstance(model, DummyRegressor):
            mlflow.log_param("median_price_rub", float(model.constant_[0, 0]))
        predictions = model.predict(X_validation)
        metrics = evaluate(data.validation["price"], predictions)
        mlflow.log_metrics(
            {
                f"validation_{key}": metrics[key]
                for key in ("mae", "rmse", "r2", "mape_pct")
            }
        )
        saved = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_validation.head(3),
        )
        loaded = mlflow.sklearn.load_model(saved.model_uri)
        np.testing.assert_allclose(loaded.predict(X_validation), predictions)
        mlflow.set_tag("reload_verified", True)

        output = Path("runs") / run.info.run_id
        output.mkdir()
        errors = data.validation[["id", "price"]].copy()
        errors["prediction"] = predictions
        errors["abs_error"] = np.abs(errors["price"] - predictions)
        errors.to_csv(output / "validation_predictions.csv", index=False)
        mlflow.log_artifact(str(output / "validation_predictions.csv"))
        result = {
            "run_id": run.info.run_id,
            "model_uri": saved.model_uri,
            "validation": metrics,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=("baseline", "linear", "ridge")
    )
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument(
        "--rooms-encoding",
        choices=("legacy", "numeric", "categorical"),
        default="legacy",
    )
    parser.add_argument("--include-floor", action="store_true")
    args = parser.parse_args()
    train(args.model, args.alpha, args.rooms_encoding, args.include_floor)


if __name__ == "__main__":
    main()
