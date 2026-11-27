import argparse
import json
from pathlib import Path
from typing import TypedDict, cast

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.entities import Run
from numpy.typing import NDArray

from service.features import RoomsEncoding, make_features
from train import (
    TRACKING_URI,
    Metrics,
    TrainingData,
    evaluate,
    load_data,
)


class TestResult(TypedDict):
    run_id: str
    model_uri: str
    n_test: int
    test: Metrics


def get_model_uri(run: Run) -> str:
    if (
        run.info.status != "FINISHED"
        or run.outputs is None
        or len(run.outputs.model_outputs) != 1
    ):
        raise ValueError("Нужен завершённый запуск с одной сохранённой моделью")
    return f"models:/{run.outputs.model_outputs[0].model_id}"


def prepare_test_features(run: Run, data: TrainingData) -> pd.DataFrame:
    if (
        run.data.tags["data_sha256"] != data.data_sha256
        or run.data.tags["split_sha256"] != data.split_sha256
    ):
        raise ValueError("Версия данных или разбиение не совпадает с запуском обучения")
    encoding = cast(RoomsEncoding, run.data.params.get("rooms_encoding", "legacy"))
    include_floor = run.data.params.get("include_floor", "False") == "True"
    features = make_features(data.test, encoding, include_floor)
    if ",".join(features.columns) != run.data.params["features"]:
        raise ValueError("Признаки не совпадают с запуском обучения")
    return features


def save_test_result(
    result: TestResult,
    test_data: pd.DataFrame,
    predictions: NDArray[np.float64],
) -> None:
    output = Path("runs") / result["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    errors = test_data[["id", "price"]].copy()
    errors["prediction"] = predictions
    errors["abs_error"] = np.abs(errors["price"] - predictions)
    errors.to_csv(output / "test_predictions.csv", index=False)
    (output / "test_result.json").write_text(json.dumps(result, indent=2) + "\n")
    with mlflow.start_run(run_id=result["run_id"]):
        mlflow.log_metrics(
            {
                f"test_{key}": result["test"][key]
                for key in ("mae", "rmse", "r2", "mape_pct")
            }
        )
        mlflow.log_param("n_test", result["n_test"])
        for path in (output / "test_predictions.csv", output / "test_result.json"):
            mlflow.log_artifact(str(path))
        mlflow.log_artifact("evaluate.py", artifact_path="evaluation_code")
        mlflow.log_artifact("train.py", artifact_path="evaluation_code")
        mlflow.log_artifact("service/features.py", artifact_path="evaluation_code")


def evaluate_run(run_id: str) -> TestResult:
    mlflow.set_tracking_uri(TRACKING_URI)
    run = mlflow.get_run(run_id)
    model_uri = get_model_uri(run)
    data = load_data()
    features = prepare_test_features(run, data)
    model = mlflow.sklearn.load_model(model_uri)
    predictions = model.predict(features)
    result: TestResult = {
        "run_id": run_id,
        "model_uri": model_uri,
        "n_test": len(data.test),
        "test": evaluate(data.test["price"], predictions),
    }
    save_test_result(result, data.test, predictions)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_run(args.run_id), indent=2))


if __name__ == "__main__":
    main()
