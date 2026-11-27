"""Загрузка моделей и параметров подготовки признаков для сервиса."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from numpy.typing import NDArray

try:
    from .features import RoomsEncoding
except ImportError:
    from features import RoomsEncoding

TRACKING_URI = "sqlite:///" + str(Path(".mlflow/mlflow.db").resolve())
EXPERIMENT_NAME = "domclick_price"
PredictionFunction = Callable[[pd.DataFrame], NDArray[np.float64] | list[int]]


@dataclass
class LoadedPredictor:
    predict: PredictionFunction
    rooms_encoding: RoomsEncoding = "legacy"
    include_floor: bool = False
    model_uri: str | None = None
    name: str = ""


def baseline_const(data: pd.DataFrame) -> list[int]:
    return [10_000_000 for _ in range(len(data))]


def lin_reg(data: pd.DataFrame) -> list[int]:
    raise NotImplementedError("lin_reg ещё не обучена")


def load_mlflow_model(uri: str) -> LoadedPredictor:
    mlflow.set_tracking_uri(TRACKING_URI)
    info = mlflow.models.get_model_info(uri)
    if info.run_id is None:
        raise ValueError("У модели нет исходного запуска обучения")
    params = mlflow.get_run(info.run_id).data.params
    model = mlflow.sklearn.load_model(uri)
    return LoadedPredictor(
        predict=model.predict,
        rooms_encoding=cast(RoomsEncoding, params.get("rooms_encoding", "legacy")),
        include_floor=params.get("include_floor", "False") == "True",
        model_uri=uri,
    )


def find_model_uri(filter_string: str) -> str:
    client = MlflowClient(tracking_uri=TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Не найден эксперимент {EXPERIMENT_NAME}")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"attributes.status = 'FINISHED' AND {filter_string}",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"Не найден запуск: {filter_string}")
    outputs = runs[0].outputs
    if outputs is None or len(outputs.model_outputs) != 1:
        raise ValueError("Выбранный запуск должен содержать одну сохранённую модель")
    return f"models:/{outputs.model_outputs[0].model_id}"


def load_baseline() -> LoadedPredictor:
    return load_mlflow_model(find_model_uri("params.model = 'baseline'"))


def load_lin_reg() -> LoadedPredictor:
    return LoadedPredictor(lin_reg)


def load_champion() -> LoadedPredictor:
    client = MlflowClient(tracking_uri=TRACKING_URI)
    versions = client.search_model_versions("tags.Champion = 'True'", max_results=2)
    if not versions:
        raise ValueError("В Model Registry нет версии с тегом Champion=True")
    if len(versions) > 1:
        raise ValueError("Тег Champion=True должен быть только у одной версии модели")
    version = versions[0]
    return load_mlflow_model(f"models:/{version.name}/{version.version}")


def load_baseline_const() -> LoadedPredictor:
    return LoadedPredictor(baseline_const)


def load_pickle_model(name: str) -> LoadedPredictor:
    return LoadedPredictor(joblib.load(name).predict)


MODEL_LOADERS: dict[str, Callable[[], LoadedPredictor]] = {
    "baseline": load_baseline,
    "lin_reg": load_lin_reg,
    "champion": load_champion,
    "baseline_const": load_baseline_const,
}


def load_predictor(name: str) -> LoadedPredictor:
    if handler := MODEL_LOADERS.get(name):
        predictor = handler()
    elif name.startswith(("models:/", "runs:/")):
        predictor = load_mlflow_model(name)
    else:
        predictor = load_pickle_model(name)
    predictor.name = name
    return predictor
