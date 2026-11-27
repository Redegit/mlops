from pathlib import Path
from typing import TypedDict, cast

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import SpanType

try:
    from .features import ListingInput, make_features
    from .predict import TRACKING_URI, LoadedPredictor
except ImportError:
    from features import ListingInput, make_features
    from predict import TRACKING_URI, LoadedPredictor

EXPERIMENT_NAME = "domclick_predictions"


class PredictionOutput(TypedDict):
    price: int


class PredictionRecord(TypedDict):
    trace_id: str
    timestamp_ms: int
    model: str
    model_uri: str | None
    inputs: ListingInput
    price: int | None
    duration_ms: int | None
    status: str
    error: str | None


def setup_monitoring() -> None:
    Path(".mlflow").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=Path(".mlflow/artifacts/predictions").resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)


def predict_price(listing: ListingInput, predictor: LoadedPredictor) -> int:
    with mlflow.start_span(name="prediction", span_type=SpanType.TASK) as span:
        span.set_inputs(listing)
        span.set_attributes({"model": predictor.name, "model_uri": predictor.model_uri})
        try:
            features = make_features(
                pd.DataFrame([listing]),
                predictor.rooms_encoding,
                predictor.include_floor,
            )
            price = int(predictor.predict(features)[0])
        except Exception as error:
            span.set_attribute("error", str(error))
            raise
        span.set_outputs({"price": price})
        return price


def get_recent_predictions(limit: int) -> list[PredictionRecord]:
    client = MlflowClient(tracking_uri=TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return []
    traces = client.search_traces(
        locations=[experiment.experiment_id],
        max_results=limit,
        order_by=["timestamp_ms DESC"],
        include_spans=True,
    )
    records: list[PredictionRecord] = []
    for trace in traces:
        root = next((span for span in trace.data.spans if span.parent_id is None), None)
        if root is None:
            continue
        output = cast(PredictionOutput | None, root.outputs)
        records.append(
            {
                "trace_id": trace.info.trace_id,
                "timestamp_ms": trace.info.request_time,
                "model": cast(str, root.get_attribute("model")),
                "model_uri": cast(str | None, root.get_attribute("model_uri")),
                "inputs": cast(ListingInput, root.inputs),
                "price": output["price"] if output else None,
                "duration_ms": trace.info.execution_duration,
                "status": trace.info.state.value,
                "error": cast(str | None, root.get_attribute("error")),
            }
        )
    return records
