import argparse
from logging.config import dictConfig
from typing import TypedDict, cast

import dotenv
from flask import Flask, render_template, request
from flask_cors import CORS
from flask_httpauth import HTTPTokenAuth

try:
    from .features import ListingInput
    from .monitoring import (
        PredictionRecord,
        get_recent_predictions,
        predict_price,
        setup_monitoring,
    )
    from .predict import LoadedPredictor, load_predictor
except ImportError:
    from features import ListingInput
    from monitoring import (
        PredictionRecord,
        get_recent_predictions,
        predict_price,
        setup_monitoring,
    )
    from predict import LoadedPredictor, load_predictor


class ApiResponse(TypedDict):
    status: str
    data: str | int


class HistoryResponse(TypedDict):
    predictions: list[PredictionRecord]


dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "service/flask.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)

app = Flask(__name__)

config = {
    **dotenv.dotenv_values(".env.example"),
    **{k: v for k, v in dotenv.dotenv_values(".env").items() if v is not None},
}

cors = CORS(
    app,
    resources={r"/api/*": {"origins": config.get("ORIGIN", "http://localhost:5000")}},
)

auth = HTTPTokenAuth(scheme="Bearer")
tokens = {
    config["APP_TOKEN"]: "user1",
}


@auth.verify_token
def verify_token(token: str) -> str | None:
    return tokens.get(token)


# Модель по умолчанию задаётся в .env
MODEL_NAME = config.get("MODEL_NAME") or "baseline_const"
setup_monitoring()


if __name__ != "__main__":
    app.config["predict"] = load_predictor(MODEL_NAME)


# Маршрут для отображения формы
@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/monitoring")
def monitoring() -> str:
    return render_template("monitoring.html")


@app.route("/api/predictions")
@auth.login_required
def prediction_history() -> HistoryResponse | tuple[ApiResponse, int]:
    try:
        limit = int(request.args.get("limit", "50"))
    except ValueError:
        return {"status": "error", "data": "N должно быть целым числом"}, 400
    if not 1 <= limit <= 200:
        return {"status": "error", "data": "N должно быть от 1 до 200"}, 400
    return {"predictions": get_recent_predictions(limit)}


# Маршрут для обработки данных формы
@app.route("/api/numbers", methods=["POST"])
@auth.login_required
def process_numbers() -> ApiResponse | tuple[ApiResponse, int]:

    data = request.get_json()

    app.logger.info(f"Requst data: {data}")
    try:
        listing: ListingInput = {
            "area": float(data["area"]),
            "rooms": int(data["rooms"]),
            "total_floors": int(data["total_floors"]),
            "floor": int(data["floor"]),
        }
    except ValueError:
        return {"status": "error", "data": "Ошибка парсинга данных"}

    predictor = cast(LoadedPredictor, app.config["predict"])
    try:
        price = predict_price(listing, predictor)
    except NotImplementedError as exc:
        app.logger.error(f"Model error: {exc}")
        return {"status": "error", "data": str(exc)}, 501

    return {"status": "success", "data": price}


if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="baseline, lin_reg, champion, путь к .pkl или MLflow URI models:/...",
        default=MODEL_NAME,
    )
    parser.add_argument(
        "-p",
        "--port",
        help="Порт для запуска сервера",
        default=5000,
    )
    args = parser.parse_args()

    predictor = load_predictor(args.model)
    app.config["predict"] = predictor
    app.logger.info("Use model: %s; URI: %s", args.model, predictor.model_uri)

    port = int(args.port)
    app.run(debug=True, port=port, host="0.0.0.0")
