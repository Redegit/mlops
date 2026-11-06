import argparse
from flask import Flask, render_template, request
from logging.config import dictConfig

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

import dotenv
config = {
    **dotenv.dotenv_values('.env.example'),
    **{k: v for k, v in dotenv.dotenv_values('.env').items() if v is not None},
}

from flask_cors import CORS
cors = CORS(app, resources={r"/api/*": {"origins": config.get("ORIGIN", "http://localhost:5000")}})

from flask_httpauth import HTTPTokenAuth
auth = HTTPTokenAuth(scheme='Bearer')
tokens = { config['APP_TOKEN']: "user1", }

@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]

import joblib

try:
    from . import predict as prediction_models
except ImportError:
    import predict as prediction_models

# Модель по умолчанию задаётся в .env
MODEL_NAME = config.get("MODEL_NAME", "baseline_const")


def load_predictor(name):
    """Возвращает predict(X) по имени модели из predict.py или пути к .pkl."""
    if name in prediction_models.MODELS:
        return prediction_models.MODELS[name]
    return joblib.load(name).predict


app.config["predict"] = load_predictor(MODEL_NAME)


# Маршрут для отображения формы
@app.route("/")
def index():
    return render_template("index.html")


# Маршрут для обработки данных формы
@app.route("/api/numbers", methods=["POST"])
@auth.login_required
def process_numbers():

    data = request.get_json()

    app.logger.info(f"Requst data: {data}")
    try:
        total_meters = float(data["area"])
        floors_count = int(data["total_floors"])
        rooms_1 = int(data["rooms"]) == 1
        rooms_2 = int(data["rooms"]) == 2
        rooms_3 = int(data["rooms"]) == 3
        first_floor = int(data["floor"]) == 1
        last_floor = int(data["floor"]) == floors_count
    except ValueError:
        return {"status": "error", "data": "Ошибка парсинга данных"}

    features = {
        "total_meters": total_meters,
        "floors_count": floors_count,
        "rooms_1": rooms_1,
        "rooms_2": rooms_2,
        "rooms_3": rooms_3,
        "first_floor": first_floor,
        "last_floor": last_floor,
    }
    row = [features[name] for name in prediction_models.FEATURES]
    try:
        price = int(app.config["predict"]([row])[0])
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
        help="Имя модели из service/predict.py или путь к сохранённой .pkl модели",
        default=MODEL_NAME,
    )
    args = parser.parse_args()

    app.config["predict"] = load_predictor(args.model)
    app.logger.info(f"Use model: {args.model}")
    app.run(debug=True)
