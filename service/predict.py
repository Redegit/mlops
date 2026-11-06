"""Модели предсказания стоимости квартиры.

Каждая модель — функция, которая принимает матрицу признаков X
(список строк в порядке FEATURES) и возвращает последовательность цен.

Чтобы добавить новую модель, достаточно объявить функцию и зарегистрировать
её в MODELS — она сразу станет доступна через `--model` и `MODEL_NAME`.
"""

FEATURES = (
    "total_meters",
    "floors_count",
    "rooms_1",
    "rooms_2",
    "rooms_3",
    "first_floor",
    "last_floor",
)


def baseline_const(X):
    return [10_000_000 for _ in X]


def lin_reg(X):
    raise NotImplementedError("lin_reg ещё не обучена")


MODELS = {
    "baseline_const": baseline_const,
    "lin_reg": lin_reg,
}


def predict(model_name, X):
    return MODELS[model_name](X)