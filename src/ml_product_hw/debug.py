from pathlib import Path

import joblib

from ml_product_hw.metrics import accuracy


def evaluate(y_true: list[int], y_pred: list[int], path: str) -> float:
    score = accuracy(y_true, y_pred)
    Path(path).parent.mkdir(exist_ok=True)
    joblib.dump({"accuracy": score}, path)
    return score


if __name__ == "__main__":
    print(evaluate([1, 0, 1, 1], [1, 1, 1, 0], "models/scores.joblib"))
