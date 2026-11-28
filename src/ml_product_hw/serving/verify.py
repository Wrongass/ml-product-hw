import argparse
import json
import math
import os
import sys
import tempfile
from typing import Any

import requests
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

from ml_product_hw.data.s3 import get_bucket, get_client
from ml_product_hw.metrics import accuracy
from ml_product_hw.train.dataset import download_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Check model metrics through the service")
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--url", default="http://localhost:8081")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--loss-tol", type=float, default=0.01)
    parser.add_argument("--accuracy-tol", type=float, default=0.002)
    parser.add_argument("--f1-tol", type=float, default=0.002)
    args = parser.parse_args()

    cfg: Any = OmegaConf.load(args.config)
    local_dir = os.path.join(args.data_dir, cfg.data_s3_path)
    download_data(cfg.data_s3_path, local_dir)
    with open(os.path.join(local_dir, "annotations", "test.json")) as f:
        images = json.load(f)["images"]

    total_loss = 0.0
    y_true = []
    y_pred = []
    session = requests.Session()
    for i, image in enumerate(images):
        with open(os.path.join(local_dir, image["file_name"]), "rb") as f:
            response = session.post(args.url + "/predict", files={"file": f})
        response.raise_for_status()
        probs = response.json()["probabilities"]
        total_loss += -math.log(max(probs[image["label"]], 1e-12))
        y_true.append(image["label"])
        y_pred.append(response.json()["class_id"])
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(images)} images")

    service = {
        "loss": total_loss / len(images),
        "accuracy": accuracy(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    tolerances = {"loss": args.loss_tol, "accuracy": args.accuracy_tol, "f1": args.f1_tol}

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "metrics.json")
        get_client().download_file(get_bucket(), cfg.model_s3_path + "/metrics.json", path)
        with open(path) as f:
            train = json.load(f)["test"]

    failed = False
    print(f"{'metric':<10} {'train':>8} {'service':>8} {'diff':>8} {'tol':>8}")
    for name in ["loss", "accuracy", "f1"]:
        diff = service[name] - train[name]
        status = "ok"
        if abs(diff) > tolerances[name]:
            status = "FAIL"
            failed = True
        print(
            f"{name:<10} {train[name]:>8.4f} {service[name]:>8.4f} {diff:>+8.4f} "
            f"{tolerances[name]:>8.4f} {status}"
        )

    if failed:
        print("metrics in service differ from training")
        sys.exit(1)
    print("metrics match")


if __name__ == "__main__":
    main()
