import argparse
import json
import os
import random
import subprocess
import tempfile
import time
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_product_hw.data.s3 import get_bucket, get_client
from ml_product_hw.metrics import accuracy
from ml_product_hw.train.dataset import download_data, load_split, load_stats
from ml_product_hw.train.models import build_model


def flatten(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            result.update(flatten(value, prefix + key + "."))
        else:
            result[prefix + key] = value
    return result


def get_git_commit() -> str:
    # in docker commit comes from env
    commit = os.environ.get("GIT_COMMIT")
    if commit is None:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        commit = result.stdout.strip()
    return commit


def evaluate(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, ...]], loss_fn: nn.Module
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            total_loss += loss_fn(out, y).item() * len(y)
            y_true.extend(y.tolist())
            y_pred.extend(out.argmax(dim=1).tolist())
    metrics = {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    return metrics, y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model from config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default="data/train")
    args = parser.parse_args()

    cfg: Any = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    local_dir = os.path.join(args.data_dir, cfg.data.s3_path)
    download_data(cfg.data.s3_path, local_dir)
    mean, std = load_stats(local_dir)
    train_x, train_y, classes = load_split(
        local_dir, "train", mean, std, cfg.data.max_train_samples
    )
    val_x, val_y, _ = load_split(local_dir, "val", mean, std, None)
    test_x, test_y, _ = load_split(local_dir, "test", mean, std, None)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=cfg.train.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=512)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=512)

    model = build_model(cfg.model, len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    loss_fn = nn.CrossEntropyLoss()

    mlflow.set_experiment(cfg.experiment_name)
    with tempfile.TemporaryDirectory() as tmp_dir, mlflow.start_run(run_name=cfg.run_name):
        params: Any = OmegaConf.to_container(cfg)
        mlflow.log_params(flatten(params))
        mlflow.set_tag("git_commit", get_git_commit())

        model_path = os.path.join(tmp_dir, "model.pt")
        best_loss = float("inf")
        best_epoch = 0
        start = time.time()
        for epoch in range(1, cfg.train.epochs + 1):
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y)
            train_loss = train_loss / len(train_y)

            val_metrics, _, _ = evaluate(model, val_loader, loss_fn)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for name in val_metrics:
                mlflow.log_metric("val_" + name, val_metrics[name], step=epoch)
            print(
                f"epoch {epoch}: train_loss {train_loss:.4f}, val_loss {val_metrics['loss']:.4f}, "
                f"val_accuracy {val_metrics['accuracy']:.4f}, val_f1 {val_metrics['f1']:.4f}"
            )

            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)

        train_time = time.time() - start
        mlflow.log_metric("train_time_sec", train_time)
        mlflow.log_metric("best_epoch", best_epoch)
        print(f"train time {train_time:.1f} sec, best epoch {best_epoch}")

        model.load_state_dict(torch.load(model_path))
        val_metrics, y_true, y_pred = evaluate(model, val_loader, loss_fn)
        test_metrics, _, _ = evaluate(model, test_loader, loss_fn)
        for name in test_metrics:
            mlflow.log_metric("test_" + name, test_metrics[name])
        print(f"test: {test_metrics}")

        fig, ax = plt.subplots(figsize=(9, 9))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=classes, xticks_rotation=45, ax=ax, colorbar=False
        )
        ax.set_title("confusion matrix (val)")
        mlflow.log_figure(fig, "confusion_matrix_val.png")
        plt.close(fig)

        OmegaConf.save(cfg, os.path.join(tmp_dir, "config.yaml"))
        metrics = {
            "best_epoch": best_epoch,
            "train_time_sec": train_time,
            "val": val_metrics,
            "test": test_metrics,
        }
        with open(os.path.join(tmp_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        client = get_client()
        bucket = get_bucket()
        s3_prefix = f"models/{cfg.experiment_name}/{cfg.run_name}/"
        for name in ["model.pt", "config.yaml", "metrics.json"]:
            mlflow.log_artifact(os.path.join(tmp_dir, name))
            client.upload_file(os.path.join(tmp_dir, name), bucket, s3_prefix + name)
        print(f"model saved to s3://{bucket}/{s3_prefix}")


if __name__ == "__main__":
    main()
