import argparse
import os
import shutil
import tempfile
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from omegaconf import OmegaConf

from ml_product_hw.data.s3 import get_bucket, get_client
from ml_product_hw.train.models import build_model

NUM_CLASSES = 10
REPO_DIR = "model_repository"
PBTXT_PATH = "configs/triton/config.pbtxt"
STATS_DIR = "serving_data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to ONNX and build model repository")
    parser.add_argument("--config", default="configs/serving.yaml")
    args = parser.parse_args()

    cfg: Any = OmegaConf.load(args.config)
    client = get_client()
    bucket = get_bucket()

    os.makedirs(STATS_DIR, exist_ok=True)
    stats_path = os.path.join(STATS_DIR, "stats.json")
    client.download_file(bucket, cfg.data_s3_path + "/stats.json", stats_path)
    print(f"downloaded stats.json to {stats_path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for name in ["model.pt", "config.yaml"]:
            client.download_file(
                bucket, cfg.model_s3_path + "/" + name, os.path.join(tmp_dir, name)
            )
        print(f"downloaded model from s3://{bucket}/{cfg.model_s3_path}")

        run_cfg: Any = OmegaConf.load(os.path.join(tmp_dir, "config.yaml"))
        model = build_model(run_cfg.model, NUM_CLASSES)
        model.load_state_dict(torch.load(os.path.join(tmp_dir, "model.pt")))
    model.eval()

    model_dir = os.path.join(REPO_DIR, cfg.model_name)
    onnx_path = os.path.join(model_dir, "1", "model.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    batch = torch.export.Dim("batch", max=64)
    torch.onnx.export(
        model,
        (torch.randn(2, 3, 32, 32),),
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: batch}},
        external_data=False,
    )

    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        torch_logits = model(x).numpy()
    session = ort.InferenceSession(onnx_path)
    onnx_logits = session.run(["logits"], {"input": x.numpy()})[0]
    np.testing.assert_allclose(torch_logits, onnx_logits, rtol=1e-4, atol=1e-5)
    print("pytorch and onnx logits match")

    shutil.copy(PBTXT_PATH, os.path.join(model_dir, "config.pbtxt"))
    print(f"model repository saved to {model_dir}")


if __name__ == "__main__":
    main()
