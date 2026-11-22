import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from ml_product_hw.data.s3 import get_bucket, get_client


def download_data(s3_path: str, local_dir: str) -> None:
    if os.path.exists(local_dir):
        print(f"{local_dir} already exists, skip download")
        return

    client = get_client()
    bucket = get_bucket()

    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_path + "/"):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    os.makedirs(os.path.join(local_dir, "images"))
    os.makedirs(os.path.join(local_dir, "annotations"))
    print(f"downloading {len(keys)} files from s3://{bucket}/{s3_path}")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for key in keys:
            path = os.path.join(local_dir, key[len(s3_path) + 1 :])
            futures.append(executor.submit(client.download_file, bucket, key, path))
        for future in futures:
            future.result()
    print("done")


def load_stats(local_dir: str) -> tuple[list[float], list[float]]:
    with open(os.path.join(local_dir, "stats.json")) as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]


def load_split(
    local_dir: str, split: str, mean: list[float], std: list[float], max_samples: int | None
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    with open(os.path.join(local_dir, "annotations", split + ".json")) as f:
        data = json.load(f)
    images = data["images"]
    if max_samples is not None:
        images = images[:max_samples]

    pixels = []
    labels = []
    for image in images:
        pixels.append(np.asarray(Image.open(os.path.join(local_dir, image["file_name"]))))
        labels.append(image["label"])

    x = np.stack(pixels).astype(np.float32) / 255.0
    x_norm = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    # channels first for torch
    x_tensor = torch.from_numpy(x_norm).permute(0, 3, 1, 2).contiguous()
    y_tensor = torch.tensor(labels)
    print(f"{split}: {len(labels)} images")
    return x_tensor, y_tensor, data["classes"]
