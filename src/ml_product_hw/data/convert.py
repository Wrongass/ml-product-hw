import argparse
import hashlib
import io
import json
import os
import pickle
import tarfile
from typing import Any

from PIL import Image

TRAIN_BATCHES = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
TEST_BATCHES = ["test_batch"]


def load_batch(tar: tarfile.TarFile, name: str) -> Any:
    f = tar.extractfile("cifar-10-batches-py/" + name)
    assert f is not None
    return pickle.load(f, encoding="bytes")


def convert_split(
    tar: tarfile.TarFile, prefix: str, batch_names: list[str], classes: list[str], out_dir: str
) -> list[dict[str, Any]]:
    images = []
    index = 0
    for name in batch_names:
        batch = load_batch(tar, name)
        pixels = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = batch[b"labels"]
        for i in range(len(labels)):
            image_id = f"{prefix}_{index:05d}"
            file_name = "images/" + image_id + ".png"

            buf = io.BytesIO()
            Image.fromarray(pixels[i]).save(buf, format="PNG")
            png = buf.getvalue()
            with open(os.path.join(out_dir, file_name), "wb") as f:
                f.write(png)

            label = int(labels[i])
            images.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "label": label,
                    "label_name": classes[label],
                    "sha256": hashlib.sha256(png).hexdigest(),
                }
            )
            index += 1
        print(f"{name}: done, {index} images")
    return images


def save_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CIFAR-10 pickle to png + json")
    parser.add_argument("--archive", default="data/raw/cifar-10-python.tar.gz")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "annotations"), exist_ok=True)

    with tarfile.open(args.archive, "r:gz") as tar:
        meta = load_batch(tar, "batches.meta")
        classes = []
        for name in meta[b"label_names"]:
            classes.append(name.decode())

        train = convert_split(tar, "train", TRAIN_BATCHES, classes, args.out_dir)
        test = convert_split(tar, "test", TEST_BATCHES, classes, args.out_dir)

    save_json(
        os.path.join(args.out_dir, "annotations", "train.json"),
        {"classes": classes, "images": train},
    )
    save_json(
        os.path.join(args.out_dir, "annotations", "test.json"),
        {"classes": classes, "images": test},
    )
    # train.json is full again, so old val.json is not needed
    val_path = os.path.join(args.out_dir, "annotations", "val.json")
    if os.path.exists(val_path):
        os.remove(val_path)
    print(f"train: {len(train)}, test: {len(test)}")


if __name__ == "__main__":
    main()
