import argparse
import json
import os
from typing import Any

from sklearn.model_selection import train_test_split


def load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
    return data


def save_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified train/val split")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, "annotations", "train.json")
    val_path = os.path.join(args.data_dir, "annotations", "val.json")

    data = load_json(train_path)
    classes = data["classes"]
    images = data["images"]
    # if split was already done, put val back so we always split full train
    if os.path.exists(val_path):
        images = images + load_json(val_path)["images"]
    images = sorted(images, key=lambda x: x["image_id"])

    labels = []
    for image in images:
        labels.append(image["label"])

    train_images, val_images = train_test_split(
        images, test_size=args.val_size, random_state=args.seed, stratify=labels
    )
    train_images = sorted(train_images, key=lambda x: x["image_id"])
    val_images = sorted(val_images, key=lambda x: x["image_id"])

    save_json(train_path, {"classes": classes, "images": train_images})
    save_json(val_path, {"classes": classes, "images": val_images})
    save_json(
        os.path.join(args.data_dir, "split_params.json"),
        {"val_size": args.val_size, "seed": args.seed},
    )
    print(f"train: {len(train_images)}, val: {len(val_images)}")


if __name__ == "__main__":
    main()
