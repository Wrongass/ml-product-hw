import argparse
import json
import os

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute mean and std per channel on train")
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "annotations", "train.json")) as f:
        images = json.load(f)["images"]

    # keep only sums, not all images in memory
    total = np.zeros(3)
    total_sq = np.zeros(3)
    count = 0
    for image in images:
        pixels = np.asarray(Image.open(os.path.join(args.data_dir, image["file_name"])))
        pixels = pixels.reshape(-1, 3).astype(np.float64) / 255.0
        total += pixels.sum(axis=0)
        total_sq += (pixels**2).sum(axis=0)
        count += pixels.shape[0]

    mean = total / count
    std = np.sqrt(total_sq / count - mean**2)

    stats = {
        "split": "train",
        "num_images": len(images),
        "scale": "0-1 (pixels / 255)",
        "channels": ["R", "G", "B"],
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
    }
    with open(os.path.join(args.data_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"mean: {stats['mean']}")
    print(f"std: {stats['std']}")


if __name__ == "__main__":
    main()
