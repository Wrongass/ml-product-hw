import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ml_product_hw.data.s3 import get_bucket, get_client

SPLITS = ["train", "val", "test"]


def file_info(data_dir: str, name: str) -> dict[str, Any]:
    with open(os.path.join(data_dir, name), "rb") as f:
        content = f.read()
    return {"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def git_output(args: list[str]) -> str:
    result = subprocess.run(["git"] + args, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload processed dataset to S3 with version")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--prefix", default="processed/cifar10")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    client = get_client()
    bucket = get_bucket()
    s3_prefix = f"{args.prefix}/{args.version}/"

    # versions are immutable
    response = client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix, MaxKeys=1)
    if response.get("KeyCount", 0) > 0:
        print(
            f"error: version {args.version} already exists in s3://{bucket}/{s3_prefix}\n"
            "choose another version, for example: make data VERSION=v2",
            file=sys.stderr,
        )
        sys.exit(1)

    counts = {}
    image_files = []
    for split in SPLITS:
        with open(os.path.join(args.data_dir, "annotations", split + ".json")) as f:
            images = json.load(f)["images"]
        counts[split] = len(images)
        for image in images:
            image_files.append(image["file_name"])

    print(f"uploading {len(image_files)} images to s3://{bucket}/{s3_prefix}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for name in image_files:
            path = os.path.join(args.data_dir, name)
            futures.append(executor.submit(client.upload_file, path, bucket, s3_prefix + name))
        for future in futures:
            future.result()

    files = []
    for name in ["annotations/train.json", "annotations/val.json", "annotations/test.json"]:
        files.append(file_info(args.data_dir, name))
    files.append(file_info(args.data_dir, "stats.json"))
    for info in files:
        client.upload_file(
            os.path.join(args.data_dir, info["path"]), bucket, s3_prefix + info["path"]
        )
        print(f"uploaded {info['path']}")

    manifest = {
        "dataset": "cifar10",
        "version": args.version,
        "num_images": counts,
        "files": files,
        "params": {"val_size": args.val_size, "seed": args.seed},
        "git_commit": git_output(["rev-parse", "HEAD"]),
    }
    manifest_path = os.path.join(args.data_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    # manifest goes last, so it appears only when everything else is uploaded
    client.upload_file(manifest_path, bucket, s3_prefix + "manifest.json")
    print(f"done, s3://{bucket}/{s3_prefix}")


if __name__ == "__main__":
    main()
