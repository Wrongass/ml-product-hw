import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import requests
from omegaconf import OmegaConf

from ml_product_hw.train.dataset import download_data


def run_worker(url: str, images: list[bytes], indices: list[int]) -> list[tuple[float, int]]:
    session = requests.Session()
    results = []
    for i in indices:
        start = time.perf_counter()
        try:
            response = session.post(url + "/predict", files={"file": ("image.png", images[i])})
            status = response.status_code
        except requests.RequestException:
            status = 0
        results.append(((time.perf_counter() - start) * 1000, status))
    return results


def run(url: str, images: list[bytes], workers: int) -> tuple[list[tuple[float, int]], float]:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for w in range(workers):
            indices = list(range(w, len(images), workers))
            futures.append(executor.submit(run_worker, url, images, indices))
        results = []
        for future in futures:
            results.extend(future.result())
    return results, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test for POST /predict")
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--url", default="http://localhost:8081")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    cfg: Any = OmegaConf.load(args.config)
    local_dir = os.path.join(args.data_dir, cfg.data_s3_path)
    download_data(cfg.data_s3_path, local_dir)
    with open(os.path.join(local_dir, "annotations", "test.json")) as f:
        test_images = json.load(f)["images"]

    rng = random.Random(args.seed)
    images = []
    for image in rng.choices(test_images, k=args.warmup + args.requests):
        with open(os.path.join(local_dir, image["file_name"]), "rb") as f:
            images.append(f.read())

    rows = []
    raw = []
    for workers in args.workers:
        run(args.url, images[: args.warmup], workers)
        results, elapsed = run(args.url, images[args.warmup :], workers)

        latencies = []
        errors = 0
        for latency, status in results:
            raw.append([workers, round(latency, 3), status])
            if status == 200:
                latencies.append(latency)
            else:
                errors += 1
        row = [
            workers,
            len(results),
            errors,
            len(latencies) / elapsed,
            float(np.mean(latencies)),
            float(np.median(latencies)),
            float(np.percentile(latencies, 90)),
            float(np.percentile(latencies, 95)),
        ]
        rows.append(row)
        print(
            f"workers {workers}: rps {row[3]:.1f}, mean {row[4]:.1f} ms, "
            f"p95 {row[7]:.1f} ms, errors {errors}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "load_test_raw.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["workers", "latency_ms", "status"])
        writer.writerows(raw)

    lines = [
        "# Нагрузочный тест",
        "",
        "| Workers | Запросов | Ошибок | RPS | Среднее, мс | Медиана, мс | p90, мс | p95, мс |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.1f} | {r[4]:.1f} | {r[5]:.1f} "
            f"| {r[6]:.1f} | {r[7]:.1f} |"
        )
    text = "\n".join(lines) + "\n"

    # keep conditions and conclusions written by hand
    report_path = os.path.join(args.out_dir, "load_test.md")
    if os.path.exists(report_path):
        with open(report_path) as f:
            old = f.read()
        if "## Условия замера" in old:
            text += "\n" + old[old.index("## Условия замера") :]
    with open(report_path, "w") as f:
        f.write(text)
    print(f"report saved to {report_path}")


if __name__ == "__main__":
    main()
