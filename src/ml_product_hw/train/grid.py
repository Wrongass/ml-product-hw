import argparse
import itertools
import os
import subprocess
import sys

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grid of experiments")
    parser.add_argument("--config", default="configs/grid.yaml")
    parser.add_argument("--out-dir", default="tmp/grid")
    args = parser.parse_args()

    grid_cfg = OmegaConf.load(args.config)
    base = OmegaConf.load(grid_cfg.base)

    keys = []
    values = []
    for key, value in grid_cfg.grid.items():
        keys.append(str(key))
        values.append(list(value))

    os.makedirs(args.out_dir, exist_ok=True)
    paths = []
    for combo in itertools.product(*values):
        changes = []
        names = {}
        for key, value in zip(keys, combo, strict=True):
            changes.append(f"{key}={value}")
            names[key.split(".")[-1]] = value
        cfg = OmegaConf.merge(base, OmegaConf.from_dotlist(changes))
        cfg.run_name = grid_cfg.run_name.format(**names)

        path = os.path.join(args.out_dir, cfg.run_name + ".yaml")
        OmegaConf.save(cfg, path)
        paths.append(path)
    print(f"{len(paths)} runs in grid")

    for path in paths:
        print("run", path)
        subprocess.run(
            [sys.executable, "-m", "ml_product_hw.train.train", "--config", path], check=True
        )


if __name__ == "__main__":
    main()
