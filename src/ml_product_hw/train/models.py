from typing import Any

import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, channels: list[int], hidden: int, num_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        # 32 -> 16 -> 8 (-> 4)
        size = 32 // 2 ** len(channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * size * size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.classifier(self.features(x))
        return out


def build_model(cfg: Any, num_classes: int) -> nn.Module:
    if cfg.type == "simple_cnn":
        return SimpleCNN(list(cfg.params.channels), cfg.params.hidden, num_classes)
    if cfg.type == "linear":
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, num_classes))
    raise ValueError("unknown model type: " + cfg.type)
