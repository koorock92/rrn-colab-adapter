from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn


class SequenceRRN(nn.Module):
    """Expose the upstream two-frame RRN cell as a seven-frame sequence model."""

    def __init__(self, upstream_arch: Path, scale: int, channels: int, blocks: int) -> None:
        super().__init__()
        spec = importlib.util.spec_from_file_location("rrn_upstream_arch", upstream_arch)
        if spec is None or spec.loader is None:
            raise ImportError(upstream_arch)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.cell = module.RRN(scale, channels, blocks)
        self.scale = scale
        self.channels = channels

    def forward(self, low_resolution: torch.Tensor) -> torch.Tensor:
        batch, _, frames, height, width = low_resolution.shape
        hidden = low_resolution.new_zeros(batch, self.channels, height, width)
        output = low_resolution.new_zeros(
            batch, self.scale * self.scale * 3, height, width
        )
        predictions = []
        for index in range(frames - 1):
            hidden, output = self.cell(
                low_resolution[:, :, index : index + 2],
                hidden,
                output,
                index == 0,
            )
            predictions.append(output)
        return torch.stack(predictions, dim=2)


def build_rrn(upstream_dir: str | Path, scale: int, channels: int, blocks: int) -> nn.Module:
    arch = Path(upstream_dir) / "RRN" / "arch.py"
    if not arch.is_file():
        raise FileNotFoundError(
            f"Missing upstream RRN architecture: {arch}. Clone "
            "https://github.com/junpan19/RRN.git first."
        )
    return SequenceRRN(arch, scale, channels, blocks)
