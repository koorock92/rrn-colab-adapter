from __future__ import annotations

from io import BytesIO
from pathlib import Path
import random
import zipfile

import lmdb
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F


ZIP_PREFIX = "vimeo_septuplet"
LMDB_LIST_PREFIX = "__lists__"


def gaussian_kernel(size: int, sigma: float, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype) - (size - 1) / 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)


def gaussian_downsample(frames: torch.Tensor, scale: int) -> torch.Tensor:
    channels, count, height, width = frames.shape
    flat = frames.contiguous().view(-1, 1, height, width)
    pad = 6 + scale * 2
    extra_h = (scale - height % scale) % scale if scale == 3 else 0
    extra_w = (scale - width % scale) % scale if scale == 3 else 0
    flat = F.pad(flat, (pad, pad + extra_w, pad, pad + extra_h), mode="reflect")
    kernel = gaussian_kernel(13, 0.4 * scale, flat.dtype).view(1, 1, 13, 13)
    flat = F.conv2d(flat, kernel, stride=scale)[:, :, 2:-2, 2:-2]
    return flat.view(channels, count, flat.shape[-2], flat.shape[-1])


class VimeoSeptupletDataset(Dataset):
    def __init__(
        self,
        source: str | Path,
        split: str = "train",
        scale: int = 4,
        crop_size: int = 64,
        max_samples: int | None = None,
        augment: bool = True,
    ) -> None:
        self.source = Path(source)
        self.scale = scale
        self.crop_size = crop_size
        self.augment = augment
        self.is_zip = self.source.is_file()
        self.is_lmdb = (self.source / "data.mdb").is_file()
        self.env = None
        list_name = f"sep_{split}list.txt"
        if self.is_lmdb:
            self.root = self.source
            with self._txn() as txn:
                value = txn.get(f"{LMDB_LIST_PREFIX}/{list_name}".encode("utf-8"))
            if value is None:
                raise RuntimeError(f"Missing {list_name} in {self.source}")
            self.samples = value.decode("utf-8-sig").splitlines()
        elif self.is_zip:
            with zipfile.ZipFile(self.source) as archive:
                member = f"{ZIP_PREFIX}/{list_name}"
                self.samples = archive.read(member).decode("utf-8-sig").splitlines()
        else:
            root = self.source / ZIP_PREFIX if (self.source / ZIP_PREFIX).is_dir() else self.source
            self.root = root
            self.samples = (root / list_name).read_text(encoding="utf-8-sig").splitlines()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise RuntimeError(f"No samples found in {self.source}")

    def __len__(self) -> int:
        return len(self.samples)

    def _get_env(self):
        if self.env is None:
            self.env = lmdb.open(
                str(self.source),
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=2048,
            )
        return self.env

    def _txn(self):
        return self._get_env().begin(buffers=False)

    def _load_frames(self, relative: str) -> list[Image.Image]:
        frames = []
        if self.is_lmdb:
            with self._txn() as txn:
                for index in range(1, 8):
                    key = f"sequences/{relative}/im{index}.png".encode("utf-8")
                    value = txn.get(key)
                    if value is None:
                        raise KeyError(key.decode("utf-8"))
                    with Image.open(BytesIO(value)) as image:
                        frames.append(image.convert("RGB").copy())
        elif self.is_zip:
            with zipfile.ZipFile(self.source) as archive:
                for index in range(1, 8):
                    member = f"{ZIP_PREFIX}/sequences/{relative}/im{index}.png"
                    with Image.open(BytesIO(archive.read(member))) as image:
                        frames.append(image.convert("RGB").copy())
        else:
            folder = self.root / "sequences" / relative
            for index in range(1, 8):
                with Image.open(folder / f"im{index}.png") as image:
                    frames.append(image.convert("RGB").copy())
        return frames

    def __getitem__(self, index: int):
        frames = self._load_frames(self.samples[index])
        width = min(frame.width for frame in frames)
        height = min(frame.height for frame in frames)
        width -= width % self.scale
        height -= height % self.scale
        if self.crop_size:
            crop = min(self.crop_size, width, height)
            crop -= crop % self.scale
            left = random.randint(0, width - crop) if width > crop else 0
            top = random.randint(0, height - crop) if height > crop else 0
            frames = [frame.crop((left, top, left + crop, top + crop)) for frame in frames]
        else:
            frames = [frame.crop((0, 0, width, height)) for frame in frames]
        if self.augment and random.random() < 0.5:
            frames = [ImageOps.flip(frame) for frame in frames]
        if self.augment and random.random() < 0.5:
            frames = [ImageOps.mirror(frame) for frame in frames]
        array = np.stack([np.asarray(frame, dtype=np.float32) / 255.0 for frame in frames])
        high_resolution = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
        if self.scale == 4:
            high_resolution = F.pad(high_resolution, (8, 8, 8, 8), mode="reflect")
        low_resolution = gaussian_downsample(high_resolution, self.scale)
        low_resolution = torch.cat((low_resolution[:, 1:2], low_resolution), dim=1)
        return low_resolution, high_resolution, self.samples[index]
