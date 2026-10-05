from __future__ import annotations

from io import BytesIO
from pathlib import Path
import random
import zipfile

import lmdb
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

try:
    import pyspng
except ImportError:  # Optional fast decoder.
    pyspng = None

try:
    from torchvision.io import ImageReadMode, decode_image
except (ImportError, RuntimeError):  # Some Torch/Torchvision builds are mismatched.
    ImageReadMode = None
    decode_image = None


ZIP_PREFIX = "vimeo_septuplet"
LMDB_LIST_PREFIX = "__lists__"


def gaussian_kernel(
    size: int,
    sigma: float,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)


def gaussian_downsample(frames: torch.Tensor, scale: int) -> torch.Tensor:
    leading = frames.shape[:-2]
    height, width = frames.shape[-2:]
    flat = frames.contiguous().view(-1, 1, height, width)
    pad = 6 + scale * 2
    extra_h = (scale - height % scale) % scale if scale == 3 else 0
    extra_w = (scale - width % scale) % scale if scale == 3 else 0
    flat = F.pad(flat, (pad, pad + extra_w, pad, pad + extra_h), mode="reflect")
    kernel = gaussian_kernel(13, 0.4 * scale, flat.dtype, flat.device).view(1, 1, 13, 13)
    flat = F.conv2d(flat, kernel, stride=scale)[:, :, 2:-2, 2:-2]
    return flat.view(*leading, flat.shape[-2], flat.shape[-1])


class VimeoSeptupletDataset(Dataset):
    def __init__(
        self,
        source: str | Path,
        split: str = "train",
        scale: int = 4,
        crop_size: int = 64,
        max_samples: int | None = None,
        augment: bool = True,
        defer_downsample: bool = False,
        decoder: str = "auto",
    ) -> None:
        self.source = Path(source)
        self.scale = scale
        self.crop_size = crop_size
        self.augment = augment
        self.defer_downsample = defer_downsample
        self.decoder = self._resolve_decoder(decoder)
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

    @staticmethod
    def _resolve_decoder(decoder: str) -> str:
        if decoder == "auto":
            if pyspng is not None:
                return "pyspng"
            if decode_image is not None:
                return "torchvision"
            return "pillow"
        if decoder == "pyspng" and pyspng is None:
            raise RuntimeError("pyspng decoder requested but pyspng is not installed")
        if decoder == "torchvision" and decode_image is None:
            raise RuntimeError("torchvision decoder requested but torchvision.io is unavailable")
        if decoder not in {"pillow", "pyspng", "torchvision"}:
            raise ValueError(f"Unknown PNG decoder: {decoder}")
        return decoder

    def _decode_rgb(self, encoded: bytes) -> torch.Tensor:
        if self.decoder == "pyspng":
            array = pyspng.load(encoded)
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=2)
            elif array.shape[2] == 4:
                array = array[..., :3]
            return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        if self.decoder == "torchvision":
            # decode_image consumes the encoded buffer synchronously, while the
            # LMDB transaction is still alive, and returns independent pixels.
            buffer = torch.frombuffer(encoded, dtype=torch.uint8)
            return decode_image(buffer, mode=ImageReadMode.RGB)
        with Image.open(BytesIO(encoded)) as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1)

    def _load_frames(self, relative: str) -> torch.Tensor:
        frames: list[torch.Tensor] = []
        if self.is_lmdb:
            with self._txn() as txn:
                for index in range(1, 8):
                    key = f"sequences/{relative}/im{index}.png".encode("utf-8")
                    value = txn.get(key)
                    if value is None:
                        raise KeyError(key.decode("utf-8"))
                    frames.append(self._decode_rgb(value))
        elif self.is_zip:
            with zipfile.ZipFile(self.source) as archive:
                for index in range(1, 8):
                    member = f"{ZIP_PREFIX}/sequences/{relative}/im{index}.png"
                    frames.append(self._decode_rgb(archive.read(member)))
        else:
            folder = self.root / "sequences" / relative
            for index in range(1, 8):
                frames.append(self._decode_rgb((folder / f"im{index}.png").read_bytes()))
        return torch.stack(frames)

    def __getitem__(self, index: int):
        frames = self._load_frames(self.samples[index])
        height = frames.shape[-2]
        width = frames.shape[-1]
        width -= width % self.scale
        height -= height % self.scale
        if self.crop_size:
            crop = min(self.crop_size, width, height)
            crop -= crop % self.scale
            left = random.randint(0, width - crop) if width > crop else 0
            top = random.randint(0, height - crop) if height > crop else 0
            frames = frames[:, :, top:top + crop, left:left + crop]
        else:
            frames = frames[:, :, :height, :width]
        if self.augment and random.random() < 0.5:
            frames = torch.flip(frames, dims=(2,))
        if self.augment and random.random() < 0.5:
            frames = torch.flip(frames, dims=(3,))
        high_resolution = frames.permute(1, 0, 2, 3).contiguous().float().div_(255.0)
        if self.scale == 4:
            high_resolution = F.pad(high_resolution, (8, 8, 8, 8), mode="reflect")
        if self.defer_downsample:
            return high_resolution, self.samples[index]
        low_resolution = gaussian_downsample(high_resolution, self.scale)
        low_resolution = torch.cat((low_resolution[:, 1:2], low_resolution), dim=1)
        return low_resolution, high_resolution, self.samples[index]
