from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import shutil

import lmdb
import numpy as np
from PIL import Image


LIST_PREFIX = "__lists__"


def png_bytes(seed: int, frame: int, size: int = 32) -> bytes:
    rng = np.random.default_rng(seed * 100 + frame)
    y, x = np.mgrid[:size, :size]
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[..., 0] = (x * 5 + seed * 17 + frame * 3) % 256
    image[..., 1] = (y * 7 + seed * 11 + frame * 5) % 256
    image[..., 2] = ((x + y) * 3 + rng.integers(0, 12, (size, size))) % 256
    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return stream.getvalue()


def write_lmdb(path: Path, samples: list[str], split: str) -> dict:
    path.mkdir(parents=True)
    env = lmdb.open(str(path), map_size=64 * 1024 * 1024, subdir=True)
    with env.begin(write=True) as txn:
        for sample_index, sample in enumerate(samples):
            for frame in range(1, 8):
                key = f"sequences/{sample}/im{frame}.png".encode()
                txn.put(key, png_bytes(sample_index + len(samples), frame))
        txn.put(
            f"{LIST_PREFIX}/sep_{split}list.txt".encode(),
            ("\n".join(samples) + "\n").encode(),
        )
    env.sync(True)
    entries = env.stat()["entries"]
    env.close()
    return {
        "name": path.name,
        "samples": len(samples),
        "entries": entries,
        "data_bytes": (path / "data.mdb").stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    shards = []
    for shard_index in range(2):
        samples = [f"mini/{shard_index:02d}{index:03d}" for index in range(4)]
        shards.append(
            write_lmdb(args.output / f"train-{shard_index:03d}.lmdb", samples, "train")
        )
    validation = write_lmdb(
        args.output / "validation.lmdb",
        ["mini/validation"],
        "test",
    )
    manifest = {
        "format": "vimeo90k-lmdb-shards-v1",
        "train_samples": sum(item["samples"] for item in shards),
        "validation_samples": validation["samples"],
        "train_shards": shards,
        "validation_shard": validation,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(args.output / "manifest.json")


if __name__ == "__main__":
    main()
