from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import time

from torch.utils.data import DataLoader

from vimeo_sharded_dataset import VimeoSeptupletDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Vimeo-90K PNG decoding and prefetch")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--decoder", choices=("auto", "pillow", "torchvision", "pyspng"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = VimeoSeptupletDataset(
        args.data, split=args.split, scale=4, crop_size=64,
        augment=True, defer_downsample=True, decoder=args.decoder,
    )
    # The constructor reads the sample list through LMDB.  Workers must open
    # their own handles (required by Windows spawn and safer on Linux too).
    if dataset.env is not None:
        dataset.env.close()
        dataset.env = None
    options = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
        "drop_last": True,
    }
    if args.workers > 0:
        options["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **options)
    timings = []
    step = 0
    previous = time.perf_counter()
    while len(timings) < args.steps:
        for _ in loader:
            step += 1
            now = time.perf_counter()
            if step > args.warmup:
                timings.append(now - previous)
            previous = now
            if len(timings) >= args.steps:
                break
    mean = statistics.fmean(timings)
    timings.sort()
    p95 = timings[min(len(timings) - 1, int(len(timings) * 0.95))]
    print(
        f"decoder={dataset.decoder} workers={args.workers} "
        f"prefetch_factor={args.prefetch_factor} batch={args.batch_size} "
        f"mean_s={mean:.4f} p95_s={p95:.4f} clips_s={args.batch_size / mean:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
