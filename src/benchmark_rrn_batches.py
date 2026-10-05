from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from train_rrn_sharded import autocast_context, make_scaler
from upstream_rrn_adapter import build_rrn
from vimeo_sharded_dataset import VimeoSeptupletDataset, gaussian_downsample


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark RRN training batch throughput")
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", default="16,32,48,64")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=8)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--decoder", default="auto")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[int(fraction * (len(values) - 1))]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    batch_sizes = [int(value) for value in args.batches.split(",")]

    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_rrn(args.upstream_dir, args.scale, args.channels, args.blocks).to(device)
        model.load_state_dict(checkpoint["model"])
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
        scaler = make_scaler(True)
        criterion = nn.L1Loss(reduction="sum")
        dataset = VimeoSeptupletDataset(
            args.data, split="train", scale=args.scale, crop_size=args.crop_size,
            augment=True, defer_downsample=True, decoder=args.decoder,
        )
        decoder = dataset.decoder
        if dataset.env is not None:
            dataset.env.close()
            dataset.env = None
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers,
            pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
            prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        )
        data_times: list[float] = []
        compute_times: list[float] = []
        wall_times: list[float] = []
        failed = None
        iterator = iter(loader)
        previous = time.perf_counter()
        try:
            for index in range(args.warmup + args.steps):
                target, _ = next(iterator)
                torch.cuda.synchronize()
                compute_started = time.perf_counter()
                target = target.to(device, non_blocking=True)
                low_resolution = gaussian_downsample(target, args.scale)
                low_resolution = torch.cat((low_resolution[:, :, 1:2], low_resolution), dim=2)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, True):
                    prediction = model(low_resolution)
                    batch, _, temporal, _, _ = low_resolution.shape
                    loss = criterion(prediction, target) / (batch * temporal)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                finished = time.perf_counter()
                if index >= args.warmup:
                    data_times.append(compute_started - previous)
                    compute_times.append(finished - compute_started)
                    wall_times.append(finished - previous)
                    measured = index - args.warmup + 1
                    if measured == 1 or measured % max(1, args.steps // 5) == 0:
                        print(
                            f"BATCH_PROGRESS batch={batch_size} step={measured}/{args.steps} "
                            f"wall_s={wall_times[-1]:.3f}",
                            flush=True,
                        )
                previous = finished
        except torch.OutOfMemoryError as error:
            failed = f"OOM: {error}"
        finally:
            if getattr(loader, "_iterator", None) is not None:
                loader._iterator._shutdown_workers()
            del iterator, loader, dataset, optimizer, scaler, model
            torch.cuda.empty_cache()

        if failed:
            result = {"batch": batch_size, "status": "oom", "error": failed}
        else:
            mean_wall = statistics.mean(wall_times)
            result = {
                "batch": batch_size,
                "status": "ok",
                "decoder": decoder,
                "steps": len(wall_times),
                "mean_wall_s": mean_wall,
                "median_wall_s": statistics.median(wall_times),
                "p95_wall_s": percentile(wall_times, 0.95),
                "mean_data_s": statistics.mean(data_times),
                "mean_compute_s": statistics.mean(compute_times),
                "clips_s": batch_size / mean_wall,
                "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
            }
        print("BATCH_BENCH " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
