from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from upstream_rrn_adapter import build_rrn
from vimeo_sharded_dataset import VimeoSeptupletDataset, gaussian_downsample


class FixedOrderSampler(Sampler[int]):
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def parse_args():
    parser = argparse.ArgumentParser(description="Resumable RRN trainer for one local LMDB")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--step-size", type=int, default=60)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--metrics-flush-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--val-samples", type=int, default=100)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gpu-downsample", action="store_true")
    parser.add_argument("--decoder", choices=("auto", "pillow", "torchvision", "pyspng"), default="auto")
    parser.add_argument("--prefetch-factor", type=int, default=8)
    return parser.parse_args()


def autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast("cuda", enabled=True)


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def close_dataset(dataset) -> None:
    if dataset is not None and dataset.env is not None:
        dataset.env.close()
        dataset.env = None


def stop_loader(loader) -> None:
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    gc.collect()


@torch.inference_mode()
def validate_psnr(model, dataset, device, amp: bool) -> float:
    model.eval()
    scores: list[float] = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for index, (low_resolution, target, _) in enumerate(loader, start=1):
        low_resolution = low_resolution.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast_context(amp):
            prediction = model(low_resolution).float().clamp_(0.0, 1.0)
        prediction = prediction[..., 8:-8, 8:-8]
        target = target[..., 8:-8, 8:-8]
        coefficients = prediction.new_tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1, 1) / 255.0
        prediction_y = (prediction * coefficients).sum(dim=1)
        target_y = (target * coefficients).sum(dim=1)
        mse = (prediction_y - target_y).square().mean(dim=(-2, -1))
        scores.extend((-10.0 * torch.log10(mse.clamp_min(1e-12))).flatten().cpu().tolist())
        if index == 1 or index % 20 == 0 or index == len(dataset):
            print(f"VALIDATION done={index} total={len(dataset)} psnr={sum(scores) / len(scores):.4f}", flush=True)
    model.train()
    return float(sum(scores) / len(scores))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not (args.data / "data.mdb").is_file():
        raise FileNotFoundError(args.data / "data.mdb")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    amp = bool(args.amp)

    args.output.mkdir(parents=True, exist_ok=True)
    last_path = args.output / "last.pt"
    best_path = args.output / "best.pt"
    metrics_path = args.output / "metrics.jsonl"

    dataset = VimeoSeptupletDataset(
        args.data, split="train", scale=args.scale, crop_size=args.crop_size,
        augment=True, defer_downsample=args.gpu_downsample, decoder=args.decoder,
    )
    close_dataset(dataset)
    validation_dataset = VimeoSeptupletDataset(
        args.data, split="test", scale=args.scale, crop_size=0,
        max_samples=args.val_samples, augment=False, decoder=args.decoder,
    )
    close_dataset(validation_dataset)

    model = build_rrn(args.upstream_dir, args.scale, args.channels, args.blocks).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.L1Loss(reduction="sum")

    epoch = 1
    next_batch = 0
    global_step = 0
    best_psnr = float("-inf")
    last_validation_psnr = float("nan")
    last_loss = float("nan")
    checkpoint_step = 0
    if args.resume and args.resume.is_file():
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        epoch = int(checkpoint["epoch"])
        next_batch = int(checkpoint.get("next_batch", 0))
        global_step = int(checkpoint["global_step"])
        best_psnr = float(checkpoint.get("best_psnr", float("-inf")))
        last_validation_psnr = float(checkpoint.get("validation_psnr", float("nan")))
        last_loss = float(checkpoint.get("loss", float("nan")))
        checkpoint_step = global_step
        print(f"RESUMED epoch={epoch} next_batch={next_batch} step={global_step}", flush=True)

    config = {
        **vars(args), "data": str(args.data), "upstream_dir": str(args.upstream_dir),
        "output": str(args.output), "samples": len(dataset),
        "validation_samples": len(validation_dataset), "gpu": torch.cuda.get_device_name(0),
    }
    (args.output / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    print(
        f"INPUT_PIPELINE decoder={dataset.decoder} workers={args.workers} "
        f"prefetch_factor={args.prefetch_factor} samples={len(dataset)} gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    metrics_buffer: list[str] = []

    def flush_metrics() -> None:
        if metrics_buffer:
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write("".join(metrics_buffer))
            metrics_buffer.clear()

    def payload(next_epoch: int, next_batch_value: int) -> dict:
        return {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": next_epoch, "next_batch": next_batch_value,
            "global_step": global_step, "loss": last_loss,
            "best_psnr": best_psnr, "validation_psnr": last_validation_psnr,
            "config": config,
        }

    if args.max_steps is not None and global_step >= args.max_steps:
        print(f"ALREADY_COMPLETE step={global_step} max_steps={args.max_steps}", flush=True)
        return

    stop = False
    try:
        while epoch <= args.epochs and not stop:
            generator = torch.Generator().manual_seed(args.seed + epoch)
            indices = torch.randperm(len(dataset), generator=generator).tolist()
            resume_batch = next_batch
            sampler = FixedOrderSampler(indices[resume_batch * args.batch_size:])
            loader_options = {
                "batch_size": args.batch_size, "sampler": sampler,
                "num_workers": args.workers, "pin_memory": True,
                "drop_last": True, "persistent_workers": args.workers > 0,
            }
            if args.workers > 0:
                loader_options["prefetch_factor"] = args.prefetch_factor
            loader = DataLoader(dataset, **loader_options)
            total_batches = len(dataset) // args.batch_size
            previous_step_finished = time.perf_counter()

            for local_batch, batch_data in enumerate(loader):
                batch_index = resume_batch + local_batch
                started = time.perf_counter()
                if args.gpu_downsample:
                    target, names = batch_data
                    target = target.to(device, non_blocking=True)
                    low_resolution = gaussian_downsample(target, args.scale)
                    low_resolution = torch.cat((low_resolution[:, :, 1:2], low_resolution), dim=2)
                else:
                    low_resolution, target, names = batch_data
                    low_resolution = low_resolution.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast_context(amp):
                    prediction = model(low_resolution)
                    batch, _, temporal, _, _ = low_resolution.shape
                    loss_sum = criterion(prediction, target)
                    loss = loss_sum / (batch * temporal)
                    mae = loss_sum.detach() / prediction.numel()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {global_step}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                last_loss = float(loss.detach().cpu())
                finished = time.perf_counter()
                record = {
                    "epoch": epoch, "batch": batch_index + 1, "batches": total_batches,
                    "global_step": global_step, "loss": last_loss, "mae": float(mae.cpu()),
                    "lr": optimizer.param_groups[0]["lr"], "seconds": finished - started,
                    "data_seconds": started - previous_step_finished,
                    "wall_seconds": finished - previous_step_finished, "sample": names[0],
                }
                previous_step_finished = finished
                metrics_buffer.append(json.dumps(record) + "\n")
                if len(metrics_buffer) >= args.metrics_flush_every:
                    flush_metrics()

                reached_limit = args.max_steps is not None and global_step >= args.max_steps
                if global_step % args.save_every == 0 or reached_limit:
                    flush_metrics()
                    save_checkpoint(last_path, payload(epoch, batch_index + 1))
                    checkpoint_step = global_step
                    print(f"CHECKPOINT step={checkpoint_step} path={last_path}", flush=True)

                if global_step == 1 or global_step % args.log_every == 0 or reached_limit:
                    wall = record["wall_seconds"]
                    print(
                        f"TRAIN epoch={epoch}/{args.epochs} batch={batch_index + 1}/{total_batches} "
                        f"step={global_step} loss={last_loss:.4f} mae={record['mae']:.6f} "
                        f"lr={record['lr']:.3e} val_psnr={last_validation_psnr:.4f} "
                        f"best_psnr={best_psnr:.4f} checkpoint_step={checkpoint_step} "
                        f"data_s={record['data_seconds']:.3f} compute_s={record['seconds']:.3f} "
                        f"wall_s={wall:.3f} samples_s={args.batch_size / max(wall, 1e-6):.1f} "
                        f"gpu_mib={torch.cuda.memory_allocated() / 2**20:.0f}",
                        flush=True,
                    )
                if reached_limit:
                    stop = True
                    break

            stop_loader(loader)
            del loader
            close_dataset(dataset)

            if stop:
                close_dataset(validation_dataset)
                last_validation_psnr = validate_psnr(model, validation_dataset, device, amp)
                close_dataset(validation_dataset)
                improved = last_validation_psnr > best_psnr
                if improved:
                    best_psnr = last_validation_psnr
                checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
                checkpoint["validation_psnr"] = last_validation_psnr
                checkpoint["best_psnr"] = best_psnr
                save_checkpoint(last_path, checkpoint)
                if improved:
                    save_checkpoint(best_path, checkpoint)
                print(
                    f"VALIDATION_COMPLETE psnr={last_validation_psnr:.4f} best_psnr={best_psnr:.4f}",
                    flush=True,
                )
                break

            close_dataset(validation_dataset)
            last_validation_psnr = validate_psnr(model, validation_dataset, device, amp)
            close_dataset(validation_dataset)
            improved = last_validation_psnr > best_psnr
            if improved:
                best_psnr = last_validation_psnr
            scheduler.step()
            epoch += 1
            next_batch = 0
            flush_metrics()
            save_checkpoint(last_path, payload(epoch, 0))
            checkpoint_step = global_step
            if improved:
                save_checkpoint(best_path, payload(epoch, 0))
            print(
                f"EPOCH_COMPLETE epoch={epoch - 1} validation_psnr={last_validation_psnr:.4f} "
                f"best_psnr={best_psnr:.4f} checkpoint_step={checkpoint_step}",
                flush=True,
            )
    finally:
        flush_metrics()
        close_dataset(dataset)
        close_dataset(validation_dataset)

    print(f"TRAINING_STOPPED_AT_STEP={global_step} last={last_path} best={best_path}", flush=True)


if __name__ == "__main__":
    main()
