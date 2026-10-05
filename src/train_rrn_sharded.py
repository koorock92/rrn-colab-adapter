from __future__ import annotations

import argparse
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from upstream_rrn_adapter import build_rrn
from vimeo_sharded_dataset import VimeoSeptupletDataset, gaussian_downsample


class FixedOrderSampler(Sampler):
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def parse_args():
    parser = argparse.ArgumentParser(description="Resumable RRN trainer for staged LMDB shards")
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--step-size", type=int, default=60)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--metrics-flush-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--gpu-downsample", action="store_true")
    parser.add_argument("--decoder", choices=("auto", "pillow", "torchvision", "pyspng"), default="auto")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    return parser.parse_args()


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def close_dataset(dataset) -> None:
    if dataset is not None and dataset.env is not None:
        dataset.env.close()
        dataset.env = None


class ShardStager:
    def __init__(self, source: Path, cache: Path) -> None:
        self.source = source.resolve()
        self.cache = cache.resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shard-copy")
        self.futures: dict[str, Future] = {}

    def _copy(self, name: str) -> Path:
        source = (self.source / name).resolve()
        target = (self.cache / name).resolve()
        if source.parent != self.source or target.parent != self.cache:
            raise ValueError(f"Unsafe shard name: {name}")
        if (target / "data.mdb").is_file():
            return target
        temporary = self.cache / f".{name}.copying"
        temporary.mkdir(parents=True, exist_ok=True)
        files = [path for path in source.rglob("*") if path.is_file()]
        total = sum(path.stat().st_size for path in files)
        copied = 0
        for source_file in files:
            relative = source_file.relative_to(source)
            target_file = temporary / relative
            target_file.parent.mkdir(parents=True, exist_ok=True)
            source_size = source_file.stat().st_size
            existing = target_file.stat().st_size if target_file.exists() else 0
            if existing > source_size:
                target_file.unlink()
                existing = 0
            copied += existing
        started = time.perf_counter()
        last_report = started
        initial = copied
        print(f"STAGE_START name={name} resumed_gib={initial / 2**30:.2f} total_gib={total / 2**30:.2f}", flush=True)
        for source_file in files:
            relative = source_file.relative_to(source)
            target_file = temporary / relative
            source_size = source_file.stat().st_size
            existing = target_file.stat().st_size if target_file.exists() else 0
            if existing == source_size:
                continue
            with source_file.open("rb") as source_stream, target_file.open("ab" if existing else "wb") as target_stream:
                source_stream.seek(existing)
                while True:
                    chunk = source_stream.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    target_stream.write(chunk)
                    copied += len(chunk)
                    now = time.perf_counter()
                    if now - last_report >= 2.0:
                        elapsed = max(now - started, 1e-6)
                        rate = max((copied - initial) / elapsed, 1.0)
                        remaining = max(total - copied, 0)
                        print(
                            f"STAGING name={name} percent={100.0 * copied / max(total, 1):.1f} "
                            f"copied_gib={copied / 2**30:.2f} total_gib={total / 2**30:.2f} "
                            f"mib_s={rate / 2**20:.1f} eta_s={remaining / rate:.0f}",
                            flush=True,
                        )
                        last_report = now
                target_stream.flush()
        temporary.replace(target)
        print(f"STAGED {name} in {time.perf_counter() - started:.1f}s", flush=True)
        return target

    def prefetch(self, name: str) -> None:
        if name not in self.futures and not (self.cache / name / "data.mdb").is_file():
            print(f"PREFETCH_START name={name}", flush=True)
            self.futures[name] = self.pool.submit(self._copy, name)

    def get(self, name: str) -> Path:
        future = self.futures.pop(name, None)
        if future and not future.done():
            print(f"PREFETCH_WAIT name={name}", flush=True)
        path = future.result() if future else self._copy(name)
        print(f"SHARD_READY name={name}", flush=True)
        return path

    def discard(self, name: str) -> None:
        target = (self.cache / name).resolve()
        if target.parent != self.cache:
            raise ValueError(target)
        if target.exists():
            shutil.rmtree(target)

    def close(self) -> None:
        self.pool.shutdown(wait=True)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def validate_psnr(model, dataset, device, amp: bool) -> float:
    model.eval()
    scores = []
    progress = tqdm(
        DataLoader(dataset, batch_size=1, num_workers=0),
        desc="Validation", unit="clip", disable=not os.isatty(2),
    )
    for low_resolution, target, _ in progress:
        low_resolution = low_resolution.to(device)
        target = target.to(device)
        with autocast_context(device, amp):
            prediction = model(low_resolution).float().clamp_(0.0, 1.0)
        prediction = prediction[..., 8:-8, 8:-8]
        target = target[..., 8:-8, 8:-8]
        coefficients = prediction.new_tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1, 1) / 255.0
        prediction_y = (prediction * coefficients).sum(dim=1)
        target_y = (target * coefficients).sum(dim=1)
        mse = (prediction_y - target_y).square().mean(dim=(-2, -1))
        scores.extend((-10.0 * torch.log10(mse.clamp_min(1e-12))).flatten().cpu().tolist())
    model.train()
    return float(sum(scores) / len(scores))


def shard_order(count: int, seed: int, epoch: int) -> list[int]:
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


def sample_order(count: int, seed: int, epoch: int, shard_index: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed + epoch * 10007 + shard_index)
    return torch.randperm(count, generator=generator).tolist()


def main() -> None:
    args = parse_args()
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    amp = bool(args.amp and device.type == "cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    manifest = json.loads((args.shards_root / "manifest.json").read_text(encoding="utf-8"))
    shards = manifest["train_shards"]
    validation_name = manifest["validation_shard"]["name"]
    args.output.mkdir(parents=True, exist_ok=True)
    last_path = args.output / "last.pt"
    best_path = args.output / "best.pt"
    metrics_path = args.output / "metrics.jsonl"

    model = build_rrn(args.upstream_dir, args.scale, args.channels, args.blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scaler = make_scaler(amp)
    criterion = nn.L1Loss(reduction="sum")

    epoch = 1
    shard_position = 0
    next_batch = 0
    global_step = 0
    best_psnr = float("-inf")
    last_loss = float("nan")
    if args.resume and args.resume.is_file():
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        epoch = int(checkpoint["epoch"])
        shard_position = int(checkpoint.get("shard_position", 0))
        next_batch = int(checkpoint.get("next_batch", 0))
        global_step = int(checkpoint["global_step"])
        best_psnr = float(checkpoint.get("best_psnr", float("-inf")))
        last_loss = float(checkpoint.get("loss", float("nan")))
        print(f"RESUMED epoch={epoch} shard_position={shard_position} next_batch={next_batch} step={global_step}", flush=True)

    config = {**vars(args), "shards_root": str(args.shards_root), "cache_dir": str(args.cache_dir), "output": str(args.output), "device_resolved": str(device), "amp_resolved": amp}
    (args.output / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    metrics_buffer: list[str] = []

    def flush_metrics() -> None:
        if not metrics_buffer:
            return
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write("".join(metrics_buffer))
        metrics_buffer.clear()

    stager = ShardStager(args.shards_root, args.cache_dir)
    validation_path = stager.get(validation_name)
    validation_dataset = VimeoSeptupletDataset(
        validation_path, split="test", scale=args.scale, crop_size=0,
        augment=False, decoder=args.decoder,
    )
    print(f"INPUT_PIPELINE decoder={validation_dataset.decoder} workers={args.workers} prefetch_factor={args.prefetch_factor}", flush=True)
    close_dataset(validation_dataset)

    def payload(next_epoch: int, next_shard: int, next_batch_value: int) -> dict:
        return {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": next_epoch, "shard_position": next_shard,
            "next_batch": next_batch_value, "global_step": global_step,
            "loss": last_loss, "best_psnr": best_psnr, "config": config,
        }

    stop = False
    try:
        while epoch <= args.epochs and not stop:
            order = shard_order(len(shards), args.seed, epoch)
            while shard_position < len(order):
                shard_index = order[shard_position]
                shard_name = shards[shard_index]["name"]
                shard_path = stager.get(shard_name)
                if not args.no_prefetch and shard_position + 1 < len(order):
                    stager.prefetch(shards[order[shard_position + 1]]["name"])
                dataset = VimeoSeptupletDataset(
                    shard_path,
                    split="train",
                    scale=args.scale,
                    crop_size=args.crop_size,
                    augment=True,
                    defer_downsample=args.gpu_downsample,
                    decoder=args.decoder,
                )
                close_dataset(dataset)
                indices = sample_order(len(dataset), args.seed, epoch, shard_index)
                resume_batch = next_batch
                sampler = FixedOrderSampler(indices[resume_batch * args.batch_size:])
                loader_options = {
                    "batch_size": args.batch_size,
                    "sampler": sampler,
                    "num_workers": args.workers,
                    "pin_memory": use_cuda,
                    "drop_last": True,
                    "persistent_workers": args.workers > 0,
                }
                if args.workers > 0:
                    loader_options["prefetch_factor"] = args.prefetch_factor
                loader = DataLoader(dataset, **loader_options)
                total_batches = len(dataset) // args.batch_size
                progress = tqdm(
                    loader,
                    total=max(0, total_batches - resume_batch),
                    desc=f"Epoch {epoch}/{args.epochs} shard {shard_position + 1}/{len(order)}",
                    unit="batch",
                    disable=not os.isatty(2),
                )
                previous_step_finished = time.perf_counter()
                for local_batch, batch_data in enumerate(progress):
                    batch_index = resume_batch + local_batch
                    started = time.perf_counter()
                    if args.gpu_downsample:
                        target, names = batch_data
                        target = target.to(device, non_blocking=use_cuda)
                        low_resolution = gaussian_downsample(target, args.scale)
                        low_resolution = torch.cat((low_resolution[:, :, 1:2], low_resolution), dim=2)
                    else:
                        low_resolution, target, names = batch_data
                        low_resolution = low_resolution.to(device, non_blocking=use_cuda)
                        target = target.to(device, non_blocking=use_cuda)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(device, amp):
                        prediction = model(low_resolution)
                        batch, _, temporal, _, _ = low_resolution.shape
                        loss = criterion(prediction, target) / (batch * temporal)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at step {global_step}")
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    global_step += 1
                    last_loss = float(loss.detach().cpu())
                    finished = time.perf_counter()
                    record = {
                        "epoch": epoch, "shard_position": shard_position,
                        "shard": shard_name, "batch": batch_index + 1,
                        "global_step": global_step, "loss": last_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                        "seconds": finished - started,
                        "wall_seconds": finished - previous_step_finished,
                        "sample": names[0],
                    }
                    previous_step_finished = finished
                    metrics_buffer.append(json.dumps(record) + "\n")
                    if len(metrics_buffer) >= args.metrics_flush_every:
                        flush_metrics()
                    progress.set_postfix(loss=f"{last_loss:.3f}", step=global_step, sec=f"{record['seconds']:.2f}")
                    if global_step == 1 or global_step % args.log_every == 0:
                        gpu_mib = torch.cuda.memory_allocated() / 2**20 if use_cuda else 0.0
                        print(
                            f"TRAIN epoch={epoch}/{args.epochs} shard={shard_position + 1}/{len(order)} "
                            f"batch={batch_index + 1}/{total_batches} step={global_step} "
                            f"loss={last_loss:.4f} compute_s={record['seconds']:.3f} "
                            f"wall_s={record['wall_seconds']:.3f} "
                            f"samples_s={args.batch_size / max(record['wall_seconds'], 1e-6):.1f} "
                            f"gpu_mib={gpu_mib:.0f}",
                            flush=True,
                        )
                    reached_limit = args.max_steps is not None and global_step >= args.max_steps
                    if global_step % args.save_every == 0 or reached_limit:
                        flush_metrics()
                        save_checkpoint(last_path, payload(epoch, shard_position, batch_index + 1))
                    if reached_limit:
                        stop = True
                        break
                progress.close()
                close_dataset(dataset)
                if stop:
                    break

                next_batch = 0
                shard_position += 1
                close_dataset(validation_dataset)
                validation_psnr = validate_psnr(model, validation_dataset, device, amp)
                close_dataset(validation_dataset)
                improved = validation_psnr > best_psnr
                if improved:
                    best_psnr = validation_psnr
                flush_metrics()
                save_checkpoint(last_path, payload(epoch, shard_position, 0))
                if improved:
                    best_payload = payload(epoch, shard_position, 0)
                    best_payload["validation_psnr"] = validation_psnr
                    save_checkpoint(best_path, best_payload)
                print(f"SHARD_COMPLETE name={shard_name} validation_psnr={validation_psnr:.4f} best_psnr={best_psnr:.4f}", flush=True)
                stager.discard(shard_name)

            if stop:
                close_dataset(validation_dataset)
                validation_psnr = validate_psnr(model, validation_dataset, device, amp)
                close_dataset(validation_dataset)
                improved = validation_psnr > best_psnr
                if improved:
                    best_psnr = validation_psnr
                checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
                checkpoint["best_psnr"] = best_psnr
                checkpoint["validation_psnr"] = validation_psnr
                save_checkpoint(last_path, checkpoint)
                if improved:
                    save_checkpoint(best_path, checkpoint)
                break

            scheduler.step()
            epoch += 1
            shard_position = 0
            next_batch = 0
            save_checkpoint(last_path, payload(epoch, 0, 0))
    finally:
        flush_metrics()
        close_dataset(validation_dataset)
        stager.close()
    print(f"TRAINING_STOPPED_AT_STEP={global_step} last={last_path} best={best_path}", flush=True)


if __name__ == "__main__":
    main()
