from __future__ import annotations

import argparse
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
import json
import lmdb
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
    parser.add_argument("--prefetch-factor", type=int, default=8)
    parser.add_argument("--startup-buffer-shards", type=int, default=2)
    parser.add_argument("--shard-prefetch-ahead", type=int, default=2)
    parser.add_argument("--shard-prefetch-mib-s", type=float, default=40.0)
    parser.add_argument("--copy-retries", type=int, default=3)
    parser.add_argument("--profile-every", type=int, default=100)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-scope", choices=("epoch", "shard"), default="epoch")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
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
    def __init__(
        self, source: Path, cache: Path, metadata: dict[str, dict],
        prefetch_mib_s: float = 40.0, retries: int = 3,
    ) -> None:
        self.source = source.resolve()
        self.cache = cache.resolve()
        self.prefetch_mib_s = max(float(prefetch_mib_s), 0.0)
        self.metadata = metadata
        self.retries = max(int(retries), 1)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shard-copy")
        self.futures: dict[str, Future] = {}
        self.copy_seconds: dict[str, float] = {}

    def _validate(self, path: Path, name: str) -> bool:
        data_file = path / "data.mdb"
        expected = self.metadata.get(name, {})
        if not data_file.is_file():
            return False
        expected_bytes = expected.get("data_bytes")
        if expected_bytes is not None and data_file.stat().st_size != int(expected_bytes):
            return False
        expected_entries = expected.get("entries")
        if expected_entries is not None:
            try:
                env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
                try:
                    if env.stat()["entries"] != int(expected_entries):
                        return False
                finally:
                    env.close()
            except lmdb.Error:
                return False
        return True

    def is_ready(self, name: str) -> bool:
        target = self.cache / name
        return (target / "READY").is_file() and self._validate(target, name)

    def _copy(self, name: str, rate_limit_mib_s: float = 0.0) -> Path:
        source = (self.source / name).resolve()
        target = (self.cache / name).resolve()
        if source.parent != self.source or target.parent != self.cache:
            raise ValueError(f"Unsafe shard name: {name}")
        if self.is_ready(name):
            print(f"CACHE_HIT name={name}", flush=True)
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
        rate_limit = rate_limit_mib_s * 2**20
        chunk_size = (1 if rate_limit else 8) * 1024 * 1024
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
                    chunk = source_stream.read(chunk_size)
                    if not chunk:
                        break
                    target_stream.write(chunk)
                    copied += len(chunk)
                    now = time.perf_counter()
                    if rate_limit:
                        expected = (copied - initial) / rate_limit
                        actual = now - started
                        if expected > actual:
                            time.sleep(expected - actual)
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
        if not self._validate(temporary, name):
            raise RuntimeError(f"Shard validation failed: {name}")
        (temporary / "READY").write_text("ready\n", encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        temporary.replace(target)
        duration = time.perf_counter() - started
        self.copy_seconds[name] = duration
        print(f"CACHE_READY name={name} copy_s={duration:.1f}", flush=True)
        return target

    def _copy_with_retry(self, name: str, rate_limit_mib_s: float = 0.0) -> Path:
        for attempt in range(1, self.retries + 1):
            try:
                return self._copy(name, rate_limit_mib_s)
            except Exception as error:
                print(f"CACHE_RETRY name={name} attempt={attempt}/{self.retries} error={error!r}", flush=True)
                if attempt == self.retries:
                    raise
                time.sleep(min(2**attempt, 10))
        raise AssertionError("unreachable")

    def prefetch(self, name: str) -> None:
        if name not in self.futures and not self.is_ready(name):
            print(f"PREFETCH_START name={name} limit_mib_s={self.prefetch_mib_s:.1f}", flush=True)
            self.futures[name] = self.pool.submit(self._copy_with_retry, name, self.prefetch_mib_s)

    def get(self, name: str) -> Path:
        future = self.futures.pop(name, None)
        waited = 0.0
        if future and not future.done():
            waiting = time.perf_counter()
            print(f"CACHE_STARVATION name={name}", flush=True)
            path = future.result()
            waited = time.perf_counter() - waiting
            print(f"CACHE_WAIT_DONE name={name} wait_s={waited:.1f}", flush=True)
        else:
            path = future.result() if future else self._copy_with_retry(name)
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
    shard_metadata = {item["name"]: item for item in shards}
    shard_metadata[validation_name] = manifest["validation_shard"]
    for item in shards:
        print(
            f"SHARD_INFO name={item['name']} clips={item.get('samples', -1)} "
            f"entries={item.get('entries', -1)} size_gib={item.get('data_bytes', 0) / 2**30:.2f}",
            flush=True,
        )
    print(
        f"SHARD_TOTAL count={len(shards)} clips={sum(int(x.get('samples', 0)) for x in shards)} "
        f"entries={sum(int(x.get('entries', 0)) for x in shards)} "
        f"size_gib={sum(int(x.get('data_bytes', 0)) for x in shards) / 2**30:.2f}",
        flush=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    last_path = args.output / "last.pt"
    best_path = args.output / "best.pt"
    metrics_path = args.output / "metrics.jsonl"

    raw_model = build_rrn(args.upstream_dir, args.scale, args.channels, args.blocks).to(device)
    model = raw_model
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scaler = make_scaler(amp)
    criterion = nn.L1Loss(reduction="sum")

    epoch = 1
    shard_position = 0
    next_batch = 0
    global_step = 0
    best_psnr = float("-inf")
    last_validation_psnr = float("nan")
    last_loss = float("nan")
    checkpoint_step = 0
    if args.resume and args.resume.is_file():
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        epoch = int(checkpoint["epoch"])
        shard_position = int(checkpoint.get("shard_position", 0))
        next_batch = int(checkpoint.get("next_batch", 0))
        global_step = int(checkpoint["global_step"])
        best_psnr = float(checkpoint.get("best_psnr", float("-inf")))
        last_validation_psnr = float(checkpoint.get("validation_psnr", float("nan")))
        last_loss = float(checkpoint.get("loss", float("nan")))
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if use_cuda and checkpoint.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        checkpoint_step = global_step
        print(f"RESUMED epoch={epoch} shard_position={shard_position} next_batch={next_batch} step={global_step}", flush=True)

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")
        model = torch.compile(raw_model, mode=args.compile_mode)
        print(f"TORCH_COMPILE mode={args.compile_mode}", flush=True)

    config = {**vars(args), "shards_root": str(args.shards_root), "cache_dir": str(args.cache_dir), "output": str(args.output), "device_resolved": str(device), "amp_resolved": amp}
    (args.output / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    metrics_buffer: list[str] = []

    def flush_metrics() -> None:
        if not metrics_buffer:
            return
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write("".join(metrics_buffer))
        metrics_buffer.clear()

    stager = ShardStager(
        args.shards_root, args.cache_dir, shard_metadata,
        args.shard_prefetch_mib_s, args.copy_retries,
    )
    validation_path = stager.get(validation_name)
    validation_dataset = VimeoSeptupletDataset(
        validation_path, split="test", scale=args.scale, crop_size=0,
        augment=False, decoder=args.decoder,
    )
    print(f"INPUT_PIPELINE decoder={validation_dataset.decoder} workers={args.workers} prefetch_factor={args.prefetch_factor}", flush=True)
    close_dataset(validation_dataset)

    def payload(next_epoch: int, next_shard: int, next_batch_value: int) -> dict:
        return {
            "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": next_epoch, "shard_position": next_shard,
            "next_batch": next_batch_value, "global_step": global_step,
            "loss": last_loss, "best_psnr": best_psnr,
            "validation_psnr": last_validation_psnr, "config": config,
            "epoch_shard_order": shard_order(len(shards), args.seed, next_epoch),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if use_cuda else None,
        }

    stop = False
    try:
        while epoch <= args.epochs and not stop:
            order = shard_order(len(shards), args.seed, epoch)
            if epoch == 1 and not args.no_prefetch:
                startup = min(max(args.startup_buffer_shards, 1), len(order) - shard_position)
                print(
                    f"STARTUP_BUFFER count={startup} resume_position={shard_position} "
                    f"order={','.join(str(x) for x in order)}",
                    flush=True,
                )
                for position in range(shard_position, shard_position + startup):
                    stager.get(shards[order[position]]["name"])
            epoch_batches = sum(int(item["samples"]) // args.batch_size for item in shards)
            while shard_position < len(order):
                shard_index = order[shard_position]
                shard_name = shards[shard_index]["name"]
                shard_path = stager.get(shard_name)
                if not args.no_prefetch and args.shard_prefetch_ahead > 0:
                    future_position = shard_position + max(args.startup_buffer_shards, 1)
                    if future_position < len(order):
                        stager.prefetch(shards[order[future_position]]["name"])
                shard_training_started = time.perf_counter()
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
                    "pin_memory": bool(use_cuda and args.pin_memory),
                    "drop_last": True,
                    "persistent_workers": bool(args.workers > 0 and args.persistent_workers),
                }
                if args.workers > 0:
                    loader_options["prefetch_factor"] = args.prefetch_factor
                loader = DataLoader(dataset, **loader_options)
                total_batches = len(dataset) // args.batch_size
                completed_epoch_batches = sum(
                    int(shards[order[position]]["samples"]) // args.batch_size
                    for position in range(shard_position)
                )
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
                    profile = bool(use_cuda and args.profile_every > 0 and (global_step + 1) % args.profile_every == 0)
                    if profile:
                        torch.cuda.synchronize()
                        transfer_started = time.perf_counter()
                    if args.gpu_downsample:
                        target, names = batch_data
                        target = target.to(device, non_blocking=use_cuda)
                        low_resolution = gaussian_downsample(target, args.scale)
                        low_resolution = torch.cat((low_resolution[:, :, 1:2], low_resolution), dim=2)
                    else:
                        low_resolution, target, names = batch_data
                        low_resolution = low_resolution.to(device, non_blocking=use_cuda)
                        target = target.to(device, non_blocking=use_cuda)
                    h2d_seconds = float("nan")
                    if profile:
                        torch.cuda.synchronize()
                        h2d_seconds = time.perf_counter() - transfer_started
                        compute_started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(device, amp):
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
                    gpu_compute_seconds = float("nan")
                    if profile:
                        torch.cuda.synchronize()
                        gpu_compute_seconds = time.perf_counter() - compute_started
                    global_step += 1
                    last_loss = float(loss.detach().cpu())
                    finished = time.perf_counter()
                    record = {
                        "epoch": epoch, "shard_position": shard_position,
                        "shard": shard_name, "batch": batch_index + 1,
                        "epoch_batch": completed_epoch_batches + batch_index + 1,
                        "epoch_batches": epoch_batches,
                        "global_step": global_step, "loss": last_loss,
                        "mae": float(mae.cpu()),
                        "lr": optimizer.param_groups[0]["lr"],
                        "seconds": finished - started,
                        "data_seconds": started - previous_step_finished,
                        "wall_seconds": finished - previous_step_finished,
                        "h2d_seconds": h2d_seconds,
                        "gpu_compute_seconds": gpu_compute_seconds,
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
                            f"epoch_batch={record['epoch_batch']}/{epoch_batches} "
                            f"loss={last_loss:.4f} mae={record['mae']:.6f} "
                            f"lr={record['lr']:.3e} val_psnr={last_validation_psnr:.4f} "
                            f"best_psnr={best_psnr:.4f} checkpoint_step={checkpoint_step} "
                            f"data_s={record['data_seconds']:.3f} compute_s={record['seconds']:.3f} "
                            f"wall_s={record['wall_seconds']:.3f} "
                            f"samples_s={args.batch_size / max(record['wall_seconds'], 1e-6):.1f} "
                            f"gpu_mib={gpu_mib:.0f}",
                            flush=True,
                        )
                    if profile:
                        profiled_step = max(record["data_seconds"] + h2d_seconds + gpu_compute_seconds, 1e-9)
                        print(
                            f"PROFILE step={global_step} data_s={record['data_seconds']:.4f} "
                            f"h2d_s={h2d_seconds:.4f} gpu_compute_s={gpu_compute_seconds:.4f} "
                            f"step_s={profiled_step:.4f} data_wait_ratio={record['data_seconds'] / profiled_step:.4f}",
                            flush=True,
                        )
                    reached_limit = args.max_steps is not None and global_step >= args.max_steps
                    if global_step % args.save_every == 0 or reached_limit:
                        flush_metrics()
                        save_checkpoint(last_path, payload(epoch, shard_position, batch_index + 1))
                        checkpoint_step = global_step
                        print(f"CHECKPOINT step={checkpoint_step} path={last_path}", flush=True)
                    if reached_limit:
                        stop = True
                        break
                progress.close()
                close_dataset(dataset)
                if stop:
                    break

                next_batch = 0
                shard_position += 1
                flush_metrics()
                save_checkpoint(last_path, payload(epoch, shard_position, 0))
                checkpoint_step = global_step
                print(f"CHECKPOINT step={checkpoint_step} path={last_path}", flush=True)
                train_seconds = time.perf_counter() - shard_training_started
                copy_seconds = stager.copy_seconds.get(shard_name, 0.0)
                print(
                    f"SHARD_COMPLETE name={shard_name} train_s={train_seconds:.1f} copy_s={copy_seconds:.1f} "
                    f"validation_psnr={last_validation_psnr:.4f} best_psnr={best_psnr:.4f}",
                    flush=True,
                )
                if args.validation_scope == "shard":
                    close_dataset(validation_dataset)
                    validation_psnr = validate_psnr(model, validation_dataset, device, amp)
                    last_validation_psnr = validation_psnr
                    close_dataset(validation_dataset)
                    improved = validation_psnr > best_psnr
                    if improved:
                        best_psnr = validation_psnr
                    flush_metrics()
                    save_checkpoint(last_path, payload(epoch, shard_position, 0))
                    checkpoint_step = global_step
                    print(
                        f"VALIDATION_COMPLETE scope=shard epoch={epoch} shard={shard_position}/{len(order)} "
                        f"psnr={validation_psnr:.4f} best_psnr={best_psnr:.4f}",
                        flush=True,
                    )
                    if improved:
                        best_payload = payload(epoch, shard_position, 0)
                        best_payload["validation_psnr"] = validation_psnr
                        save_checkpoint(best_path, best_payload)
                if copy_seconds and train_seconds < copy_seconds:
                    print(
                        f"CACHE_WARNING name={shard_name} train_s={train_seconds:.1f} copy_s={copy_seconds:.1f} "
                        "message=GPU_may_catch_up_with_downloader",
                        flush=True,
                    )
                if not args.keep_cache:
                    stager.discard(shard_name)

            if stop:
                close_dataset(validation_dataset)
                validation_psnr = validate_psnr(model, validation_dataset, device, amp)
                last_validation_psnr = validation_psnr
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

            improved = False
            if args.validation_scope == "epoch":
                close_dataset(validation_dataset)
                validation_psnr = validate_psnr(model, validation_dataset, device, amp)
                last_validation_psnr = validation_psnr
                close_dataset(validation_dataset)
                improved = validation_psnr > best_psnr
                if improved:
                    best_psnr = validation_psnr
                print(
                    f"VALIDATION_COMPLETE scope=epoch epoch={epoch} "
                    f"psnr={validation_psnr:.4f} best_psnr={best_psnr:.4f}",
                    flush=True,
                )
            scheduler.step()
            epoch += 1
            shard_position = 0
            next_batch = 0
            save_checkpoint(last_path, payload(epoch, 0, 0))
            checkpoint_step = global_step
            if improved:
                save_checkpoint(best_path, payload(epoch, 0, 0))
            print(
                f"EPOCH_COMPLETE epoch={epoch - 1} validation_psnr={last_validation_psnr:.4f} "
                f"best_psnr={best_psnr:.4f} checkpoint_step={checkpoint_step}",
                flush=True,
            )
    finally:
        flush_metrics()
        close_dataset(validation_dataset)
        stager.close()
    print(f"TRAINING_STOPPED_AT_STEP={global_step} last={last_path} best={best_path}", flush=True)


if __name__ == "__main__":
    main()
