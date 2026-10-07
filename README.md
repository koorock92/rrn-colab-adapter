# RRN Colab training adapter

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/koorock92/rrn-colab-adapter/blob/main/notebooks/RRN_Sharded_GPU_Training.ipynb)

This repository contains Colab orchestration, a sharded Vimeo-90K LMDB reader, and resumable training utilities developed for a personal research workflow.

It does **not** redistribute the upstream RRN implementation, Vimeo-90K, LMDB shards, pretrained weights, or checkpoints. At setup time the notebook clones the official upstream repository separately:

- Upstream project: https://github.com/junpan19/RRN
- Paper: *Revisiting Temporal Modeling for Video Super-resolution* (BMVC 2020)

The upstream repository did not include a license when this adapter was prepared. Its source remains subject to its authors' terms and copyright. The license in this repository applies only to the original adapter files here.

## Workflow

1. Keep Vimeo-90K LMDB shards and checkpoints in Google Drive.
2. Clone this adapter and the upstream project into the Colab VM.
3. Stage two LMDB shards into /content before training and retain all completed shards in the VM cache.
4. Save last.pt for exact resume and best.pt by validation PSNR.
5. Run the resumable 300-step smoke test before enabling full training.

The corrected-crop notebook uses a 64x64 low-resolution training crop, which
corresponds to a 256x256 high-resolution crop for x4 RRN before the degradation
border is added. This is intentionally different from the legacy
`--crop-size 64` command, which meant a 64x64 HR crop. New commands should use
`--lr-crop-size 64`; `--crop-size` remains only for backward compatibility.

For an 8-vCPU Colab VM the notebook defaults to six DataLoader workers and a
prefetch factor of four, leaving two cores for the main process, shard copier,
and notebook. It uses paper-equivalent batch size 4, AMP, background shard
prefetch, and checkpointing to Google Drive every 1,000 steps. The corrected
run is written to a new output directory and initialized from the previous
`best.pt`, so the completed run remains untouched.

Validation defaults to once per epoch (`--validation-scope epoch`) rather than
once per shard. Shard boundaries still save `last.pt`, so reducing validation
frequency does not reduce resume safety. Use `--validation-scope shard` only
when per-shard PSNR is specifically needed.

The input pipeline uses a native PNG decoder when available (`pyspng` first,
then `torchvision.io`, with Pillow as a compatibility fallback). DataLoader
workers keep four batches prefetched per worker by default. Use
`src/benchmark_input_pipeline.py` to compare decoders on a local shard before a
long Colab run.

`src/benchmark_rrn_batches.py` measures batches 1/2/4/8 with the corrected crop
without modifying
the saved checkpoint. It can also benchmark `torch.compile` with `--compile`;
the main trainer supports the same optional flag while always saving the
uncompiled model state for checkpoint compatibility.
