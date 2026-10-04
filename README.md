# RRN Colab training adapter

This repository contains Colab orchestration, a sharded Vimeo-90K LMDB reader, and resumable training utilities developed for a personal research workflow.

It does **not** redistribute the upstream RRN implementation, Vimeo-90K, LMDB shards, pretrained weights, or checkpoints. At setup time the notebook clones the official upstream repository separately:

- Upstream project: https://github.com/junpan19/RRN
- Paper: *Revisiting Temporal Modeling for Video Super-resolution* (BMVC 2020)

The upstream repository did not include a license when this adapter was prepared. Its source remains subject to its authors' terms and copyright. The license in this repository applies only to the original adapter files here.

## Workflow

1. Keep Vimeo-90K LMDB shards and checkpoints in Google Drive.
2. Clone this adapter and the upstream project into the Colab VM.
3. Stage one LMDB shard into /content before training.
4. Save last.pt for exact resume and best.pt by validation PSNR.
5. Run the 20-step smoke test before enabling full training.
