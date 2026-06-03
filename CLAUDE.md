# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Official JAX codebase for *Generative Modeling via Drifting* — one-step class-conditional image generation on ImageNet 256×256. The method trains a generator by minimizing a particle-based drift loss that moves generated samples toward the data distribution stored in a class-wise memory bank.

## Environment Setup

```bash
conda create -n drifting-release python=3.10 -y
conda activate drifting-release
pip install -r requirements.txt
export JAX_PLATFORMS=tpu,cpu   # keep TPU as default while exposing CPU for Flax VAE paths
```

**Before running anything:** set paths in `utils/env.py`:
- `IMAGENET_PATH` — root with `train/` and `val/` subdirectories
- `IMAGENET_CACHE_PATH` — root for precomputed VAE latent `.pt` files (latent-space models only)
- `IMAGENET_FID_NPZ` / `IMAGENET_PR_NPZ` — FID/precision-recall reference stats
- `HF_ROOT` — local cache dir for HuggingFace artifacts (auto-downloaded)

## Common Commands

### Inference / FID Evaluation
```bash
python inference.py --init-from "hf://latent_L_sota" --cfg-scale 1.0 \
  --num-samples 50000 --eval-batch-size 256 --json-out results.json
```
`--init-from` accepts `hf://<model_id>` (downloads from HuggingFace) or a local workdir path.

### Generator Training
```bash
python main.py --gen --config configs/gen/latent_sota_L.yaml --workdir runs/gen_latent_sota_L
```

### MAE Pretraining (optional — pretrained weights are available via `hf://`)
```bash
python main.py --config configs/mae/latent_640.yaml --workdir runs/mae_latent_640
```

### Build Latent Cache (latent-space generators only)
```bash
python -m dataset.latent \
  --data-path /path/to/imagenet --target-path /path/to/latent_cache \
  --local-batch-size 128 --num-workers 8 --pin-memory
```

## Architecture

### Training Pipeline (`main.py` → `train.py` / `train_mae.py`)
- `main.py` is the single CLI entrypoint; `--gen` selects generator training, otherwise MAE pretraining.
- `train.py` contains the generator training loop: `train_step` builds CFG-conditioned samples, computes `drift_loss`, and updates the `TrainState` (Flax's `train_state.TrainState` extended with EMA params).
- Checkpoints saved via Orbax; EMA-only artifacts go to `<workdir>/params_ema/ema_params.msgpack`.

### Core Loss (`drift_loss.py`)
`drift_loss(gen, fixed_pos, fixed_neg, ...)` computes the drifting objective:
- Operates on feature-space tokens `[B, C, S]`.
- Scales pairwise distances and applies multi-temperature softmax kernels (`R_list`) to produce attractive/repulsive forces.
- Gradients flow only through the `gen` term; `fixed_pos`/`fixed_neg` are stop-gradient.

### Memory Bank (`memory_bank.py`)
`ArrayMemoryBank` — per-class ring buffer (numpy on CPU). Each training host maintains its own bank. `add(samples, labels)` inserts; `sample(labels, n_samples)` draws and returns a `jnp.ndarray`. Increasing `push_per_step` compensates for fewer hosts.

### Models
- `models/generator.py` — Transformer-based conditional generator with sincos positional embeddings and classifier-free guidance.
- `models/mae_model.py` — MAE-ResNet feature extractor (ResNet backbone, GroupNorm). Used frozen during generator training.
- `models/convnext.py` — optional ConvNeXt feature extractor.
- `models/hf.py` — HuggingFace download/load utilities for pretrained artifacts.

### Dataset (`dataset/`)
- `dataset/dataset.py` — `create_imagenet_split(resolution, use_aug, use_latent, use_cache, ...)` returns `(loader, preprocess_fn, postprocess_fn)`. `use_cache=True` reads prebuilt latent `.pt` files instead of encoding online.
- `dataset/latent.py` — offline latent cache builder.
- `dataset/vae.py` — Stable Diffusion VAE encode/decode (runs on CPU via PyTorch+diffusers).

### Distributed Sharding (`utils/hsdp_util.py`)
Uses JAX's `Mesh` with `data` and `fsdp` axes. `set_global_mesh(hsdp_dim)` initialises the mesh; all sharding utils (`data_shard`, `ddp_shard`, `map_to_sharding`, `enforce_ddp`) operate relative to this global mesh.

### Config System (`utils/misc.py`)
Configs are `ml-collections` YAML files under `configs/gen/` and `configs/mae/`. Loaded with `load_config(path)` and accessed as attribute dicts.

## Pretrained Models (HuggingFace `Goodeat/drifting`)

| HF ID | Description |
|---|---|
| `hf://latent_L_sota` | Drift-L latent, FID 1.53 |
| `hf://latent_B_sota` | Drift-B latent, FID 1.74 |
| `hf://pixel_L_sota` | Drift-L pixel, FID 1.62 |
| `hf://pixel_B_sota` | Drift-B pixel, FID 1.73 |
| `hf://mae_latent_640` | MAE-640 feature extractor (latent) |
| `hf://mae_pixel_640` | MAE-640 feature extractor (pixel) |
