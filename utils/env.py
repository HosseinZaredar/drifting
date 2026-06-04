"""Global paths for the public Drift release."""

from __future__ import annotations

import os

IMAGENET_PATH = "/dev/shm/latent_cache"
IMAGENET_CACHE_PATH = "/dev/shm/latent_cache"
IMAGENET_FID_NPZ = "/dev/shm/stats/imagenet_256_fid_stats.npz"
IMAGENET_PR_NPZ = "/dev/shm/stats/imagenet_val_prc_arr0.npz"

HF_REPO_ID = "Goodeat/drifting"
HF_ROOT = os.environ.get("HF_ROOT", "~/hf_cache")
