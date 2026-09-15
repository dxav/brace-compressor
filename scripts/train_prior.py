#!/usr/bin/env python3
"""Train the transformer base-prior on HOAPS wvpa masked-cell prediction (T037).

Self-supervised task: sample a missingness mask, hide the valid values,
and ask the network to predict the hidden values from the 6-channel
mask context. Loss = MSE (optionally Huber). This matches the inference
role of ``TransformerPredictor.base_prior`` exactly (mask -> values).

The script exports the optimized weights via
``TransformerPredictor.serialize_weights()`` to a versioned ``.hwpm``
blob that ``TransformerPredictor.load_weights()`` can restore.

Run from the repository root with the project venv:

    .venv/bin/python scripts/train_prior.py --input data/wvpa_2020-08-01_07.npy
    .venv/bin/python scripts/train_prior.py --input data/wvpa_2020-08-01_07.npy --epochs 40 --out src/hoaps_compressor/model/weights/prior_v3.hwpm

Without ``--input`` the script generates a deterministic synthetic
smooth space-time field (so it is runnable in a fresh checkout with no
real HOAPS data), per the data-prerequisite note in tasks.md T037.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make `src/` importable when running straight from a checkout.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hoaps_compressor.model.transformer import TransformerPredictor  # noqa: E402


def generate_synthetic_field(
    shape: tuple[int, int, int], seed: int, missing_frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth space-time field with a land-strip + scattered missing mask."""
    t, lat, lon = shape
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 2.0 * np.pi, lon)
    y = np.linspace(0.0, np.pi, lat)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    field = np.empty(shape, dtype=np.float32)
    for ti in range(t):
        phase = 0.35 * ti
        base = 25.0 + 12.0 * np.sin(yy + phase) * np.cos(xx - 0.5 * phase)
        detail = rng.normal(0.0, 0.8, size=(lat, lon))
        field[ti] = (base + detail).astype(np.float32)

    mask = np.zeros(shape, dtype=bool)
    mask[:, : max(1, lat // 8), :] = True  # "land" strip
    rng2 = np.random.default_rng(seed + 1)
    mask |= rng2.random(shape) < missing_frac
    return field, mask


def load_field(path: str | None, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Load a real .npy field (or synthesize one) and return (field, mask)."""
    if path is None:
        return generate_synthetic_field(shape, seed=7, missing_frac=0.3)
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected 2-D or 3-D .npy, got {arr.ndim}-D")
    field = arr.astype(np.float32)
    mask = np.isnan(field)
    return field, mask


def coarse_downsample(
    field: np.ndarray, mask: np.ndarray, max_cells: int = 8192
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample (field, mask) to a coarse grid of <= max_cells cells.

    Mirrors ``TransformerPredictor._coarse_shape`` so training runs on the
    same resolution the base prior uses at inference (the prior is a smooth
    cold-start hint; full-resolution training is O(N^2) and impractical).
    """
    from hoaps_compressor.model.transformer import TransformerPredictor

    t, lat, lon = field.shape
    coarse = TransformerPredictor._coarse_shape((t, lat, lon), max_cells)
    if coarse == (t, lat, lon):
        return field, mask
    ct, clat, clon = coarse
    bt, bl, bo = t // ct, lat // clat, lon // clon
    # Block-mean the field over valid cells only; missing cells excluded.
    out = np.empty(coarse, dtype=np.float32)
    out_mask = np.empty(coarse, dtype=bool)
    for ti in range(ct):
        for yi in range(clat):
            for xi in range(clon):
                blk = field[ti * bt : (ti + 1) * bt, yi * bl : (yi + 1) * bl, xi * bo : (xi + 1) * bo]
                blk_mask = mask[ti * bt : (ti + 1) * bt, yi * bl : (yi + 1) * bl, xi * bo : (xi + 1) * bo]
                valid = blk[~blk_mask]
                if valid.size:
                    out[ti, yi, xi] = float(valid.mean())
                    out_mask[ti, yi, xi] = False
                else:
                    out[ti, yi, xi] = 0.0
                    out_mask[ti, yi, xi] = True
    return out, out_mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=None, help="path to .npy HOAPS field (optional)")
    ap.add_argument("--shape", nargs=3, type=int, default=[8, 90, 180],
                    help="synthetic shape when --input is absent")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--max-cells", type=int, default=8192,
                    help="coarse-grid token cap for training (matches base_prior)")
    ap.add_argument("--huber", action="store_true", help="use Huber loss instead of MSE")
    ap.add_argument("--out", default=str(ROOT / "src/hoaps_compressor/model/weights/prior_v3.hwpm"),
                    help="output .hwpm weights path")
    args = ap.parse_args()

    import torch

    field, mask = load_field(args.input, tuple(args.shape))
    field, mask = coarse_downsample(field, mask, args.max_cells)
    t, lat, lon = field.shape
    valid = ~mask
    n_valid = int(valid.sum())
    print(f"coarse field shape={field.shape} valid={n_valid} ({100.0*n_valid/field.size:.1f}%)")

    # Normalize target values to the [0, 64] band the prior is rescaled to.
    vmin, vmax = float(np.nanmin(field)), float(np.nanmax(field))
    span = (vmax - vmin) or 1.0
    target = (field - vmin) / span * 64.0  # [0, 64]

    pred = TransformerPredictor()
    model = pred.model
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.HuberLoss() if args.huber else torch.nn.MSELoss()

    # Precompute the 6-channel context once (mask-only, decode-consistent).
    ctx = pred._context(mask)  # [T, lat, lon, 6]
    feats = torch.from_numpy(ctx.reshape(-1, 6))
    pos = pred._positional(feats.shape[0], model.d_model)
    y = torch.from_numpy(target.reshape(-1).astype(np.float32))
    valid_idx = torch.from_numpy(np.flatnonzero(valid.reshape(-1)))

    n = valid_idx.numel()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(epoch))
        total = 0.0
        nbatch = 0
        for start in range(0, n, args.batch):
            idx = valid_idx[perm[start : start + args.batch]]
            opt.zero_grad()
            with torch.no_grad():
                h = model.in_proj(feats[idx]) + pos[idx]
            out = model.encoder(h)
            out = model.out_proj(out).squeeze(-1) * torch.tanh(model.out_scale)
            # base_prior rescales (out+1)*32; invert for the loss.
            pred_val = (out + 1.0) * 32.0
            loss = loss_fn(pred_val, y[idx])
            loss.backward()
            opt.step()
            total += float(loss.item()) * idx.numel()
            nbatch += idx.numel()
        print(f"epoch {epoch:3d}  loss={total/max(nbatch,1):.4f}")

    model.eval()
    blob = pred.serialize_weights()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    print(f"wrote {len(blob)} bytes -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())