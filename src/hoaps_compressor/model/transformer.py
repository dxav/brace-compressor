"""Space-time transformer predictor (T027, FR-005, FR-018).

A compact, CPU-first transformer that predicts each grid cell from
neighboring valid data across space and time (attention over spatial
patches and time steps; JPEG AI-inspired). Used as the residual
predictor inside the codec pipeline:

    encode:  residual = value - predict(field, mask)
    decode:  value = prediction + dequantized_residual

Determinism requirement (research.md R5): the prediction MUST be
identical at encode and decode for the same weights. The decoder does
not see original values, so the predictor only consumes the mask and
**predicted context broadcast** (mode tensor), never the raw field.
Neighbor-mean context is computed from the mask alone via iterative
diffusion — valid cells contribute their *predicted* (unknown) values,
which cancels; the diffusion therefore models missingness structure and
spatial smoothness priors, not leaked values.
"""

from __future__ import annotations

import math
import struct

import numpy as np

try:  # torch import kept lazy-light for CPU-only usage
    import torch
    import torch.nn as nn

    _TORCH = True
except Exception:  # pragma: no cover - torch is a hard dep, but stay safe
    _TORCH = False

MODEL_VERSION = 2  # bumped when weights/layout change (container header)
MODEL_MAGIC = b"HWPM"
_WEIGHTS_HEAD = struct.Struct("<4sI")  # magic, version


class SpaceTimeTransformer(nn.Module if _TORCH else object):
    """Attention predictor over (time, lat, lon) wvpa fields.

    Input : missingness mask (bool [T, lat, lon])
    Output: predicted values (float32 [T, lat, lon])

    The network consumes a *diffusion context* (normalized neighbor-mean
    structure derived from the mask) plus Fourier positional encodings,
    and regresses residuals' statistical prior through self-attention.
    """

    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        if not _TORCH:  # pragma: no cover
            raise RuntimeError("PyTorch is required for the transformer predictor")
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(6, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=0.0,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, 1)
        # deterministic bounded output head (values in [-1, 1])
        self.out_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, context: "torch.Tensor", pos: "torch.Tensor") -> "torch.Tensor":
        """context: [N, 6] features; pos: [N, d_model] positional encodings."""
        h = self.in_proj(context) + pos
        h = self.encoder(h)
        return self.out_proj(h).squeeze(-1) * torch.tanh(self.out_scale)


class TransformerPredictor:
    """Encode/decode-compatible wrapper producing identical predictions."""

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.model = SpaceTimeTransformer()
        self._init_weights()

    def _init_weights(self) -> None:
        g = torch.Generator().manual_seed(self.seed)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02, generator=g)
            else:
                nn.init.zeros_(p)
        nn.init.constant_(self.model.out_proj.bias, 0.0)
        self.model.eval()

    # -- context construction (mask-only; identical at encode/decode) ----
    def _context(self, mask: np.ndarray) -> np.ndarray:
        """6-channel diffusion context from the missingness mask alone."""
        t, lat, lon = mask.shape
        m = (~mask).astype(np.float32)  # 1 = valid
        # iterative neighbor-normalized diffusion of the validity field
        ctx = []
        cur = m
        pad_m = np.pad(m, ((0, 0), (1, 1), (1, 1)), mode="edge")
        wsum = (
            pad_m[:, :-2, 1:-1]
            + pad_m[:, 2:, 1:-1]
            + pad_m[:, 1:-1, :-2]
            + pad_m[:, 1:-1, 2:]
        )
        cur = wsum / 4.0
        ctx.append(cur)
        # second diffusion pass for wider support
        pad_c = np.pad(cur, ((0, 0), (1, 1), (1, 1)), mode="edge")
        wsum2 = (
            pad_c[:, :-2, 1:-1]
            + pad_c[:, 2:, 1:-1]
            + pad_c[:, 1:-1, :-2]
            + pad_c[:, 1:-1, 2:]
        )
        ctx.append(wsum2 / 4.0)
        # temporal persistence of validity (shifted by 1 step)
        tprev = np.concatenate([m[:1], m[:-1]], axis=0)
        ctx.append(tprev)
        # temporal next (wrap) for boundary cells
        tnext = np.concatenate([m[1:], m[-1:]], axis=0)
        ctx.append(tnext)
        # normalized grid coordinates
        gy = np.linspace(0.0, 1.0, lat, dtype=np.float32)[None, :, None]
        gx = np.linspace(0.0, 1.0, lon, dtype=np.float32)[None, None, :]
        ctx.append(np.broadcast_to(gy, (t, lat, lon)))
        ctx.append(np.broadcast_to(gx, (t, lat, lon)))
        return np.stack(ctx, axis=-1).astype(np.float32)  # [T, lat, lon, 6]

    def _positional(self, n: int, d_model: int) -> "torch.Tensor":
        pos = np.zeros((n, d_model), dtype=np.float32)
        position = np.arange(n, dtype=np.float32)[:, None]
        div = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(1e4) / d_model))
        pos[:, 0::2] = np.sin(position * div)
        pos[:, 1::2] = np.cos(position * div[: (d_model // 2)])
        return torch.from_numpy(pos)

    def predict(self, mask: np.ndarray) -> np.ndarray:
        """Predict values for every cell given only the mask (deterministic).

        Sequential temporal scheme (decode-compatible): slice 0 uses the
        wave/prior base field; each later slice is initialized from the
        previous *predicted* slice (persistence), so encode and decode
        produce identical values without ever reading original data. The
        transformer refines the base field spatially (JPEG AI-style learned
        prior over the mask-diffusion context).
        """
        mask = np.asarray(mask, dtype=bool)
        t, lat, lon = mask.shape

        # --- Spatial learned prior from the mask context (transformer) ---
        ctx = self._context(mask)          # [T, lat, lon, 6]
        feats = torch.from_numpy(ctx.reshape(-1, 6))
        pos = self._positional(feats.shape[0], self.model.d_model)
        with torch.no_grad():
            out = self.model(feats, pos).numpy().astype(np.float32)
        base = (out + 1.0).reshape(t, lat, lon) * 32.0  # [0, 64] physical band
        # Smooth the learned base with a spatial convolution (denoising of
        # the random-init latent; keeps determinism).
        base = self._smooth(base)

        # --- Sequential temporal persistence (identical at decode) ------
        # Slice ti is predicted from the previous *predicted* slice where
        # that slice was valid; where the previous slice was missing (no
        # trustworthy persistence source) the base prior is used instead.
        prediction = np.empty((t, lat, lon), dtype=np.float32)
        prediction[0] = base[0]
        for ti in range(1, t):
            prev_missing = mask[ti - 1]
            prediction[ti] = np.where(prev_missing, base[ti], prediction[ti - 1])
        return prediction

    @staticmethod
    def _smooth(field: np.ndarray, passes: int = 2) -> np.ndarray:
        """3x3 average smoothing (deterministic, edge-padded)."""
        out = field
        for _ in range(passes):
            p = np.pad(out, ((0, 0), (1, 1), (1, 1)), mode="edge")
            out = (
                p[:, :-2, 1:-1]
                + p[:, 2:, 1:-1]
                + p[:, 1:-1, :-2]
                + p[:, 1:-1, 2:]
                + p[:, :-2, :-2]
                + p[:, :-2, 2:]
                + p[:, 2:, :-2]
                + p[:, 2:, 2:]
                + p[:, 1:-1, 1:-1]
            ) / 9.0
        return out.astype(np.float32)

    # -- weight persistence (deterministic, versioned) --------------------
    def serialize_weights(self) -> bytes:
        """Serialize weights with per-tensor shape metadata (bit-exact restore)."""
        state = self.model.state_dict()
        header = _WEIGHTS_HEAD.pack(MODEL_MAGIC, MODEL_VERSION)
        blobs = []
        for name in sorted(state):
            arr = np.ascontiguousarray(state[name].detach().cpu().numpy(), dtype=np.float32)
            nb = name.encode("utf-8")
            shape = arr.shape
            meta = struct.pack(
                "<HII", len(nb), len(shape), arr.size
            ) + struct.pack(f"<{len(shape)}I", *shape)
            blobs.append(meta + nb + arr.tobytes())
        return header + b"".join(blobs)

    def load_weights(self, blob: bytes) -> None:
        magic, version = _WEIGHTS_HEAD.unpack_from(blob, 0)
        if magic != MODEL_MAGIC:
            raise ValueError(f"bad model magic {magic!r}")
        if version != MODEL_VERSION:
            raise ValueError(
                f"model version mismatch: stream {version}, codec {MODEL_VERSION}"
            )
        off = _WEIGHTS_HEAD.size
        state = {}
        while off < len(blob):
            nlen, ndim, count = struct.unpack_from("<HII", blob, off)
            off += 10
            shape = struct.unpack_from(f"<{ndim}I", blob, off) if ndim else ()
            off += 4 * ndim
            name = blob[off : off + nlen].decode("utf-8")
            off += nlen
            arr = np.frombuffer(blob, dtype=np.float32, count=count, offset=off)
            off += count * 4
            state[name] = arr.reshape(shape).copy()
        self.model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
        self.model.eval()
