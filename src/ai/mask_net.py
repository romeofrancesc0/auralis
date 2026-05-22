"""Lightweight CNN for refined T-F mask estimation.

Takes the STFT log-magnitude spectrogram, the per-frame attention weights from
the GMM+MLP+HMM module, and the NMF-guided IRM as a 3-channel input, and
outputs a refined T-F mask in [0, 1].

Architecture: 5 fully-convolutional layers (~75K parameters).
Runs on CPU, CUDA, or Apple MPS (M-series chips) without code changes.
Train on GPU (see train_mask_net.py); inference runs on any device.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    class _CNN(nn.Module):
        """Fully-convolutional mask estimator (5 Conv2d layers, ~75K params).

        All 3×3 convolutions use same-padding so the (F, T) resolution is
        preserved throughout. A final 1×1 conv collapses channels to one mask.
        """

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x).squeeze(1)  # (B, F, T)

        @property
        def n_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

    _TORCH_AVAILABLE = True
except ImportError:
    pass


def _check_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for MaskNet. "
            "Install with: pip install torch  (or: pip install 'auralis[torch]')"
        )


def _auto_device() -> str:
    """Select CUDA > MPS > CPU in order of availability."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MaskNet:
    """Inference wrapper around _CNN.

    Handles device selection, numpy I/O, and input normalization so
    callers only deal with plain numpy arrays.
    """

    def __init__(self, device: str | None = None) -> None:
        _check_torch()
        self._device = device or _auto_device()
        self._model = _CNN().to(self._device)
        self._model.eval()
        logger.debug("MaskNet ready on %s  (%d params)", self._device, self._model.n_params)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def refine(
        self,
        magnitude: np.ndarray,
        attention_weights: np.ndarray,
        nmf_irm: np.ndarray,
    ) -> np.ndarray:
        """Refine the NMF-IRM with the learned CNN.

        Args:
            magnitude:         STFT magnitude, shape (n_freqs, n_frames)
            attention_weights: per-frame attention in [0, 1], shape (n_frames,)
            nmf_irm:           NMF-guided IRM in [0, 1], shape (n_freqs, n_frames)

        Returns:
            Refined mask in [0, 1], shape (n_freqs, n_frames)
        """
        x = build_input(magnitude, attention_weights, nmf_irm)  # (3, F, T)
        x_t = torch.from_numpy(x).unsqueeze(0).to(self._device)  # (1, 3, F, T)
        with torch.no_grad():
            out = self._model(x_t)            # (1, F, T)
        return out.squeeze(0).cpu().numpy()   # (F, T)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save(self._model.state_dict(), path)
        logger.info("MaskNet saved → %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "MaskNet":
        net = cls(device=device)
        state = torch.load(path, map_location=net._device, weights_only=True)
        net._model.load_state_dict(state)
        net._model.eval()
        logger.info("MaskNet loaded from %s  (device: %s)", path, net._device)
        return net


def build_input(
    magnitude: np.ndarray,
    attention_weights: np.ndarray,
    nmf_irm: np.ndarray,
) -> np.ndarray:
    """Build the 3-channel float32 input tensor for the CNN.

    Channel 0 — log-magnitude (zero-mean, unit-variance): spectral energy.
    Channel 1 — attention weights broadcast to (F, T): temporal F/M signal.
    Channel 2 — NMF-IRM in [0, 1]: frequency-resolved classical-pipeline prior.
    """
    n_freqs, n_frames = magnitude.shape

    log_mag = np.log1p(magnitude).astype(np.float32)
    mu, sigma = log_mag.mean(), log_mag.std() + 1e-8
    log_mag = (log_mag - mu) / sigma

    n_attn = min(len(attention_weights), n_frames)
    attn = np.zeros(n_frames, dtype=np.float32)
    attn[:n_attn] = attention_weights[:n_attn].astype(np.float32)
    attn_2d = np.tile(attn[np.newaxis, :], (n_freqs, 1))  # (F, T)

    return np.stack([log_mag, attn_2d, nmf_irm.astype(np.float32)], axis=0)  # (3, F, T)
