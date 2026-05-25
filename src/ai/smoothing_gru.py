"""GRU-based temporal smoothing for the per-frame attention mask.

Replaces the fixed-topology HMM in smoothing.py with a learnable 1-layer
bidirectional GRU that adapts its temporal dynamics from data.

Why bidirectional GRU over HMM
--------------------------------
The HMM uses manually tuned fixed transition probabilities (p_ff=0.95,
p_mf=0.20), which are a good heuristic but cannot adapt to variable speaker
turn durations, overlapping speech, or silence segments.  A bidirectional GRU:
  - Learns optimal temporal smoothing from IBM-labelled frames
  - Integrates context from both past and future frames
  - Preserves the probabilistic (soft) output expected by the hybrid IRM

Architecture (~26K parameters)
---------------------------------
  Input (B, T, 1) → BiGRU(input=1, hidden=64, bidirectional=True)
                  → Linear(128, 1) → Sigmoid
                  → Output (B, T, 1)

The GRUSmoother class exposes a smooth(mask) → mask interface that is a
drop-in replacement for hmm_smooth() in attention.py.
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

    class _GRUNet(nn.Module):
        """1-layer bidirectional GRU for sequence smoothing.

        Operates on normalised raw classifier probabilities as the input
        sequence and outputs smoothed posteriors for the female (target) state.
        """

        def __init__(self, hidden_size: int = 64) -> None:
            super().__init__()
            self.gru    = nn.GRU(1, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            self.linear = nn.Linear(hidden_size * 2, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (B, T, 1) raw mask values in [0, 1]

            Returns:
                (B, T, 1) smoothed mask values in [0, 1]
            """
            h, _ = self.gru(x)          # (B, T, 2*hidden)
            return self.sigmoid(self.linear(h))   # (B, T, 1)

        @property
        def n_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

    _TORCH_AVAILABLE = True
except ImportError:
    pass


def _check_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for GRUSmoother. "
            "Install with: pip install torch  (or: pip install 'auralis[torch]')"
        )


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GRUSmoother:
    """Inference wrapper around _GRUNet.

    Exposes a smooth(mask) interface compatible with hmm_smooth() so it can
    be used as a drop-in replacement in AttentionModule.compute_mask().
    """

    def __init__(self, device: str | None = None) -> None:
        _check_torch()
        self._device = device or _auto_device()
        self._model  = _GRUNet().to(self._device)
        self._model.eval()
        logger.debug(
            "GRUSmoother ready on %s  (%d params)", self._device, self._model.n_params
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def smooth(self, mask: np.ndarray) -> np.ndarray:
        """Smooth a per-frame attention mask with the learned GRU.

        Args:
            mask: per-frame P(female), shape (n_frames,), values in [0, 1]

        Returns:
            smoothed: shape (n_frames,), values in [0, 1]
        """
        x = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(-1)
        x = x.to(self._device)   # (1, T, 1)
        with torch.no_grad():
            out = self._model(x)  # (1, T, 1)
        return out.squeeze(0).squeeze(-1).cpu().numpy()   # (T,)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save(self._model.state_dict(), path)
        logger.info("GRUSmoother saved → %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "GRUSmoother":
        smoother = cls(device=device)
        state = torch.load(path, map_location=smoother._device, weights_only=True)
        smoother._model.load_state_dict(state)
        smoother._model.eval()
        logger.info("GRUSmoother loaded from %s  (device: %s)", path, smoother._device)
        return smoother
