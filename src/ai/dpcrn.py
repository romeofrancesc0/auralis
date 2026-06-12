"""Dual-Path Convolutional Recurrent Network (DPCRN) for T-F mask refinement.

DPCRN alternates between two complementary processing paths:
  - Intra-frame (frequency axis): dilated Conv2d captures local harmonic structure
  - Inter-frame (time axis):      GRU captures speaker-turn dynamics

Inspired by:
  "DPCRN: Dual-Path Convolution Recurrent Network for Single Channel
  Speech Enhancement" — Le et al., ICASSP 2022.

Architecture (C=64, N=8 DualPath blocks):
  Input (B, 3, F, T) → Encoder → [DualPath block × 8] → Decoder → (B, F, T)
  ~302K parameters. Compatible drop-in for MaskNet — same build_input(), same
  inference interface.

Runs on CPU, CUDA, or Apple MPS without code changes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

N_CHANNELS: int = 64   # feature channels inside the network
N_BLOCKS:   int = 8    # number of DualPath blocks

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    class _DualPathBlock(nn.Module):
        """One DualPath processing block.

        Intra-frame path: Conv2d(C, C, (3,1)) along frequency axis — models local
        harmonic and formant structure within each time frame.

        Inter-frame path: GRU(C, C) along time axis — models temporal dynamics
        of speaker dominance across frames.

        Residual connection added after each path to stabilise training.
        """

        def __init__(self, n_ch: int) -> None:
            super().__init__()
            # Frequency-axis conv: kernel (3,1) processes 3 adjacent freq bins
            self.intra_conv = nn.Conv2d(n_ch, n_ch, kernel_size=(3, 1), padding=(1, 0))
            self.intra_bn   = nn.BatchNorm2d(n_ch)
            self.intra_act  = nn.ReLU(inplace=True)
            # Time-axis GRU: processes each freq bin as an independent sequence
            self.inter_gru  = nn.GRU(n_ch, n_ch, batch_first=True)
            self.inter_bn   = nn.BatchNorm1d(n_ch)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, C, F, T)
            B, C, F, T = x.shape

            # Intra-frame path (frequency)
            h = self.intra_act(self.intra_bn(self.intra_conv(x)))
            x = x + h

            # Inter-frame path (time)
            # .contiguous() required before reshape on MPS (non-contiguous permute)
            x_t = x.permute(0, 2, 3, 1).contiguous().reshape(B * F, T, C)   # (B*F, T, C)
            h_t, _ = self.inter_gru(x_t)                                      # (B*F, T, C)
            # BatchNorm1d expects (N, C, L) for sequence data
            h_t = self.inter_bn(h_t.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
            x_t = x_t + h_t
            x = x_t.reshape(B, F, T, C).permute(0, 3, 1, 2).contiguous()    # (B, C, F, T)

            return x

    class _DPCRN(nn.Module):
        """Full DPCRN: encoder + N DualPath blocks + decoder.

        Input/output format is identical to _CNN in mask_net.py so DPCRN
        can serve as a higher-capacity drop-in.
        """

        def __init__(self, n_ch: int = N_CHANNELS, n_blocks: int = N_BLOCKS) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, n_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_ch),
                nn.ReLU(inplace=True),
            )
            self.blocks = nn.ModuleList(
                [_DualPathBlock(n_ch) for _ in range(n_blocks)]
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(n_ch, 1, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (B, 3, F, T)

            Returns:
                (B, F, T) refined mask in [0, 1]
            """
            h = self.encoder(x)
            for block in self.blocks:
                h = block(h)
            return self.decoder(h).squeeze(1)   # (B, F, T)

        @property
        def n_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

    class _DPCRNSeparator(nn.Module):
        """DPCRN as a primary two-speaker separator (not a mask refiner).

        Unlike _DPCRN, which refines a heuristic IRM and therefore needs the
        attention/NMF channels, the separator sees only the normalised
        log-magnitude spectrogram of the mixture and emits one mask per source.
        Both masks are trained jointly with utterance-level permutation
        invariant training (uPIT, Kolbæk et al. 2017), so neither output is
        tied to a gender a priori — the attention module performs the
        top-down stream selection afterwards (cocktail-party model:
        bottom-up segregation, then attentional selection).
        """

        N_SOURCES = 2

        def __init__(self, n_ch: int = N_CHANNELS, n_blocks: int = N_BLOCKS) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, n_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_ch),
                nn.ReLU(inplace=True),
            )
            self.blocks = nn.ModuleList(
                [_DualPathBlock(n_ch) for _ in range(n_blocks)]
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(n_ch, self.N_SOURCES, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (B, 1, F, T) normalised log-magnitude of the mixture

            Returns:
                (B, 2, F, T) one mask per source, values in [0, 1]
            """
            h = self.encoder(x)
            for block in self.blocks:
                h = block(h)
            return self.decoder(h)

        @property
        def n_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

    _TORCH_AVAILABLE = True
except ImportError:
    pass


def _check_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for DPCRN. "
            "Install with: pip install torch  (or: pip install 'auralis[torch]')"
        )


def _auto_device() -> str:
    """Select CUDA > MPS > CPU in order of availability."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DPCRN:
    """Inference wrapper around _DPCRN.

    Drop-in replacement for MaskNet — identical refine() / save() / load()
    interface, uses the same build_input() helper from mask_net.py.
    """

    def __init__(self, device: str | None = None) -> None:
        _check_torch()
        self._device = device or _auto_device()
        self._model = _DPCRN().to(self._device)
        self._model.eval()
        logger.debug(
            "DPCRN ready on %s  (%d params)", self._device, self._model.n_params
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def refine(
        self,
        magnitude: np.ndarray,
        attention_weights: np.ndarray,
        nmf_irm: np.ndarray,
        gender: int = 0,   # accepted for API compatibility; DPCRN does not use it
    ) -> np.ndarray:
        """Refine the NMF-IRM with the DPCRN.

        Args:
            magnitude:         STFT magnitude, shape (n_freqs, n_frames)
            attention_weights: per-frame attention in [0, 1], shape (n_frames,)
            nmf_irm:           NMF-guided IRM in [0, 1], shape (n_freqs, n_frames)
            gender:            accepted for interface compatibility; unused by DPCRN

        Returns:
            Refined mask in [0, 1], shape (n_freqs, n_frames)
        """
        from src.ai.mask_net import build_input
        x = build_input(magnitude, attention_weights, nmf_irm)   # (3, F, T)
        x_t = torch.from_numpy(x).unsqueeze(0).to(self._device)  # (1, 3, F, T)
        with torch.no_grad():
            out = self._model(x_t)              # (1, F, T)
        return out.squeeze(0).cpu().numpy()     # (F, T)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save(self._model.state_dict(), path)
        logger.info("DPCRN saved → %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "DPCRN":
        net = cls(device=device)
        state = torch.load(path, map_location=net._device, weights_only=True)
        net._model.load_state_dict(state)
        net._model.eval()
        logger.info("DPCRN loaded from %s  (device: %s)", path, net._device)
        return net


# ----------------------------------------------------------------------
# Separator: STFT helpers shared by training and inference
# ----------------------------------------------------------------------
# torch.stft/istft are used (instead of librosa) so masking and
# reconstruction stay on the GPU and analysis/synthesis are guaranteed
# to match between training and inference.

def stft_torch(wav: "torch.Tensor") -> "torch.Tensor":
    """Complex STFT, shape (B, F, T). Parameters match src.dsp.stft."""
    from src.dsp.stft import HOP_LENGTH, N_FFT
    window = torch.hann_window(N_FFT, device=wav.device)
    return torch.stft(
        wav, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT,
        window=window, center=True, return_complex=True,
    )


def istft_torch(stft: "torch.Tensor", length: int) -> "torch.Tensor":
    """Inverse of stft_torch. Accepts (..., F, T), returns (..., length)."""
    from src.dsp.stft import HOP_LENGTH, N_FFT
    window = torch.hann_window(N_FFT, device=stft.device)
    shape = stft.shape
    flat = stft.reshape(-1, shape[-2], shape[-1])
    wav = torch.istft(
        flat, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT,
        window=window, center=True, length=length,
    )
    return wav.reshape(*shape[:-2], length)


def normalize_log_mag(magnitude: "torch.Tensor", eps: float = 1e-8) -> "torch.Tensor":
    """Per-utterance standardised log-magnitude: (log|X| − μ) / σ.

    Args:
        magnitude: (B, F, T)

    Returns:
        (B, 1, F, T) network-ready input
    """
    log_mag = torch.log(magnitude + eps)
    mean = log_mag.mean(dim=(-2, -1), keepdim=True)
    std = log_mag.std(dim=(-2, -1), keepdim=True)
    return ((log_mag - mean) / (std + eps)).unsqueeze(1)


class DPCRNSeparator:
    """Inference wrapper around _DPCRNSeparator.

    separate() returns BOTH estimated sources; which one is the target is
    decided downstream by the attention module (top-down selection).
    """

    def __init__(self, device: str | None = None) -> None:
        _check_torch()
        self._device = device or _auto_device()
        self._model = _DPCRNSeparator().to(self._device)
        self._model.eval()
        logger.debug(
            "DPCRNSeparator ready on %s  (%d params)",
            self._device, self._model.n_params,
        )

    def separate(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Separate a 2-speaker mixture into two estimated source waveforms.

        Masks are applied to the complex mix STFT (mix phase preserved),
        so no Griffin-Lim pass is needed.

        Args:
            audio: mixture waveform, shape (n_samples,)

        Returns:
            (source_a, source_b) — each shape (n_samples,); permutation is
            arbitrary, use the attention module to pick the target stream.
        """
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            mix_stft = stft_torch(wav)                       # (1, F, T)
            x = normalize_log_mag(mix_stft.abs())            # (1, 1, F, T)
            masks = self._model(x)                           # (1, 2, F, T)
            masked = mix_stft.unsqueeze(1) * masks           # (1, 2, F, T)
            sources = istft_torch(masked, length=len(audio)) # (1, 2, n)
        out = sources.squeeze(0).cpu().numpy()
        return out[0], out[1]

    def save(self, path: str | Path) -> None:
        torch.save(self._model.state_dict(), path)
        logger.info("DPCRNSeparator saved → %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "DPCRNSeparator":
        net = cls(device=device)
        state = torch.load(path, map_location=net._device, weights_only=True)
        net._model.load_state_dict(state)
        net._model.eval()
        logger.info("DPCRNSeparator loaded from %s  (device: %s)", path, net._device)
        return net
