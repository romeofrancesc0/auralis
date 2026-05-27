"""NMF-based source separation guided by the attention classifier.

The mixture spectrogram is decomposed into K spectral components via NMF (V ≈ W × H).
Each component is then scored as "female" or "male" by correlating its temporal
activation H[k,:] with the classifier's attention weights (per-frame F probability).
A per-bin IRM (approximate Ideal Ratio Mask) is built from the NMF reconstructions
and refined with the existing pitch masks.

The optional MaskNet (src.ai.mask_net) can further refine the IRM using a small CNN
trained to match the ideal IRM computed from clean sources.
"""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import NMF

if TYPE_CHECKING:
    from src.ai.mask_net import MaskNet

logger = logging.getLogger(__name__)

from src.dsp.separation import (
    HARMONIC_FLOOR,
    MALE_SUPPRESSION,
    compute_male_suppression_mask,
    compute_pitch_mask,
    sharpen_mask,
)
from src.dsp.stft import HOP_LENGTH, N_FFT, compute_istft, compute_stft
from src.utils import SAMPLE_RATE

N_COMPONENTS = 16       # NMF components — 16 gives finer spectral resolution for 2-speaker mixes
NMF_MAX_ITER = 1000     # generous headroom for mu solver convergence on K=16
IRM_ATTN_BLEND = 0.75   # weight of attention mask in the hybrid IRM
IRM_NMF_BLEND = 0.25    # weight of NMF soft mask; must sum to 1 with IRM_ATTN_BLEND
IRM_FLOOR = 0.15        # global minimum IRM: prevents spectral holes that hurt intelligibility
IRM_SMOOTH_SIGMA = (1.0, 2.0)   # (freq_bins, time_frames) Gaussian σ — reduces NMF musical noise
GRIFFIN_LIM_ITERS = 32  # phase reconstruction iterations (0 = keep mix phase, fast but artifacted)


def _score_components(H: np.ndarray, attention_weights: np.ndarray) -> np.ndarray:
    """Assign each NMF component a "femaleness" score in [0, 1].

    Dominant-frame strategy: the score of component k is the mean of the
    attention weights over the frames where k has the highest activation among
    all components. In those frames the mix is dominated by k's spectral
    structure, so the classifier signal is most informative. Shared frames
    (where k is just one of many active components) are excluded.

    After scoring, values are linearly rescaled to span [0, 1] to maximise
    separability even when the classifier is uncertain.

    Args:
        H:                 NMF activation matrix, shape (K, n_frames)
        attention_weights: per-frame F probability from the classifier, shape (n_frames,)

    Returns:
        scores: shape (K,), values in [0, 1]
    """
    K = H.shape[0]
    eps = 1e-8
    n_frames = min(H.shape[1], len(attention_weights))
    attn = attention_weights[:n_frames]
    H_aligned = H[:, :n_frames]

    # For each frame, which component has the highest activation?
    dominant = np.argmax(H_aligned, axis=0)   # (n_frames,)

    scores = np.full(K, 0.5, dtype=np.float32)
    for k in range(K):
        frames_k = dominant == k
        if frames_k.sum() >= 3:               # at least 3 frames for a stable estimate
            scores[k] = float(attn[frames_k].mean())

    # Linear rescaling to [0, 1]: if all scores were 0.5 (fully uncertain classifier)
    # the range would be 0 → no change.
    score_min, score_max = scores.min(), scores.max()
    score_range = score_max - score_min
    if score_range > 0.05:                    # only if there is meaningful variation
        scores = (scores - score_min) / score_range

    return scores


def _build_irm(
    stft: np.ndarray,
    magnitude: np.ndarray,
    attention_weights: np.ndarray,
    audio: np.ndarray,
    sr: int,
    n_components: int,
    component_sharpening: float,
    refine_with_pitch: bool,
) -> np.ndarray:
    """Compute the NMF-guided IRM from a pre-computed STFT.

    Implements steps 1–5 of the separation pipeline:
        1. NMF decomposition of the magnitude spectrogram
        2. Component scoring via attention weights
        3. Per-bin IRM from female/male NMF reconstructions
        4. Hybrid blend: IRM = 0.75 * attention + 0.25 * NMF-IRM
        5. Pitch mask refinement (harmonic floor + male suppression + IRM floor)

    Args:
        stft:               complex STFT, shape (n_freqs, n_frames_stft)
        magnitude:          |stft|, shape (n_freqs, n_frames_stft)
        attention_weights:  per-frame F probability, shape (n_frames,)
        audio:              raw waveform — required for pYIN pitch detection
        sr:                 sample rate
        n_components:       number of NMF components K
        component_sharpening: sigmoid steepness for component score mapping
        refine_with_pitch:  if True, apply harmonic floor + male suppression

    Returns:
        irm: shape (n_freqs, n_frames), values in [0, 1]
    """
    n_freqs, n_frames_stft = stft.shape

    eps = 1e-8
    scale = magnitude.max() + eps
    magnitude_norm = np.maximum(magnitude / scale, eps)

    nmf = NMF(
        n_components=n_components,
        init="random",
        solver="mu",
        max_iter=NMF_MAX_ITER,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        H_sk = nmf.fit_transform(magnitude_norm.T)  # (n_frames_stft, K)
    W_sk = nmf.components_                           # (K, n_freqs)

    W = np.nan_to_num(W_sk.T.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    H = np.nan_to_num(H_sk.T.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    col_max = W.max(axis=0) + eps
    W = W / col_max[np.newaxis, :]
    H = H * col_max[:, np.newaxis]

    raw_scores = _score_components(H, attention_weights)
    female_weights = 0.05 + 0.90 * sharpen_mask(raw_scores, power=component_sharpening)

    n_frames = min(H.shape[1], n_frames_stft)
    H = H[:, :n_frames]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        V_female = W @ (female_weights[:, np.newaxis] * H)
        V_male   = W @ ((1.0 - female_weights)[:, np.newaxis] * H)

    eps = 1e-8
    irm_nmf = V_female / (V_female + V_male + eps)

    n_frames_attn = min(len(attention_weights), n_frames)
    attn_col = np.pad(
        attention_weights[:n_frames_attn].astype(np.float32),
        (0, n_frames - n_frames_attn),
        constant_values=0.5,
    )
    attn_mask = attn_col[np.newaxis, :]

    irm = IRM_ATTN_BLEND * attn_mask + IRM_NMF_BLEND * irm_nmf
    irm = np.clip(irm, 0.0, 1.0).astype(np.float32)

    # 2-D Gaussian smoothing: suppresses NMF musical noise (isolated tonal artefacts)
    # applied before pitch refinement so harmonic corrections can still override it
    irm = gaussian_filter(irm.astype(np.float64), sigma=IRM_SMOOTH_SIGMA).astype(np.float32)
    irm = np.clip(irm, 0.0, 1.0)

    logger.debug("IRM pre-pitch: mean=%.3f  >0.6: %d%%  <0.4: %d%%",
                 irm.mean(),
                 int((irm > 0.6).mean() * 100),
                 int((irm < 0.4).mean() * 100))

    male_suppressed_bins = np.zeros(irm.shape, dtype=bool)

    if refine_with_pitch:
        _, harmonic_bins = compute_pitch_mask(
            audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        )
        harmonic_bins = harmonic_bins[:, :n_frames]
        irm = np.where(harmonic_bins, np.maximum(irm, HARMONIC_FLOOR), irm)

        male_mask = compute_male_suppression_mask(
            audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        )
        male_harmonic_bins = male_mask[:, :n_frames] < 0.5
        irm = np.where(male_harmonic_bins, np.minimum(irm, MALE_SUPPRESSION), irm)
        male_suppressed_bins = male_harmonic_bins

    if float(attention_weights.mean()) >= 0.25:
        irm = np.where(~male_suppressed_bins, np.maximum(irm, IRM_FLOOR), irm)

    return irm


def _griffin_lim_reconstruct(
    target_magnitude: np.ndarray,
    init_phase: np.ndarray,
    n_iter: int,
    length: int,
) -> np.ndarray:
    """Phase reconstruction via Griffin-Lim (Griffin & Lim 1984) initialized from mix phase.

    Iteratively enforces STFT consistency on the target magnitude spectrogram.
    Starting from the mix phase (rather than random) reduces iterations needed
    and preserves approximate phase structure in the early iterations.

    Args:
        target_magnitude: masked magnitude spectrogram, shape (n_freqs, n_frames)
        init_phase:       initial phase estimate (mix phase), shape (n_freqs, n_frames)
        n_iter:           number of Griffin-Lim iterations
        length:           target output length in samples

    Returns:
        reconstructed waveform, shape (length,)
    """
    phase = init_phase.copy()
    for _ in range(n_iter):
        signal = compute_istft(target_magnitude * np.exp(1j * phase), length=length)
        stft_iter = compute_stft(signal)
        n_f, n_t = target_magnitude.shape
        phase = np.angle(stft_iter[:n_f, :n_t])
    return compute_istft(target_magnitude * np.exp(1j * phase), length=length)


def compute_nmf_irm(
    audio: np.ndarray,
    attention_weights: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_components: int = N_COMPONENTS,
    component_sharpening: float = 3.0,
    refine_with_pitch: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the NMF-guided IRM and the STFT magnitude spectrogram.

    Public entry point used by the MaskNet training script to obtain the
    classical-pipeline IRM as a training input feature.

    Args:
        audio:             mixture waveform, shape (n_samples,)
        attention_weights: per-frame F probability, shape (n_frames,)
        sr:                sample rate
        n_components:      number of NMF components K
        component_sharpening: sigmoid steepness for component scoring
        refine_with_pitch: if True, apply harmonic floor + male suppression

    Returns:
        irm:       shape (n_freqs, n_frames), values in [0, 1]
        magnitude: shape (n_freqs, n_frames)
    """
    stft = compute_stft(audio)
    magnitude = np.abs(stft)
    irm = _build_irm(
        stft, magnitude, attention_weights, audio,
        sr, n_components, component_sharpening, refine_with_pitch,
    )
    n_frames = irm.shape[1]
    return irm, magnitude[:, :n_frames]


def separate_nmf(
    audio: np.ndarray,
    attention_weights: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_components: int = N_COMPONENTS,
    component_sharpening: float = 3.0,
    refine_with_pitch: bool = True,
    mask_net: "MaskNet | None" = None,
    target_gender: int = 0,
    griffin_lim_iters: int = GRIFFIN_LIM_ITERS,
) -> np.ndarray:
    """Separate the female voice from the mixture using classifier-guided NMF.

    Full pipeline:
        1. STFT → magnitude spectrogram V
        2–5. NMF decomposition + scoring + IRM blending + Gaussian smoothing + pitch refinement
             (delegated to _build_irm)
        6. [optional] MaskNet / DPCRN CNN refinement of the IRM
        7. Apply mask to magnitude; reconstruct phase via Griffin-Lim (or ISTFT with mix phase)

    Args:
        audio:                mixture waveform, shape (n_samples,)
        attention_weights:    per-frame F probability, shape (n_frames,)
        sr:                   sample rate
        n_components:         number of NMF components K
        component_sharpening: sigmoid steepness applied to component scores
        refine_with_pitch:    if True, apply harmonic floor + male suppression
        mask_net:             optional MaskNet or DPCRN instance for IRM refinement
        target_gender:        0=Female (default), 1=Male — passed to MaskNet conditioning
        griffin_lim_iters:    Griffin-Lim phase reconstruction iterations; 0 = use mix phase

    Returns:
        reconstructed waveform, shape (n_samples,)
    """
    stft = compute_stft(audio)
    magnitude = np.abs(stft)

    irm = _build_irm(
        stft, magnitude, attention_weights, audio,
        sr, n_components, component_sharpening, refine_with_pitch,
    )

    if mask_net is not None:
        n_frames = irm.shape[1]
        refined = mask_net.refine(
            magnitude[:, :n_frames], attention_weights, irm, gender=target_gender
        )
        irm = np.clip(refined, 0.0, 1.0).astype(np.float32)
        logger.debug("IRM post-MaskNet: mean=%.3f", irm.mean())

    n_frames = irm.shape[1]
    target_magnitude = magnitude[:, :n_frames] * irm

    if griffin_lim_iters > 0:
        return _griffin_lim_reconstruct(
            target_magnitude,
            np.angle(stft[:, :n_frames]),
            griffin_lim_iters,
            len(audio),
        )

    return compute_istft(target_magnitude * np.exp(1j * np.angle(stft[:, :n_frames])), length=len(audio))
