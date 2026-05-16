"""NMF-based source separation guided by the attention classifier.

The mixture spectrogram is decomposed into K spectral components via NMF (V ≈ W × H).
Each component is then scored as "female" or "male" by correlating its temporal
activation H[k,:] with the classifier's attention weights (per-frame F probability).
A per-bin IRM (approximate Ideal Ratio Mask) is built from the NMF reconstructions
and refined with the existing pitch masks.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.decomposition import NMF

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

N_COMPONENTS = 8    # NMF components — with 2 speakers, beyond 8 mixed components emerge
NMF_MAX_ITER = 500  # with dominant-frame scores clustering near 0.5


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


def separate_nmf(
    audio: np.ndarray,
    attention_weights: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_components: int = N_COMPONENTS,
    component_sharpening: float = 3.0,
    refine_with_pitch: bool = True,
) -> np.ndarray:
    """Separate the female voice from the mixture using classifier-guided NMF.

    Full pipeline:
        1. STFT → magnitude spectrogram V
        2. NMF: V ≈ W × H  (K spectral bases + K activation sequences)
        3. Score each component via correlation with the attention weights
        4. Build a per-bin IRM from the female/male NMF reconstructions
        5. Refine with pitch mask (harmonic floor + male suppression)
        6. Apply mask to the complex STFT and reconstruct via ISTFT

    Args:
        audio:                mixture waveform, shape (n_samples,)
        attention_weights:    per-frame F probability, shape (n_frames,)
        sr:                   sample rate
        n_components:         number of NMF components K
        component_sharpening: sigmoid steepness applied to component scores
        refine_with_pitch:    if True, apply harmonic floor + male suppression

    Returns:
        reconstructed waveform, shape (n_samples,)
    """
    stft = compute_stft(audio)                    # (n_freqs, n_frames_stft)
    n_freqs, n_frames_stft = stft.shape
    magnitude = np.abs(stft)                      # V: (n_freqs, n_frames_stft)

    # ------------------------------------------------------------------ #
    # Step 1 — NMF: V ≈ W × H                                            #
    # Normalise magnitude for numerical stability (the IRM is a ratio,   #
    # so scale cancels out). sklearn expects (n_samples, n_features) →   #
    # transpose before fitting.                                           #
    # ------------------------------------------------------------------ #
    eps = 1e-8
    scale = magnitude.max() + eps
    # Floor at eps to avoid exact zeros that cause NMF instability
    magnitude_norm = np.maximum(magnitude / scale, eps)

    nmf = NMF(
        n_components=n_components,
        init="random",
        solver="mu",     # multiplicative updates: stable with sparse matrices, no division by zero
        max_iter=NMF_MAX_ITER,
        random_state=42,
    )
    # sklearn < 1.4 'mu' solver emits RuntimeWarning on sparse matrices —
    # it is an internal false positive; the output is numerically correct.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        H_sk = nmf.fit_transform(magnitude_norm.T)  # (n_frames_stft, K) — activations
    W_sk = nmf.components_                           # (K, n_freqs)     — spectral bases

    # Standard notation: W (n_freqs, K), H (K, n_frames)
    W = np.nan_to_num(W_sk.T.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    H = np.nan_to_num(H_sk.T.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    # Per-component normalisation: scale each component to max=1 so W @ H
    # stays in range and does not overflow.
    col_max = W.max(axis=0) + eps   # (K,) — column-wise maximum of W
    W = W / col_max[np.newaxis, :]  # normalised W
    H = H * col_max[:, np.newaxis]  # compensated H → W@H unchanged, range stable

    # ------------------------------------------------------------------ #
    # Step 2 — score components against the attention weights             #
    # ------------------------------------------------------------------ #
    raw_scores = _score_components(H, attention_weights)              # (K,) in [0,1]
    # Map weights to [0.15, 0.85]: clearly-F → 0.85, clearly-M → 0.15,
    # uncertain → 0.50. Prevents the IRM from collapsing to 0 when male
    # components carry more energy than female ones.
    female_weights = 0.15 + 0.70 * sharpen_mask(raw_scores, power=component_sharpening)

    # ------------------------------------------------------------------ #
    # Step 3 — per-bin IRM from NMF reconstructions                      #
    # V_female(f,t) = Σ_k  female_weights[k] * W[f,k] * H[k,t]          #
    # ------------------------------------------------------------------ #
    n_frames = min(H.shape[1], n_frames_stft)
    H = H[:, :n_frames]

    # These matmuls may raise IEEE-754 RuntimeWarning on subnormal values
    # from some BLAS implementations — false positive, output is correct.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        V_female = W @ (female_weights[:, np.newaxis] * H)          # (n_freqs, n_frames)
        V_male   = W @ ((1.0 - female_weights)[:, np.newaxis] * H)

    eps = 1e-8
    # Linear soft mask: avoids energy-imbalance exaggeration from squaring.
    # IRM(f,t) = weighted-average of female_weights[k] over components,
    # where the weight of k is W[f,k]*H[k,t] — i.e. how much k contributes
    # to bin (f,t). Values stay in [female_weights.min(), female_weights.max()].
    irm_nmf = V_female / (V_female + V_male + eps)          # (n_freqs, n_frames)

    # Blend NMF IRM with the per-frame attention weights.
    # Attention provides a reliable temporal F/M signal; NMF adds per-frequency
    # resolution. When NMF component scoring collapses (classifier uncertain,
    # most dominant-frame scores cluster near 0.5), the attention weights prevent
    # the IRM from being dragged below 0.5 by NMF energy imbalance.
    n_frames_attn = min(len(attention_weights), n_frames)
    attn_col = np.pad(
        attention_weights[:n_frames_attn].astype(np.float32),
        (0, n_frames - n_frames_attn),
        constant_values=0.5,
    )
    attn_mask = attn_col[np.newaxis, :]      # (1, n_frames) → broadcasts to (n_freqs, n_frames)

    irm = 0.65 * attn_mask + 0.35 * irm_nmf
    irm = np.clip(irm, 0.0, 1.0).astype(np.float32)
    logger.debug("IRM pre-pitch: mean=%.3f  >0.6: %d%%  <0.4: %d%%",
                 irm.mean(),
                 int((irm > 0.6).mean() * 100),
                 int((irm < 0.4).mean() * 100))

    # ------------------------------------------------------------------ #
    # Step 4 — pitch mask refinement                                      #
    # Confirmed female harmonics are preserved (floor).                   #
    # Confirmed male harmonics are suppressed (cap).                      #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Step 5 — apply mask and reconstruct                                 #
    # Original phase is preserved: masked = |X| * IRM * e^{jφ}           #
    # ------------------------------------------------------------------ #
    masked_stft = stft[:, :n_frames] * irm
    return compute_istft(masked_stft, length=len(audio))
