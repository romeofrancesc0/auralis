"""NMF-based source separation guided by the attention classifier.

Approccio:
    Il Mix viene decomposto in K componenti spettrali tramite NMF (V ≈ W × H).
    Ogni componente viene poi classificata come "femminile" o "maschile"
    calcolando la correlazione tra la sua attivazione temporale H[k,:] e le
    attention weights del classificatore (probabilità F per frame).
    Si costruisce una IRM (Ideal Ratio Mask approssimata) per voce per voce
    e la si raffina con le pitch mask già esistenti.
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

N_COMPONENTS = 8    # componenti NMF — con 2 parlanti, 8 è il limite oltre cui emergono
NMF_MAX_ITER = 500  # componenti "miste" con score vicino a 0.5


def _score_components(H: np.ndarray, attention_weights: np.ndarray) -> np.ndarray:
    """Assegna a ogni componente NMF uno score "femminilità" in [0, 1].

    Strategia dominant-frame: lo score di k è la media delle attention weights
    nei soli frame in cui k ha l'attivazione più alta tra tutte le componenti.
    In quei frame il mix è dominato dalla struttura spettrale di k, quindi
    il classificatore è più informativo. I frame "condivisi" (dove k è solo una
    delle tante componenti attive) vengono ignorati.

    Dopo il calcolo, gli score vengono riscalati a coprire l'intero range [0,1]
    così da massimizzare la separabilità anche quando il classificatore è incerto.

    Args:
        H:                 matrice di attivazione NMF, shape (K, n_frames)
        attention_weights: probabilità F per frame dal classificatore, shape (n_frames,)

    Returns:
        scores: shape (K,), valori in [0, 1]
    """
    K = H.shape[0]
    eps = 1e-8
    n_frames = min(H.shape[1], len(attention_weights))
    attn = attention_weights[:n_frames]
    H_aligned = H[:, :n_frames]

    # Per ogni frame, quale componente ha l'attivazione massima?
    dominant = np.argmax(H_aligned, axis=0)   # (n_frames,)

    scores = np.full(K, 0.5, dtype=np.float32)
    for k in range(K):
        frames_k = dominant == k
        if frames_k.sum() >= 3:               # almeno 3 frame per avere una stima stabile
            scores[k] = float(attn[frames_k].mean())

    # Riscalamento lineare a [0, 1]: se tutti gli score fossero 0.5
    # (classificatore completamente incerto) il range sarebbe 0 → nessun cambio.
    score_min, score_max = scores.min(), scores.max()
    score_range = score_max - score_min
    if score_range > 0.05:                    # solo se c'è variazione significativa
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
    """Separa la voce femminile dal mix tramite NMF guidata dal classificatore.

    Pipeline completa:
        1. STFT → spettrogramma di magnitudine V
        2. NMF: V ≈ W × H  (K basi spettrali + K sequenze di attivazione)
        3. Score di ogni componente via correlazione con le attention weights
        4. Costruzione della IRM per bin da ricostruzioni NMF F vs M
        5. Raffinamento con pitch mask (harmonic floor + male suppression)
        6. Applicazione maschera allo STFT complesso e ricostruzione ISTFT

    Args:
        audio:                mixture waveform, shape (n_samples,)
        attention_weights:    probabilità F per frame, shape (n_frames,)
        sr:                   sample rate
        n_components:         numero di componenti NMF K
        component_sharpening: steepness del sigmoid sui component scores
        refine_with_pitch:    se True, applica harmonic floor + male suppression

    Returns:
        segnale ricostruito, shape (n_samples,)
    """
    stft = compute_stft(audio)                    # (n_freqs, n_frames_stft)
    n_freqs, n_frames_stft = stft.shape
    magnitude = np.abs(stft)                      # V: (n_freqs, n_frames_stft)

    # ------------------------------------------------------------------ #
    # Step 1 — NMF: V ≈ W × H                                            #
    # Normalizziamo la magnitudine per stabilità numerica (l'IRM è un    #
    # rapporto, quindi la scala si cancella). sklearn si aspetta           #
    # (n_samples, n_features) → trasponiamo.                              #
    # ------------------------------------------------------------------ #
    eps = 1e-8
    scale = magnitude.max() + eps
    # Floor a eps per evitare zeri esatti che causano instabilità nell'NMF
    magnitude_norm = np.maximum(magnitude / scale, eps)

    nmf = NMF(
        n_components=n_components,
        init="random",
        solver="mu",     # multiplicative updates: stabile con matrici sparse, no divisione per zero
        max_iter=NMF_MAX_ITER,
        random_state=42,
    )
    # Il solver 'mu' di sklearn < 1.4 emette RuntimeWarning su matrici sparse;
    # è un falso positivo interno: l'output è numericamente corretto.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        H_sk = nmf.fit_transform(magnitude_norm.T)  # (n_frames_stft, K) — attivazioni
    W_sk = nmf.components_                           # (K, n_freqs)     — basi spettrali

    # Notazione standard: W (n_freqs, K), H (K, n_frames)
    W = np.nan_to_num(W_sk.T.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    H = np.nan_to_num(H_sk.T.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    # Normalizzazione per componente: scala ciascuna componente a max=1
    # così W @ H rimane in range e non va in overflow
    col_max = W.max(axis=0) + eps   # (K,) — massimo per ogni colonna di W
    W = W / col_max[np.newaxis, :]  # W normalizzato
    H = H * col_max[:, np.newaxis]  # H compensato → W@H invariato, range stabile

    # ------------------------------------------------------------------ #
    # Step 2 — score delle componenti con le attention weights            #
    # ------------------------------------------------------------------ #
    raw_scores = _score_components(H, attention_weights)              # (K,) in [0,1]
    # Mappa i pesi in [0.15, 0.85]: componenti chiaramente F → 0.85,
    # chiaramente M → 0.15, incerte → 0.50. Evita che l'IRM collassi a 0
    # quando le componenti maschili portano più energia di quelle femminili.
    female_weights = 0.15 + 0.70 * sharpen_mask(raw_scores, power=component_sharpening)

    # ------------------------------------------------------------------ #
    # Step 3 — IRM per bin da ricostruzioni NMF                          #
    # V_female(f,t) = Σ_k  female_weights[k] * W[f,k] * H[k,t]          #
    # ------------------------------------------------------------------ #
    n_frames = min(H.shape[1], n_frames_stft)
    H = H[:, :n_frames]

    # Le due matmul possono sollevare RuntimeWarning IEEE 754 su certi valori
    # subnormali — è un falso positivo del BLAS, l'output rimane corretto.
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
    # Step 4 — raffinamento con pitch mask                                #
    # Gli armonici femminili confermati vengono preservati (floor).       #
    # Gli armonici maschili confermati vengono soppressi (cap).           #
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
    # Step 5 — applica la maschera e ricostruisci                         #
    # La fase originale viene preservata: masked = |X| * IRM * e^{jφ}    #
    # ------------------------------------------------------------------ #
    masked_stft = stft[:, :n_frames] * irm
    return compute_istft(masked_stft, length=len(audio))
