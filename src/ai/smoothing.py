"""HMM-based temporal smoothing for the per-frame attention mask.

A 2-state HMM (state 0 = F-dominant, state 1 = M-dominant) models the
temporal structure of speaker dominance and resolves frame-level noise in
the classifier output via the forward-backward algorithm.

Why forward-backward instead of Viterbi
----------------------------------------
Viterbi returns the single most-likely state *sequence* (hard 0/1).
Forward-backward returns the posterior *probability* P(state | all obs) at
each frame — a soft value in [0, 1] that integrates context from both past
and future frames.  The result is a smooth mask that retains the probabilistic
nature expected by the hybrid IRM in nmf_separation.py.

Transition probability design
------------------------------
  p_ff = P(F → F) = 0.95   high inertia in female state
  p_fm = P(F → M) = 0.05   rarely leaves female state
  p_mf = P(M → F) = 0.20   faster recovery than departure (female-biased)
  p_mm = P(M → M) = 0.80   moderate inertia in male state

At HOP_LENGTH=128 / SR=16000 → frame ≈ 8 ms:
  Expected time in F before switching: 1/0.05 = 20 frames ≈ 160 ms
  Expected time in M before switching: 1/0.20 =  5 frames ≈  40 ms
  → Brief male interruptions in female speech are smoothed out.
"""
from __future__ import annotations

import numpy as np

P_FF_DEFAULT: float = 0.95   # P(F-dominant → F-dominant)
P_MF_DEFAULT: float = 0.20   # P(M-dominant → F-dominant)  >  P(F→M) = 0.05


def hmm_smooth(
    mask: np.ndarray,
    p_start_female: float = 0.7,
    p_ff: float = P_FF_DEFAULT,
    p_mf: float = P_MF_DEFAULT,
) -> np.ndarray:
    """Smooth an attention mask with a 2-state HMM (forward-backward algorithm).

    The raw per-frame probabilities from the classifier are used directly as
    emission probabilities:
        P(obs_t | state = F) = mask[t]
        P(obs_t | state = M) = 1 − mask[t]

    Args:
        mask:            per-frame P(female), shape (n_frames,), values in [0, 1]
        p_start_female:  prior P(F-dominant) at frame 0
        p_ff:            P(stay in F-dominant state)
        p_mf:            P(M-dominant → F-dominant)

    Returns:
        smoothed: posterior P(F-dominant | all observations), shape (n_frames,),
                  values in [0, 1]
    """
    eps = 1e-12
    T = len(mask)

    # Transition matrix A[s_from, s_to]
    A = np.array(
        [[p_ff,        1.0 - p_ff],
         [p_mf,        1.0 - p_mf]],
        dtype=np.float64,
    )

    pi = np.array([p_start_female, 1.0 - p_start_female], dtype=np.float64)

    # Emission probabilities B[t, s]
    m = np.clip(mask.astype(np.float64), eps, 1.0 - eps)
    B = np.column_stack([m, 1.0 - m])   # (T, 2)

    # Forward pass — normalised at each step to prevent underflow
    alpha = np.empty((T, 2), dtype=np.float64)
    alpha[0] = pi * B[0]
    alpha[0] /= alpha[0].sum() + eps
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[t]
        alpha[t] /= alpha[t].sum() + eps

    # Backward pass — normalised at each step
    beta = np.ones((T, 2), dtype=np.float64)
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[t + 1] * beta[t + 1])
        beta[t] /= beta[t].sum() + eps

    # Posterior P(state | all obs)
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True) + eps

    return gamma[:, 0].astype(np.float32)   # P(F-dominant | all observations)
