# ROADMAP — Implementation Plan

> **Status:** in progress — Phases 0–6 complete (end-to-end pipeline working). Missing: evaluation notebook and automated tests with objective metrics.
> **Last updated:** 2026-05-16.

This document tracks the full implementation plan. It must be consulted and updated at the start of each phase. Decisions taken move from the "Open questions" section into the body of the document.

---

## 6-Phase Incremental Structure

Each phase produces something functional and testable. Phases respect technical dependencies: DSP feeds features to AI; AI returns decisions to DSP for separation.

---

### Phase 0 — Foundations (infrastructure)

**Goal:** reliable audio I/O and shared utilities before writing any domain logic.

1. `src/utils.py` — Audio I/O functions (load/save WAV via `librosa`/`soundfile`), resampling to a standard frequency (16 kHz), structured logging, input validation.
2. `tests/test_utils.py` — Tests for load→save round-trip and sample-rate conversions.

**Deliverable:** any module can reliably load and save audio files.

---

### Phase 1 — DSP: transform and features

**Goal:** given an audio signal, produce a feature vector for every time window.

1. `src/dsp/stft.py` — Wrapper around `librosa.stft` / `istft` with standardised parameters (n_fft, hop_length, window). Configuration must be centralised because STFT and ISTFT must share the same parameters.
2. `src/dsp/features.py` — Extractors for:
   - MFCC (timbre)
   - Pitch via pYIN (fundamental frequency)
   - RMS energy (signal intensity)
   - Spectral centroid + rolloff (timbral colour)
   - `extract_all(audio) → feature_matrix` returning everything concatenated per frame.
3. `tests/test_features.py` — Sanity checks on synthetic signals (sinusoids at known frequencies → correct pitch, silence → energy ≈ 0).

**Deliverable:** DSP pipeline up to the feature vector, tested on known signals.

---

### Phase 2 — Dataset and ground truth

**Goal:** data to train and evaluate on. **Critical phase** — nothing works without data.

Options considered:

- **Option A — Public dataset:** subset of **LibriSpeech** (clean labelled speech) + synthetic controlled mixes (two voices summed at known SNR).
- **Option B — Minimal custom dataset:** 5–10 clips per male/female voice, generate mixes.

**Operational plan:**
1. `src/utils.py` — `make_mixture(voice_a, voice_b, snr_db)` to create reproducible synthetic mixes.
2. `notebooks/01_data_exploration.ipynb` — Explore data, listen to mixes, visualise spectrograms.

**Deliverable:** small `data/raw/` dataset with labelled mixes.

---

### Phase 3 — AI: attention classifier

**Goal:** given a feature vector, decide which speaker is the target.

**Initial approach (simple and interpretable):**

1. `src/ai/train.py` — Script that:
   - Loads labelled audio for the target speaker,
   - Extracts features via `src/dsp/features`,
   - Trains a classifier (starting point: **Random Forest** — robust, minimal tuning, handles heterogeneous features),
   - Saves the model via `joblib` to `models/`.
2. `src/ai/classifier.py` — `SpeakerClassifier` class encapsulating load/predict.
3. `src/ai/attention.py` — High-level logic: given a multi-speaker spectrogram, returns per-frame probability of belonging to the target.
4. `tests/test_classifier.py` — Tests on known features → expected prediction.

**Deliverable:** trained classifier that, given an audio frame, decides "target" or "interferer".

---

### Phase 4 — DSP: separation and reconstruction

**Goal:** given the mix and the AI decisions, reconstruct the audio of the target only.

1. `src/dsp/separation.py` — Binary or soft **time-frequency mask** (Ratio Mask) based on the attention module output.
2. Apply the mask to the mixture spectrogram.
3. Reconstruction via ISTFT.
4. `tests/test_separation.py` — Tests on synthetic mixes: the reconstructed signal must be perceptually closer to the target than to the original mix (SDR / SI-SDR).

**Deliverable:** isolated audio of the target speaker.

---

### Phase 5 — Enhancement post-processing

**Goal:** refine the reconstructed audio by reducing artefacts and interferer residuals.

1. `src/dsp/enhancement.py` — Wrapper around `noisereduce` + optional lightweight Wiener filtering via `scipy.signal`.
2. A/B comparison: audio before vs. after enhancement.

**Deliverable:** final, clean version of the isolated audio.

---

### Phase 6 — End-to-end pipeline and evaluation

**Goal:** wire everything together behind a single CLI interface.

1. `src/pipeline.py` — CLI with `argparse`:
   ```
   python -m src.pipeline --input mix.wav --model models/classifier.joblib --output out.wav
   ```
   Orchestrates: load → STFT → features → AI decision → mask → ISTFT → enhancement → save.
2. `notebooks/02_evaluation.ipynb` — Quantitative evaluation:
   - Metrics: **SI-SDR**, **PESQ**, **STOI**.
   - Comparison against baseline.
   - Spectrograms before/after.
3. `tests/test_pipeline.py` — Smoke test end-to-end.

**Deliverable:** complete system + evaluation report with concrete numbers.

---

## Phase Dependencies

```
Phase 0 ──┐
          ├──► Phase 1 (DSP features) ──┐
          │                             ├──► Phase 3 (AI) ──┐
          └──► Phase 2 (Dataset) ───────┘                   │
                                                            ├──► Phase 4 ──► Phase 5 ──► Phase 6
                                       Phase 1 ─────────────┘
```

Phases 1 and 2 can proceed in parallel after Phase 0. Phase 3 requires both.

---

## Effort Estimate (relative)

| Phase | Complexity | Notes |
|---|---|---|
| 0 | Low | Standard I/O |
| 1 | Medium | STFT/feature parameter calibration |
| 2 | High | Most delicate: data selection and target criterion |
| 3 | Medium | Random Forest is low-effort, but requires a good training set |
| 4 | Medium | Masking is standard, but artefact management is tricky |
| 5 | Low | Post-processing with existing libraries |
| 6 | Medium | Integration + metrics |

---

## Open Questions — Decided

1. **Dataset:** ✅ LibriSpeech `dev-clean` (~337 MB). Download from `openslr.org/12/` and place under `data/raw/librispeech/dev-clean/`.
2. **Attention criterion:** ✅ Vocal gender (M/F). Start simple, raise the bar once the system works.
3. **Number of speakers in the mix:** ✅ 2 fixed speakers for v1 (1M + 1F). The classifier can scale to N speakers in future.
4. **Work order:** ✅ Phases 0+1 started immediately in parallel with data preparation.

---

## Progress

| Phase | Status | Notes |
|---|---|---|
| 0 | ✅ Done | `src/utils.py` + `tests/test_utils.py` |
| 1 | ✅ Done | `src/dsp/stft.py` + `src/dsp/features.py` (44 features) + `tests/test_features.py` |
| 2 | ✅ Done | `src/dsp/dataset.py` — SPEAKERS.TXT parsing, M+F mix, multi-SNR IBM dataset |
| 3 | ✅ Done | `src/ai/classifier.py`, `train.py`, `attention.py` — CV on 400 samples × 3 SNR values |
| 4 | ✅ Done | `src/dsp/nmf_separation.py` (primary module) + `separation.py` (utilities/fallback) |
| 5 | ✅ Done | `src/dsp/enhancement.py` — noisereduce + peak normalisation |
| 6 | ✅ Done | `src/pipeline.py` + `demo.py` + `tests/test_pipeline.py` (6 tests) + `notebooks/02_evaluation.ipynb` (SI-SDR, PESQ, STOI) |

> Legend: ⬜ Not started · 🟡 In progress · ✅ Done · 🔴 Active bug

---

## Implemented Improvements (v2 pipeline)

After the first manual test (v1 barely suppressed the male voice), three improvement steps were applied:

| Step | Change | File |
|---|---|---|
| 1 — IBM training | Retrained on real mixture frames with IBM frame-level labels. Resolves domain mismatch: the model now sees the same signal type at training and inference. | `src/dsp/dataset.py`, `src/ai/train.py` |
| 2 — Pitch mask | Frequency-selective mask: detects female F0 in the mix (150–310 Hz) and marks harmonic bins as "target". Suppresses the male voice in bins with no female harmonics. | `src/dsp/separation.py` |
| 3 — Sigmoid sharpening | Sigmoid centred at 0.5 (`1/(1+exp(-k*(m-0.5)))`): pushes values toward 0/1. Corrects the previous `mask**power` which collapsed everything toward 0. | `src/dsp/separation.py` |

---

## Implemented Improvements (v3 pipeline)

After further testing (v2 attenuated both voices in overlapping sections), the following improvements were applied:

| Step | Change | File |
|---|---|---|
| 4 — Extended features | From 17 to 44 features per frame: MFCC delta + delta-delta (26 additional features), ZCR, female-specific pitch range (150–310 Hz). Improves M/F discrimination in overlapping frames. | `src/dsp/features.py` |
| 5 — Multi-SNR training | Training on mixes at −3, 0, +3 dB SNR (instead of 0 dB only). The classifier becomes robust to different M/F energy ratios. | `src/ai/train.py` |
| 6 — Priority masking | Priority masking in `separation.py`: UNCERTAIN_CAP=0.25 for ambiguous frames, HARMONIC_FLOOR=0.85 for confirmed female harmonics, MALE_SUPPRESSION=0.08 for detected male harmonics. | `src/dsp/separation.py` |
| 7 — NMF separation | New `nmf_separation.py` module: decomposes the spectrogram into K=8 NMF components, assigns each component to F or M via dominant-frame scoring, builds a per-bin IRM. | `src/dsp/nmf_separation.py` |
| 8 — Hybrid IRM | The NMF IRM uses a linear soft mask `V_f/(V_f+V_m)` (avoids energy-imbalance collapse from squaring) blended at 65% with the classifier's attention weights. Mean IRM: from 0.16 to 0.43+ on test samples. | `src/dsp/nmf_separation.py` |

---

## Next Tasks (priority order)

1. ~~**Automated tests** — `tests/test_pipeline.py`~~ ✅ Done (6 tests, 26 total passing)
2. ~~**Evaluation notebook** — `notebooks/02_evaluation.ipynb`~~ ✅ Done (SI-SDR +3.1 dB, PESQ +0.036, STOI −0.007)
3. ~~**Phase 6 complete**~~ ✅ Done → next: tag release **v0.1.1**
