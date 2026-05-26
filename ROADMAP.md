# ROADMAP — Implementation Plan

> **Status:** in progress — Phases 0–6 complete. v0.2.0 tagged. v0.2.1: SI-SDR loss, FiLM conditioning, DPCRN, GRU smoother — code complete, bugs fixed, models require retraining. v0.3.0 will be tagged after metric validation.
> **Last updated:** 2026-05-26.

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
| 1 | ✅ Done | `src/dsp/stft.py` + `src/dsp/features.py` (56 features: MFCC+Δ+ΔΔ+pitch+rms+centroid+rolloff+ZCR+LPC) + `tests/test_features.py` |
| 2 | ✅ Done | `src/dsp/dataset.py` — SPEAKERS.TXT parsing, M+F mix, multi-SNR IBM dataset |
| 3 | ✅ Done | `src/ai/classifier.py`, `train.py`, `attention.py` — CV on 400 samples × 3 SNR values |
| 4 | ✅ Done | `src/dsp/nmf_separation.py` (primary module) + `separation.py` (utilities/fallback) |
| 5 | ✅ Done | `src/dsp/enhancement.py` — log-MMSE (Ephraim & Malah 1985) + minimum-statistics + peak normalisation |
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

1. ~~**Automated tests** — `tests/test_pipeline.py`~~ ✅ Done (6 tests, 30 total passing)
2. ~~**Evaluation notebook** — `notebooks/02_evaluation.ipynb`~~ ✅ Done (SI-SDR +3.1 dB, PESQ +0.036, STOI −0.007)
3. ~~**Phase 6 complete**~~ ✅ Done → released as **v0.1.1**
4. ~~**MaskNet CNN** — code implemented~~ ✅ Done (2026-05-19)
5. ~~**LPC features (order 12)**~~ ✅ Done (2026-05-19) — `N_FEATURES` 44 → 56, 4 new tests
6. ~~**Log-MMSE enhancement**~~ ✅ Done (2026-05-19) — replaces `noisereduce`, pure numpy/scipy
7. ~~**Retrain classifier + GMM**~~ ✅ Done (2026-05-22) — N_FEATURES 44→56
8. ~~**Train MaskNet**~~ ✅ Done (2026-05-22) — val_loss=0.147, desktop GPU
9. ~~**Evaluate MaskNet**~~ ✅ Done (2026-05-22) — SI-SDR +3.945 dB, aurally confirmed → **v0.2.0**

### Completed (2026-05-22)

| # | Task | Result |
|---|---|---|
| 1 | **Retrain classifier + GMM** (N_FEATURES 44→56) | ✅ Done — `classifier.joblib`, `gender_gmm.joblib` regenerated |
| 2 | **Train MaskNet** on desktop GPU (12GB VRAM) | ✅ Done — `mask_net.pt`, best val_loss=0.147 |
| 3 | **Evaluate MaskNet** vs baseline | ✅ Done — SI-SDR +3.945 dB over baseline, audibly confirmed |

### v0.2.1 — Code complete (2026-05-24 committed, bugs fixed 2026-05-26)

> **v0.3.0 will be tagged only after metric validation on a trained model.**

| # | Task | Status | Notes |
|---|---|---|---|
| 1 | **SI-SDR loss for MaskNet/DPCRN** | ✅ Code done | `train_mask_net.py`: loss `combined` (default) = 0.7×neg_SI-SDR + 0.3×MSE. `torch.istft` differentiable in loop. |
| 2 | **FiLM gender conditioning on MaskNet** | ✅ Code done | `mask_net.py`: `_FiLMBlock` + `nn.Embedding(2,16)`. ~82K params (was ~75K). `MaskNet.refine(…, gender=0)`. |
| 3 | **GRU smoother (replaces HMM)** | ✅ Code done | `smoothing_gru.py`: BiGRU ~26K params. `train_smoothing.py`: BCE on IBM sequences. `attention.py`: auto-selects GRU if loaded. |
| 4 | **DPCRN architecture** | ✅ Code done | `dpcrn.py`: 8 DualPath blocks (freq Conv2d + time GRU), ~302K params. Drop-in for MaskNet. `--model-type dpcrn` in `train_mask_net.py`. `--dpcrn` in `pipeline.py`. |

**Bugs fixed (2026-05-26, smoke tests on Mac MPS):**
- `train_smoothing.py` — `raw_mask` (float64) cast to float32 before tensor creation; MPS rejects float64.
- `dpcrn.py` — added `.contiguous()` after each `.permute()` in `_DualPathBlock.forward()`; MPS backward pass requires contiguous tensors.

**Training feasibility on Mac MPS (Apple Silicon):**
| Model | Params | Estimated time (200 samples × 3 SNR × 50 epochs) |
|---|---|---|
| GRUSmoother | 26K | ~40 min |
| MaskNet | 82K | ~2–3 h |
| DPCRN | 302K | ~100 h — **GPU desktop required** |

### Next steps (retraining required for v0.3.0 validation)

```bash
# On Mac (MPS): train MaskNet and GRU smoother
python -m src.ai.train_mask_net --classifier models/classifier.joblib \
    --gmm models/gender_gmm.joblib --n-samples 200 --loss combined --out models/mask_net.pt

python -m src.ai.train_smoothing --classifier models/classifier.joblib \
    --gmm models/gender_gmm.joblib --n-samples 200 --out models/smoothing_gru.pt

# On GPU desktop only: train DPCRN
python -m src.ai.train_mask_net --model-type dpcrn --classifier models/classifier.joblib \
    --gmm models/gender_gmm.joblib --n-samples 200 --loss combined --out models/dpcrn.pt
```

Then evaluate all variants with `notebooks/02_evaluation.ipynb` and compare vs v0.2.0 baseline (+3.945 dB SI-SDR).
Tag v0.3.0 only if at least one variant improves the SI-SDR metric.

### Research: candidate improvements to MaskNet / separation stage (2026-05-23)

Four approaches identified during baseline investigation. Listed from lowest to highest implementation cost.

#### Option A — SI-SDR loss for MaskNet (recommended first step)

Replace the current MSE training loss in `train_mask_net.py` with SI-SDR loss, optionally combined as `0.7 * SI-SDR_loss + 0.3 * MSE`:

- **Why it helps:** MSE minimises per-bin squared error, which does not correlate strongly with perceived separation quality. SI-SDR is the primary evaluation metric — aligning training and evaluation loss removes the optimisation mismatch.
- **SI-SDR loss formulation:** `L = −SI-SDR(target_waveform, reconstructed_waveform)`. Requires applying the mask to the STFT, performing ISTFT inside the training loop (differentiable via `torch.stft`/`torch.istft`), and computing SI-SDR on the output waveform.
- **Cost:** ~30 lines of change in `train_mask_net.py` + `mask_net.py`. Requires GPU retraining (~same time as current training).
- **Files to change:** `src/ai/train_mask_net.py`, `src/ai/mask_net.py`.

#### Option B — Gender conditioning on MaskNet

Add a learnable gender embedding (2-class → 16-dim) as a conditioning signal to MaskNet:

- **Why it helps:** the current MaskNet receives the attention weights as a channel but has no explicit knowledge of the target gender. Conditioning on a gender vector biases the mask refinement toward the expected spectral characteristics of the target (female: higher harmonics, narrower pitch range; male: lower fundamental, more energy below 500 Hz). This also strengthens the theoretical link to "cocktail party attention".
- **Architecture change:** embed `gender ∈ {0, 1}` → 16-dim vector via `nn.Embedding`; broadcast and concatenate as a 4th input channel (or use FiLM-style affine conditioning on each conv layer).
- **Cost:** ~5K extra params. `build_input()` in `mask_net.py` gains a `gender` argument; `train_mask_net.py` extracts gender from speaker metadata. No change to pipeline CLI.
- **Files to change:** `src/ai/mask_net.py`, `src/ai/train_mask_net.py`.

#### Option C — GRU temporal smoothing (replace HMM)

Replace `hmm_smooth()` in `src/ai/smoothing.py` with a 1-layer bidirectional GRU (~30K params):

- **Why it helps:** the HMM uses fixed transition probabilities (`p_ff=0.95`, `p_mf=0.20`) tuned manually. A GRU learns optimal temporal dynamics from data, handling variable-length speaker turns and partial overlaps better than a fixed-topology HMM.
- **Integration:** `GRUSmooother` class with a `smooth(mask: np.ndarray) -> np.ndarray` interface, drop-in replacement for `hmm_smooth()`. Requires a short training script (supervised on IBM frame sequences from the existing dataset).
- **Cost:** new file `src/ai/smoothing_gru.py` + `train_smoothing.py`. Adds a third model artifact (`smoothing_gru.pt`). Pipeline needs a `--smoothing-gru` CLI flag.
- **Files to change/add:** `src/ai/smoothing_gru.py` (new), `src/ai/train_smoothing.py` (new), `src/ai/attention.py`, `src/pipeline.py`.

#### Option D — DPCRN / Conv-TasNet as MaskNet replacement

Replace the 5-layer CNN (75K params) with a DPCRN (~300K, T-F domain) or Conv-TasNet (time domain):

- **DPCRN:** dilated convolutions + gated RNN, operates on the STFT magnitude. Nearest-drop-in replacement for MaskNet — same input/output format. Better long-range frequency context than plain Conv2d. ~300K params, CPU-feasible with quantization.
- **Conv-TasNet:** end-to-end time-domain model, bypasses STFT entirely. Best theoretical performance but breaks the existing STFT-based pipeline; NMF/IRM preprocessing cannot be used as input. Higher complexity (~2M params in typical configs).
- **Recommendation:** start with DPCRN if this option is chosen — same T-F domain, compatible with the existing `_build_irm()` pipeline.
- **Cost:** high. New architecture file, full retraining, potential pipeline restructure for Conv-TasNet.

---

### Baseline regression investigation (closed — 2026-05-23)

The −0.228 dB figure from 2026-05-22 was **not a systematic regression**. Root cause: the old +2.680 dB was measured on a single easy demo sample (N_FEATURES=44 models), while −0.228 was measured on 6 harder samples with retrained models — comparing different evaluation sets.

Fresh 6-sample evaluation with current models (seed=99, SNR≈0 dB):

| Configuration | Avg SI-SDR (dB) | Δ vs mix |
|---|---|---|
| Mix (input) | +0.096 | — |
| MLP only (no GMM) | +1.442 | **+1.346** |
| MLP + GMM (w=0.4) | +1.762 | **+1.666** |

Observations:
- **GMM bias**: on mixture input, `gmm_proba mean ≈ 0.393` (slightly male-biased), caused by LPC coefficients of a 2-speaker mix not matching the clean-speech GMM distribution. Despite this, the blend still helps (+0.32 dB avg). No fix needed.
- **Occasional failures**: some samples yield negative ΔSI-SDR (e.g. −2.49 dB), likely where NMF decomposition fails or M/F pitch ranges overlap. These are not caused by the retrained models.
- **Comments fixed**: stale "44-dim" comments in `attention.py` and `train_gmm.py` updated to `N_FEATURES`.

## Post-release improvements (v0.2.0)

### Diagnosis (2026-05-16)
Classifier diagnosed as the bottleneck: 56.9% uncertain frames, IBM accuracy 74.6%, recall on F-frames only 67.2%.
Root cause: no temporal context (single-frame features) + insufficient training data.

### Implemented (2026-05-16)
- `src/dsp/features.py` — added `apply_window()` and `extract_windowed()` (sliding context window, W=11 frames → 484 features)
- `src/ai/classifier.py` — replaced Random Forest with MLP (256→128→64), added `window_size` param, updated save/load
- `src/ai/attention.py` — auto-selects windowed vs flat feature extraction based on `classifier.window_size`
- `src/dsp/dataset.py` — `make_ibm_dataset()` now accepts `window_size`
- `src/ai/train.py` — new defaults: 400 samples × 3 SNR values (−3, 0, +3 dB), `--window-size 11`, CV opt-in via `--cv-folds`

### Re-train and evaluate (✅ Done — 2026-05-17)
Results: uncertain frames 2.7% (was 56.9%), IBM accuracy 67.9%, F-recall 72.0%.
Root cause shifted to **masking stage** (NMF/IRM). SI-SDR +2.56 dB (PASS), STOI −0.116 (open issue).

### Implemented (2026-05-17) — HMM smoothing + IRM tuning
- `src/ai/smoothing.py` — new module: `hmm_smooth()`, 2-state HMM forward-backward. Eliminates mask choppiness and biases toward F via asymmetric transitions (p_ff=0.95, p_mf=0.20).
- `src/ai/attention.py` — `compute_mask()` now accepts `smooth=True` (default); applies HMM smoothing post-classifier.
- `src/dsp/nmf_separation.py` — IRM tuning:
  - Blend ratio 65/35 → **75/25** (trust classifier more since it is now reliable)
  - `female_weights` range [0.15, 0.85] → **[0.05, 0.95]** (M-scored NMF components contribute ~0 to female reconstruction)
  - Note: IRM_ATTN_SHARPENING removed after testing — it amplified misclassified frames and worsened output.

### Implemented (2026-05-17) — GMM likelihood ratio (solution #2)
- `src/ai/gmm_classifier.py` — new `GenderGMM` class: two `GaussianMixture` (16 components, diag) trained on CLEAN speech, LLR = log P(X|GMM_F) − log P(X|GMM_M), sigmoid-normalised → P(female) ∈ [0,1].
- `src/ai/train_gmm.py` — training script: loads clean LibriSpeech clips per gender, extracts 44-dim features, fits GenderGMM, saves to `models/gender_gmm.joblib`.
- `src/ai/attention.py` — `AttentionModule.__init__()` now accepts optional `gmm` and `gmm_weight=0.4`; `compute_mask()` blends `(1−0.4)*mlp_mask + 0.4*gmm_proba` before HMM smoothing.
- `src/pipeline.py` — new `--gmm` flag to pass the GMM path at inference.
- `demo.py` — auto-loads `models/gender_gmm.joblib` if present.

### Implemented (2026-05-17) — IRM selective floor (STOI improvement)
Diagnosis: after all previous improvements, STOI remained negative (−0.069 from −0.098). Root cause: in male-dominant frames the IRM collapses near zero for bins that still carry female energy (unvoiced consonants, broadband fricatives), creating spectral holes that hurt short-time intelligibility.

- `src/dsp/nmf_separation.py` — added `IRM_FLOOR = 0.15`: a selective floor applied after all masking steps.
  - Applied only to bins NOT subject to explicit male harmonic suppression (so `MALE_SUPPRESSION = 0.08` is preserved for confirmed male harmonics).
  - Gated on `attention_weights.mean() >= 0.25`: skipped when the classifier is almost certain everything is male, to avoid adding male leakage in degenerate cases.
  - Effect: STOI −0.098 → −0.069 (+0.029), PESQ +0.006 → +0.030, SI-SDR +2.540 → +2.680 dB.

**Note — priority fix tested and reverted:** an additional change (female harmonic bins override male suppression) was tested but caused pYIN to amplify wrongly-detected F0 bins in edge cases (e.g., synthetic sine mixtures where pYIN picks a spurious F0). Reverted to preserve test suite stability and safety margin.

**Current metrics (2026-05-17, all 26 tests passing):**

| Metric | Baseline (mix) | System | Δ |
|---|---|---|---|
| SI-SDR (dB) | 0.020 | 2.700 | **+2.680** |
| PESQ | 1.092 | 1.122 | **+0.030** |
| STOI | 0.612 | 0.543 | −0.069 ⚠️ |

---

---

## Implemented (2026-05-19) — MaskNet CNN (AI extension)

Context: the project now explicitly covers two academic areas — *Analisi Intelligente dei Segnali* (DSP layer) and *Intelligenza Artificiale* (AI layer). MaskNet is the main AI-layer upgrade: a learned CNN that refines the NMF-IRM output.

### Architecture
- **Input:** 3-channel tensor (B, 3, 257, T)
  - Channel 0: log-magnitude spectrogram (standardised)
  - Channel 1: per-frame attention weights broadcast to (F, T)
  - Channel 2: NMF-IRM from the classical pipeline
- **Network:** 5 fully-convolutional Conv2d layers with BatchNorm + ReLU, output 1×1 conv + Sigmoid
- **Output:** refined mask in [0, 1], shape (B, 257, T)
- **Parameters:** ~75K — runs in real-time on CPU and Apple MPS (M-series chips)
- **Training target:** IRM computed from clean sources: `|F(f,t)|² / (|F(f,t)|² + |M(f,t)|²)`
- **Loss:** MSE. Optimiser: Adam + CosineAnnealingLR

### Changes
| File | Change |
|---|---|
| `src/ai/mask_net.py` | New — MaskNet model class + inference wrapper + `build_input()` |
| `src/ai/train_mask_net.py` | New — second-stage training script (requires classifier + GMM) |
| `src/dsp/nmf_separation.py` | Refactored — IRM logic extracted to `_build_irm()`, added `compute_nmf_irm()` (public, for training), added `mask_net` param to `separate_nmf()` |
| `src/pipeline.py` | Added `--mask-net` CLI argument |
| `requirements.txt` / `pyproject.toml` | Added `torch>=2.0` as optional extra (`pip install 'auralis[torch]'`) |

### Training pipeline (second-stage)
```
Stage 1: train MLP classifier     →  models/classifier.joblib
Stage 1: train GenderGMM          →  models/gender_gmm.joblib
Stage 2: train MaskNet            →  models/mask_net.pt
         (feeds stage-1 outputs as inputs + IRM target from clean sources)
```

### Status
- ✅ Code implemented and tested (30/30 tests passing)
- ✅ MaskNet trained — desktop GPU (12GB VRAM), best val_loss=0.147, 200 samples × 50 epochs
- ✅ Evaluated vs baseline (2026-05-22, 6 samples, SNR=0 dB)

| Metric | Mix (input) | No MaskNet | With MaskNet | MaskNet Δ |
|---|---|---|---|---|
| SI-SDR (dB) | −0.068 | −0.228 | **+3.718** | **+3.945** |

> ⚠️ Baseline regression noted: pipeline without MaskNet dropped from +2.680 dB (v0.1 models, N_FEATURES=44) to −0.228 dB (retrained models, N_FEATURES=56). Root cause under investigation.

---

## Implemented (2026-05-19) — LPC features (DSP coverage Cap. 10)

Linear Predictive Coding (order 12) added to the feature set. LPC models the vocal tract as a 12th-order all-pole filter via the autocorrelation method (Levinson-Durbin recursion). The coefficients capture formant structure, complementary to the cepstral MFCC representation.

| File | Change |
|---|---|
| `src/dsp/features.py` | New `N_LPC = 12`, `extract_lpc()`, integrated into `extract_all()` |
| `src/dsp/features.py` | `N_FEATURES` updated 44 → 56 |
| `tests/test_features.py` | 4 new LPC tests (shape, no-NaN, silence, short-audio) |

**Breaking change:** `N_FEATURES` 44 → 56 → `classifier.joblib` and `gender_gmm.joblib` must be retrained (windowed features: 484 → 616 dims).

---

## Implemented (2026-05-19) — Log-MMSE enhancement (DSP coverage Cap. 11)

Replaced `noisereduce` with a pure numpy/scipy implementation of the log-MMSE spectral amplitude estimator (Ephraim & Malah 1985) with decision-directed a priori SNR estimation (Ephraim & Malah 1984) and minimum-statistics noise PSD tracking (Martin 2001).

| File | Change |
|---|---|
| `src/dsp/enhancement.py` | Full rewrite — `_estimate_noise_psd()`, `_log_mmse_gain()`, `mmse_stsa_enhance()` |
| `requirements.txt` | Removed `noisereduce>=3.0` |
| `pyproject.toml` | Removed `noisereduce>=3.0` from dependencies |

**Algorithm:** `G(ξ,γ) = ξ/(1+ξ) · exp(½·E₁(ν))`, `ν = ξγ/(1+ξ)`, where `E₁` is the exponential integral (`scipy.special.expn`). Decision-directed SNR: `ξ[t] = α·G[t-1]²·γ[t-1] + (1-α)·max(γ[t]-1, 0)`, `α=0.98`. Floor `γ ≥ GAMMA_MIN=2.0` prevents over-suppression of stationary signals.

---

## Candidate Solutions to Validate (classifier bottleneck)

The following approaches were identified as potential fixes for the classifier bottleneck (74.6% IBM accuracy, 56.9% uncertain frames, 67.2% F-recall). Each must be evaluated for actual impact before integration.

| # | Method | Target problem | Library | Status |
|---|---|---|---|---|
| 1 | HMM smoothing | Uncertain frames / temporal incoherence | `hmmlearn` | ✅ Done |
| 2 | GMM likelihood ratio | Weak M/F discrimination | `sklearn.mixture` | ✅ Done |
| 3 | FastICA pre-separation | Noisy features fed to classifier | `sklearn.decomposition` | ⬜ Not started |

### 1 — HMM Smoothing on classifier output

A 2-state HMM (F-dominant / M-dominant) models frame transitions and resolves ambiguous frames via Viterbi decoding. The classifier's per-frame probabilities become HMM emission probabilities; the Viterbi path replaces the raw frame-level decisions.

- **Why it helps:** uncertain frames are resolved by temporal context, not by the single-frame probability alone.
- **Integration point:** post-processing layer on top of `AttentionModule.compute_mask()` output, no change to existing architecture.
- **To validate:** does the Viterbi path reduce the uncertain-frame rate below 56.9%? Does SI-SDR improve?

### 2 — GMM Likelihood Ratio for gender modeling

Train two GMMs (`GaussianMixture`) on clean male and female speech features separately. At inference, compute the log-likelihood ratio `log P(frame|GMM_F) - log P(frame|GMM_M)` and use it as an additional feature or as a second-opinion decision criterion alongside the MLP.

- **Why it helps:** GMMs capture the global timbral distribution of each gender, not just single-frame snapshots. Complementary to the MLP.
- **Integration point:** `src/ai/classifier.py` or as a standalone `GenderGMM` module feeding into `attention.py`.
- **To validate:** does adding the LLR feature improve IBM accuracy above 74.6%? Does F-recall exceed 67.2%?

### 3 — FastICA pre-separation

Apply `FastICA` before feature extraction to decompose the mixture into statistically independent components. Feed the ICA-separated components to the classifier instead of (or alongside) the raw mix.

- **Why it helps:** gives the classifier cleaner input features with less cross-speaker contamination.
- **Limitation:** most effective with stereo input (2 channels → 2 sources mathematically guaranteed). Usefulness on mono must be verified experimentally.
- **Integration point:** `src/dsp/features.py` or as a pre-step in `src/pipeline.py`.
- **To validate:** does ICA pre-processing reduce feature noise? Does it help on mono or only on stereo?
