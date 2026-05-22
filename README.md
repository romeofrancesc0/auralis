# Auralis

> **AI-based selective auditory attention system inspired by the cocktail party problem.**
> Not just source separation — a simulation of the human cognitive mechanism that decides *which* speaker to listen to.

---

## The Problem

In a crowded room — a party, a café, a meeting — the human brain performs a remarkable feat: it focuses on a single voice while filtering out all others. This is the **cocktail party problem**, first formalised by Colin Cherry in 1953.

Traditional *source separation* systems approach it algorithmically: separate every audio source present in the signal. But they miss the crucial next step — the one that is truly human: **deciding which voice to pay attention to**.

Auralis addresses both sides of the problem:

- **Separation** (DSP) — isolate the acoustic components of each speaker.
- **Selective attention** (AI) — automatically select the target speaker based on acoustic features (pitch, energy, timbre, vocal gender), simulating the human auditory attention mechanism.

The project covers two academic areas: **Intelligent Signal Analysis** (DSP layer) and **Artificial Intelligence** (AI layer).

---

## Architecture

```
                  ┌────────────────────────┐
   Audio mix ───► │  DSP: feature extract  │  (STFT, MFCC, pitch, energy)
                  └───────────┬────────────┘
                              │ 56 features / frame
                              ▼
                  ┌────────────────────────┐
                  │  AI: attention module  │  MLP + GenderGMM + HMM smoothing
                  └───────────┬────────────┘
                              │ per-frame attention mask
                              ▼
                  ┌────────────────────────┐
                  │  DSP: NMF separation   │  classifier-guided NMF + IRM
                  └───────────┬────────────┘
                              │ NMF-IRM
                              ▼
                  ┌────────────────────────┐   (optional)
                  │  AI: MaskNet CNN       │  learned IRM refinement ~75K params
                  └───────────┬────────────┘
                              │ refined mask
                              ▼
                  ┌────────────────────────┐
                  │  DSP: reconstruction   │  ISTFT + speech enhancement
                  └───────────┬────────────┘
                              ▼
                       Isolated audio
```

---

## How It Works

### 1. Pre-processing
The input audio is loaded, resampled to 16 kHz, and transformed into the time-frequency domain via **Short-Time Fourier Transform (STFT)**.

### 2. Feature Extraction — 56 features per frame
| Feature group | Count | Description |
|---|---|---|
| MFCC | 13 | Timbral envelope (cepstral coefficients) |
| MFCC Δ | 13 | First-order temporal derivative |
| MFCC ΔΔ | 13 | Second-order temporal derivative |
| Pitch (F0) | 1 | Female vocal range only: 150–310 Hz via pYIN |
| RMS energy | 1 | Signal intensity |
| Spectral centroid | 1 | Timbral brightness |
| Spectral rolloff | 1 | High-frequency energy distribution |
| ZCR | 1 | Voiced/unvoiced indicator |
| LPC | 12 | Vocal tract all-pole model, order 12 (Levinson-Durbin) |

### 3. Attention Module
The attention module combines two complementary models:

- **MLP classifier** (256 → 128 → 64, ReLU, early stopping) + StandardScaler trained on IBM frame-level labels from LibriSpeech mixtures at variable SNR (−3, 0, +3 dB). Each sample is a sliding window of 11 consecutive frames (616 features total), giving the model temporal context. Training on mixture frames — not isolated voices — eliminates the train/inference domain mismatch.
- **GenderGMM** — two Gaussian Mixture Models (16 components, diagonal covariance) trained on clean (non-mixed) LibriSpeech speech, one per gender. At inference, the log-likelihood ratio `log P(X|GMM_F) − log P(X|GMM_M)` is sigmoid-normalised to P(female) ∈ [0, 1]. Complementary to the MLP: captures the marginal acoustic distribution of each gender rather than the discriminative boundary on mixture frames.

The final per-frame mask blends both signals: `0.6 × MLP + 0.4 × GMM`. A 2-state **HMM** (F-dominant / M-dominant) then applies forward-backward smoothing with asymmetric transition probabilities (p_ff=0.95, p_mf=0.20), absorbing short male interruptions (< ~40 ms) and eliminating choppy mask artefacts.

### 4. NMF-Guided Separation
The separation module (`nmf_separation.py`) combines two complementary signals:

1. **NMF decomposition** — the magnitude spectrogram is factored into K=8 spectral bases via Non-negative Matrix Factorisation (V ≈ W×H).
2. **Dominant-frame scoring** — each NMF component k receives a "femaleness" score computed as the mean attention weight over the frames where k is dominant.
3. **Hybrid IRM** — the final per-bin mask blends 75% attention weights (reliable temporal F/M signal) with 25% NMF soft mask (per-frequency resolution). This prevents IRM collapse when the classifier is uncertain.
4. **Pitch refinement** — confirmed female harmonic bins are raised to ≥ 0.85 (harmonic floor); confirmed male harmonic bins are suppressed to ≤ 0.08 (male suppression).
5. **IRM selective floor** — a global floor of 0.15 is applied after pitch refinement, only to bins not subject to explicit male harmonic suppression and only when the mean attention weight ≥ 0.25. Prevents spectral holes in male-dominant frames that degrade short-time intelligibility.

### 5. MaskNet CNN Refinement *(optional)*
A lightweight fully-convolutional network (~75K parameters) refines the NMF-IRM using three input channels:

| Channel | Content |
|---|---|
| 0 | Log-magnitude spectrogram (zero-mean, unit-variance) |
| 1 | Attention weights broadcast across frequency |
| 2 | NMF-IRM as-is — the classical-pipeline prior |

Trained against the ideal IRM computed from clean sources (`IRM_target(f,t) = |F|² / (|F|² + |M|²)`), using MSE loss. Runs on CPU, CUDA, and Apple MPS (M-series chips) without code changes. Requires `torch>=2.0`.

### 6. Reconstruction
The (optionally refined) IRM is applied to the complex STFT (preserving the original phase), then **ISTFT** converts back to the time domain. A final **speech enhancement** step applies the log-MMSE spectral amplitude estimator (Ephraim & Malah 1985) with decision-directed a priori SNR estimation and minimum-statistics noise tracking, followed by peak normalisation.

---

## Requirements

- **Python 3.10+**
- **OS:** Linux, macOS, or Windows
- **RAM:** 4 GB recommended
- **GPU:** optional — required only for MaskNet training; inference runs on CPU/MPS

### Main Python dependencies

| Library | Purpose |
|---|---|
| `numpy`, `scipy` | Numerical operations, DSP, log-MMSE enhancement |
| `librosa`, `soundfile` | Audio I/O and feature extraction |
| `scikit-learn` | MLP, GMM, StandardScaler |
| `matplotlib` | Spectrograms and diagnostics |
| `pytest` | Unit testing |
| `torch>=2.0` *(optional)* | MaskNet training and inference |

See `requirements.txt` for the full list with pinned versions.

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd auralis

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
# venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -e ".[dev]"

# 4. (Optional) Install PyTorch for MaskNet
pip install -e ".[torch]"
```

> **Dataset:** download LibriSpeech `dev-clean` from [openslr.org/12](https://www.openslr.org/12/) and place it under `data/raw/librispeech/dev-clean/`.

---

## Usage

### Quick demo

```bash
python demo.py
```

Automatically generates a male/female mixture from LibriSpeech, runs the full pipeline, and saves four files to `data/processed/demo/`:

| File | Description |
|---|---|
| `mix.wav` | Original mixture (F + M) |
| `target.wav` | Female voice — ground truth |
| `interferer.wav` | Male voice — ground truth |
| `output.wav` | System output — extracted female voice |

### Train the models

```bash
# Stage 1a — MLP classifier (IBM multi-SNR, ~10-20 min):
python -m src.ai.train --out models/classifier.joblib

# Stage 1b — GenderGMM on clean LibriSpeech clips (~1 min):
python -m src.ai.train_gmm --out models/gender_gmm.joblib

# Stage 2 — MaskNet CNN (requires stage 1a + 1b, GPU recommended, ~30-60 min):
python -m src.ai.train_mask_net \
    --classifier models/classifier.joblib \
    --gmm models/gender_gmm.joblib \
    --n-samples 200 \
    --out models/mask_net.pt
```

### End-to-end pipeline

```bash
# Baseline (MLP + GMM + NMF):
python -m src.pipeline \
    --input mix.wav \
    --model models/classifier.joblib \
    --gmm models/gender_gmm.joblib \
    --output out.wav

# With MaskNet refinement (best quality):
python -m src.pipeline \
    --input mix.wav \
    --model models/classifier.joblib \
    --gmm models/gender_gmm.joblib \
    --mask-net models/mask_net.pt \
    --output out.wav
```

### Run tests

```bash
pytest tests/
```

---

## Project Structure

```
auralis/
├── README.md
├── ROADMAP.md                 # Development plan and progress tracking
├── requirements.txt
├── demo.py                    # Quick demo script
│
├── src/
│   ├── dsp/
│   │   ├── stft.py            # STFT / ISTFT (centralised parameters)
│   │   ├── features.py        # 56-feature extraction per frame (MFCC, pitch, LPC, ...)
│   │   ├── dataset.py         # LibriSpeech loader, M+F mixer, IBM dataset builder
│   │   ├── enhancement.py     # Log-MMSE speech enhancement (Ephraim & Malah 1985)
│   │   ├── separation.py      # T-F masking utilities (ratio mask, pitch mask)
│   │   └── nmf_separation.py  # Primary separation module (classifier-guided NMF + MaskNet hook)
│   │
│   ├── ai/
│   │   ├── classifier.py      # SpeakerClassifier (MLP + StandardScaler)
│   │   ├── attention.py       # AttentionModule: MLP+GMM blend + HMM smoothing
│   │   ├── smoothing.py       # hmm_smooth(): 2-state forward-backward HMM
│   │   ├── gmm_classifier.py  # GenderGMM: LLR = log P(X|GMM_F) - log P(X|GMM_M)
│   │   ├── mask_net.py        # MaskNet: CNN for learned IRM refinement (~75K params)
│   │   ├── train.py           # Multi-SNR IBM training script (MLP)
│   │   ├── train_gmm.py       # GenderGMM training script
│   │   └── train_mask_net.py  # MaskNet training script (second-stage)
│   │
│   ├── pipeline.py            # End-to-end CLI
│   └── utils.py               # Audio I/O utilities
│
├── data/
│   ├── raw/librispeech/       # LibriSpeech dev-clean (not tracked by git)
│   └── processed/demo/        # Demo output files
│
├── models/                    # Trained models (not tracked by git)
│   │                          # classifier.joblib, gender_gmm.joblib, mask_net.pt
├── notebooks/
│   ├── 02_evaluation.ipynb    # Quantitative metrics: SI-SDR, PESQ, STOI
│   └── 03_diagnosis.ipynb     # Classifier vs masking stage diagnosis
├── tests/
│   ├── test_utils.py
│   ├── test_features.py
│   └── test_pipeline.py       # End-to-end integration tests (30 tests total)
└── docs/                      # Theoretical documentation
```

---

## Theoretical References

- **Cherry, E. C.** (1953). *Some Experiments on the Recognition of Speech, with One and with Two Ears.* Journal of the Acoustical Society of America, 25(5), 975–979. — Original definition of the cocktail party problem.
- **Bregman, A. S.** (1990). *Auditory Scene Analysis: The Perceptual Organization of Sound.* MIT Press. — Theoretical foundation of auditory perception and source segregation.
- **Wang, D., & Brown, G. J.** (2006). *Computational Auditory Scene Analysis: Principles, Algorithms, and Applications.* Wiley-IEEE Press.
- **Ephraim, Y., & Malah, D.** (1984). *Speech Enhancement Using a Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator.* IEEE Transactions on Acoustics, Speech, and Signal Processing, 32(6), 1109–1121. — MMSE-STSA estimator; basis for the decision-directed a priori SNR approach.
- **Ephraim, Y., & Malah, D.** (1985). *Speech Enhancement Using a Minimum Mean-Square Error Log-Spectral Amplitude Estimator.* IEEE Transactions on Acoustics, Speech, and Signal Processing, 33(2), 443–445. — Log-MMSE gain function used in the enhancement module.
- **Martin, R.** (2001). *Noise Power Spectral Density Estimation Based on Optimal Smoothing and Minimum Statistics.* IEEE Transactions on Speech and Audio Processing, 9(5), 504–512. — Minimum-statistics noise PSD estimator.
- **Hyvärinen, A., & Oja, E.** (2000). *Independent Component Analysis: Algorithms and Applications.* Neural Networks, 13(4–5), 411–430.

---

## License

To be defined.
