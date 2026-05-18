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

---

## Architecture

```
                  ┌───────────────────────┐
   Audio mix ───► │  DSP: feature extract │  (STFT, MFCC, pitch, energy)
                  └───────────┬───────────┘
                              │ features
                              ▼
                  ┌───────────────────────┐
                  │  AI: attention module │  (target speaker classifier)
                  └───────────┬───────────┘
                              │ per-frame attention mask
                              ▼
                  ┌───────────────────────┐
                  │  DSP: separation +    │  (mask application, ISTFT,
                  │  reconstruction       │   speech enhancement)
                  └───────────┬───────────┘
                              ▼
                       Isolated audio
```

---

## How It Works

### 1. Pre-processing
The input audio is loaded, resampled to 16 kHz, and transformed into the time-frequency domain via **Short-Time Fourier Transform (STFT)**.

### 2. Feature Extraction — 44 features per frame
| Feature group | Count | Description |
|---|---|---|
| MFCC | 13 | Timbral envelope |
| MFCC Δ | 13 | First-order temporal derivative |
| MFCC ΔΔ | 13 | Second-order temporal derivative |
| Pitch (F0) | 1 | Female vocal range only: 150–310 Hz via pYIN |
| RMS energy | 1 | Signal intensity |
| Spectral centroid | 1 | Timbral brightness |
| Spectral rolloff | 1 | High-frequency energy distribution |
| ZCR | 1 | Voiced/unvoiced indicator |

### 3. Attention Module
The attention module combines two complementary models:

- **MLP classifier** (256 → 128 → 64, ReLU, early stopping) + StandardScaler trained on IBM frame-level labels from LibriSpeech mixtures at variable SNR (−3, 0, +3 dB). Each sample is a sliding window of 11 consecutive frames (484 features total), giving the model temporal context. Training on mixture frames — not isolated voices — eliminates the train/inference domain mismatch.
- **GenderGMM** — two Gaussian Mixture Models (16 components, diagonal covariance) trained on clean (non-mixed) LibriSpeech speech, one per gender. At inference, the log-likelihood ratio `log P(X|GMM_F) − log P(X|GMM_M)` is sigmoid-normalised to P(female) ∈ [0, 1]. Complementary to the MLP: captures the marginal acoustic distribution of each gender rather than the discriminative boundary on mixture frames.

The final per-frame mask blends both signals: `0.6 × MLP + 0.4 × GMM`. A 2-state **HMM** (F-dominant / M-dominant) then applies forward-backward smoothing with asymmetric transition probabilities (p_ff=0.95, p_mf=0.20), absorbing short male interruptions (< ~40 ms) and eliminating choppy mask artefacts.

### 4. NMF-Guided Separation
The separation module (`nmf_separation.py`) combines two complementary signals:

1. **NMF decomposition** — the magnitude spectrogram is factored into K=8 spectral bases via Non-negative Matrix Factorisation (V ≈ W×H).
2. **Dominant-frame scoring** — each NMF component k receives a "femaleness" score computed as the mean attention weight over the frames where k is dominant.
3. **Hybrid IRM** — the final per-bin mask blends 75% attention weights (reliable temporal F/M signal) with 25% NMF soft mask (per-frequency resolution). This prevents IRM collapse when the classifier is uncertain.
4. **Pitch refinement** — confirmed female harmonic bins are raised to ≥ 0.85 (harmonic floor); confirmed male harmonic bins are suppressed to ≤ 0.08 (male suppression).
5. **IRM selective floor** — a global floor of 0.15 is applied after pitch refinement, only to bins not subject to explicit male harmonic suppression and only when the mean attention weight ≥ 0.25. Prevents spectral holes in male-dominant frames that degrade short-time intelligibility.

### 5. Reconstruction
The IRM is applied to the complex STFT (preserving the original phase), then **ISTFT** converts back to the time domain. A final **speech enhancement** step (`noisereduce` + peak normalisation) cleans up residual artefacts.

---

## Requirements

- **Python 3.10+**
- **OS:** Linux, macOS, or Windows
- **RAM:** 4 GB recommended

### Main Python dependencies

| Library | Purpose |
|---|---|
| `numpy`, `scipy` | Numerical operations and DSP |
| `librosa`, `soundfile` | Audio I/O and feature extraction |
| `scikit-learn` | Lightweight ML models |
| `noisereduce` | Spectral noise reduction |
| `matplotlib` | Spectrograms and diagnostics |
| `pytest` | Unit testing |

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

### End-to-end pipeline on an external file

```bash
python -m src.pipeline --input mix.wav --model models/classifier.joblib --output out.wav
```

### Train the models

```bash
# MLP classifier — IBM multi-SNR training (~10-20 min):
python -m src.ai.train --out models/classifier.joblib

# With cross-validation (~1-1.5 h):
python -m src.ai.train --cv-folds 5 --out models/classifier.joblib

# GenderGMM — trained on clean LibriSpeech clips (~1 min):
python -m src.ai.train_gmm --out models/gender_gmm.joblib
```

### End-to-end pipeline with GMM blend

```bash
python -m src.pipeline --input mix.wav --model models/classifier.joblib --gmm models/gender_gmm.joblib --output out.wav
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
│   │   ├── features.py        # 44-feature extraction per frame
│   │   ├── dataset.py         # LibriSpeech loader, M+F mixer, IBM dataset builder
│   │   ├── enhancement.py     # Speech enhancement: noisereduce + peak normalisation
│   │   ├── separation.py      # T-F masking utilities (ratio mask, pitch mask)
│   │   └── nmf_separation.py  # Primary separation module (classifier-guided NMF)
│   │
│   ├── ai/
│   │   ├── classifier.py      # SpeakerClassifier (MLP + StandardScaler)
│   │   ├── attention.py       # AttentionModule: MLP+GMM blend + HMM smoothing
│   │   ├── smoothing.py       # hmm_smooth(): 2-state forward-backward HMM
│   │   ├── gmm_classifier.py  # GenderGMM: LLR = log P(X|GMM_F) - log P(X|GMM_M)
│   │   ├── train.py           # Multi-SNR IBM training script
│   │   └── train_gmm.py       # GenderGMM training script
│   │
│   ├── pipeline.py            # End-to-end CLI
│   └── utils.py               # Audio I/O utilities
│
├── data/
│   ├── raw/librispeech/       # LibriSpeech dev-clean (not tracked by git)
│   └── processed/demo/        # Demo output files
│
├── models/                    # Trained models (not tracked by git)
├── notebooks/
│   ├── 02_evaluation.ipynb    # Quantitative metrics: SI-SDR, PESQ, STOI
│   └── 03_diagnosis.ipynb     # Classifier vs masking stage diagnosis
├── tests/
│   ├── test_utils.py
│   ├── test_features.py
│   └── test_pipeline.py       # End-to-end integration tests (26 tests)
└── docs/                      # Theoretical documentation
```

---

## Theoretical References

- **Cherry, E. C.** (1953). *Some Experiments on the Recognition of Speech, with One and with Two Ears.* Journal of the Acoustical Society of America, 25(5), 975–979. — Original definition of the cocktail party problem.
- **Bregman, A. S.** (1990). *Auditory Scene Analysis: The Perceptual Organization of Sound.* MIT Press. — Theoretical foundation of auditory perception and source segregation.
- **Wang, D., & Brown, G. J.** (2006). *Computational Auditory Scene Analysis: Principles, Algorithms, and Applications.* Wiley-IEEE Press.
- **Hyvärinen, A., & Oja, E.** (2000). *Independent Component Analysis: Algorithms and Applications.* Neural Networks, 13(4–5), 411–430.

---

## License

To be defined.
