# Auralis

> **AI-based selective auditory attention system inspired by the cocktail party problem.**
> Not just source separation — a simulation of the human cognitive mechanism that decides *which* speaker to listen to.

---

## The Problem

In a crowded room — a party, a café, a meeting — the human brain performs a remarkable feat: it focuses on a single voice while filtering out all others. This is the **cocktail party problem**, first formalised by Colin Cherry in 1953.

Traditional *source separation* systems approach it algorithmically: separate every audio source present in the signal. But they miss the crucial next step — the one that is truly human: **deciding which voice to pay attention to**.

Auralis addresses both sides of the problem:

- **Separation** (DSP + AI) — the DPCRNSeparator segregates the acoustic scene into individual streams bottom-up, using the STFT as its analysis domain.
- **Selective attention** (AI) — the attention module decides which stream corresponds to the target speaker, top-down, simulating the human auditory attention mechanism.

---

## Architecture — Separate-then-Select

```
                  ┌──────────────────────────────┐
   Audio mix ───► │  DPCRNSeparator (uPIT)       │  bottom-up segregation
                  │  STFT → 2 soft masks → ISTFT │  ~301K params, no Griffin-Lim
                  └──────────────┬───────────────┘
                                 │ stream A, stream B
                                 ▼
                  ┌──────────────────────────────┐
                  │  AttentionModule             │  top-down selection
                  │  MLP + GenderGMM score       │  (or pitch-based, language-agnostic)
                  └──────────────┬───────────────┘
                                 │ selected stream
                                 ▼
                           Isolated audio
```

The **bottom-up / top-down** split mirrors the two-stage model of human cocktail-party attention (Bregman 1990): the auditory system first pre-attentively segregates the scene, then selective attention focuses on the target.

---

## How It Works

### 1. Separation — DPCRNSeparator
The mixture is analysed via **STFT** (N_FFT=512 / 32 ms, hop=128 / 8 ms, Hann window, 75 % overlap). The **DPCRNSeparator** — a Dual-Path Convolutional Recurrent Network with 8 alternating Conv2d (frequency axis) + GRU (time axis) blocks — takes the normalised log-magnitude spectrogram and predicts **two soft masks** simultaneously. The masks are applied to the complex STFT of the mixture; ISTFT reconstructs two waveforms, one per source.

Training uses **utterance-level Permutation Invariant Training (uPIT)**, with neg-SI-SDR loss on the reconstructed waveforms. Because neither output is assigned to a gender during training, the separator is structurally symmetric: it handles female and male targets equally well (SI-SDRi +6.73 dB female, +6.90 dB male).

### 2. Attentional Stream Selection
Given the two separated streams, the **AttentionModule** scores each and picks the target:

- **`method="classifier"` (default):** a sliding-window MLP (11 frames × 56 features = 616 inputs, trained on IBM mixture labels) blended with a **GenderGMM** log-likelihood ratio (`log P(X|GMM_F) − log P(X|GMM_M)`, trained on clean speech) yields a per-stream P(female) score. The stream whose score best matches the requested gender is selected.
- **`method="pitch"` (language-agnostic):** mean F0 over voiced frames (70–400 Hz via pYIN) — no trained model required, works across languages.

### 3. Optional VAD Gate
`--vad-gate` applies a smooth amplitude gate to the selected stream, muting inter-word pauses where residual bleedthrough from the other speaker may emerge. Hold time (200 ms) protects word endings from being clipped.

---

## Evaluation Results

Evaluated on 36 synthetic mixtures (LibriSpeech dev-clean, SNR ∈ {−3, 0, +3} dB, seed=123):

| Target | SI-SDRi | PESQ | STOI | Stream selection accuracy |
|---|---|---|---|---|
| Female | **+6.73 dB** | 1.475 | 0.830 | 91.7 % |
| Male   | **+6.90 dB** | 1.505 | 0.842 | 100 %  |

Mix baseline: PESQ ≈ 1.16, STOI ≈ 0.72.

---

## Requirements

- **Python 3.10+**
- **OS:** Linux, macOS, or Windows
- **GPU:** optional — required for training; inference runs on CPU / Apple MPS

### Main Python dependencies

| Library | Purpose |
|---|---|
| `numpy`, `scipy` | Numerical operations, DSP |
| `librosa`, `soundfile` | Audio I/O and feature extraction |
| `scikit-learn` | MLP, GMM, StandardScaler |
| `torch>=2.0` | DPCRNSeparator training and inference |
| `matplotlib` | Spectrograms and diagnostics |
| `pytest` | Unit testing |

---

## Installation

```bash
git clone <repo-url>
cd auralis
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
# venv\Scripts\activate      # Windows
pip install -e ".[dev,torch]"
```

> **Dataset:** download LibriSpeech `dev-clean` from [openslr.org/12](https://www.openslr.org/12/) and place it under `data/raw/librispeech/dev-clean/`.

---

## Usage

### Quick demo

```bash
python demo.py
```

Generates a M+F mixture from LibriSpeech, runs the pipeline, and saves five files to `data/processed/demo/`.

### Train the models

```bash
# Stage 1a — MLP classifier (IBM multi-SNR labels, ~10-20 min):
python -m src.ai.train --out models/classifier.joblib

# Stage 1b — GenderGMM on clean LibriSpeech (~1 min):
python -m src.ai.train_gmm --out models/gender_gmm.joblib

# Stage 2 — DPCRNSeparator, uPIT (standalone, GPU recommended, ~2-4 h):
python -m src.ai.train_separator \
    --n-samples 200 --epochs 60 --batch-size 4 \
    --out models/separator.pt
```

#### Robustness training (reverberant audio)

```bash
python -m src.ai.train_separator \
    --n-samples 200 --epochs 60 --batch-size 4 \
    --rir-dir data/raw/rir --rir-prob 0.5 \
    --out models/separator_robust.pt
```

> **RIR datasets:** [MIT Acoustical Reverberation](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html),
> [OpenSLR28](https://www.openslr.org/28/), [BUT ReverbDB](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database).

### End-to-end pipeline

```bash
# Recommended (classifier stream selection):
python -m src.pipeline \
    --input mix.wav \
    --model models/classifier.joblib \
    --gmm   models/gender_gmm.joblib \
    --separator models/separator.pt \
    --target female \
    --output out.wav

# Language-agnostic (pitch-based, no trained model needed):
python -m src.pipeline \
    --input mix.wav \
    --separator models/separator.pt \
    --target male \
    --stream-select pitch \
    --vad-gate \
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
├── ROADMAP.md
├── requirements.txt
├── demo.py                    # Quick demo script
│
├── src/
│   ├── dsp/
│   │   ├── stft.py            # STFT / ISTFT (centralised parameters)
│   │   ├── features.py        # 56-feature extraction per frame (MFCC, pitch, LPC, ...)
│   │   ├── dataset.py         # LibriSpeech loader, M+F mixer, IBM dataset builder
│   │   ├── augment.py         # RIR reverb augmentation (synthetic-to-real robustness)
│   │   └── enhancement.py     # Voice activity gate
│   │
│   ├── ai/
│   │   ├── classifier.py      # SpeakerClassifier (MLP + StandardScaler)
│   │   ├── gmm_classifier.py  # GenderGMM: LLR = log P(X|GMM_F) - log P(X|GMM_M)
│   │   ├── attention.py       # AttentionModule: score_female() + select_stream()
│   │   ├── dpcrn.py           # DPCRNSeparator — primary separator (~301K params, uPIT)
│   │   ├── train.py           # MLP classifier training script
│   │   ├── train_gmm.py       # GenderGMM training script
│   │   └── train_separator.py # DPCRNSeparator training script (uPIT, dynamic mixing)
│   │
│   ├── pipeline.py            # End-to-end CLI
│   └── utils.py               # Audio I/O utilities
│
├── data/
│   ├── raw/librispeech/       # LibriSpeech dev-clean (not tracked by git)
│   ├── raw/rir/               # Room Impulse Responses for reverb augmentation (optional)
│   └── processed/demo/        # Demo output files
│
├── models/                    # Trained models (not tracked by git)
│   │                          # classifier.joblib, gender_gmm.joblib, separator.pt
├── notebooks/
│   ├── 02_evaluation.ipynb    # Quantitative metrics: SI-SDR, PESQ, STOI
│   └── 03_diagnosis.ipynb     # Classifier vs masking stage diagnosis
├── scripts/                   # Evaluation and research scripts
└── tests/
    ├── test_utils.py
    ├── test_features.py
    ├── test_augment.py
    └── test_pipeline.py
```

---

## Theoretical References

- **Cherry, E. C.** (1953). *Some Experiments on the Recognition of Speech, with One and with Two Ears.* JASA, 25(5). — Original definition of the cocktail party problem.
- **Bregman, A. S.** (1990). *Auditory Scene Analysis.* MIT Press. — Bottom-up / top-down model of auditory attention.
- **Kolbæk, M. et al.** (2017). *Multitalker Speech Separation with Utterance-Level Permutation Invariant Training.* IEEE/ACM TASLP. — uPIT training used for the DPCRNSeparator.
- **Le, X. et al.** (2022). *DPCRN: Dual-Path Convolution Recurrent Network for Single Channel Speech Enhancement.* ICASSP. — Architectural inspiration for the separator.
- **Wang, D., & Brown, G. J.** (2006). *Computational Auditory Scene Analysis.* Wiley-IEEE Press.

---

## License

To be defined.
