# Auralis

> **Single-channel speech separation when the number of speakers is unknown.**
> A research project on the cocktail party problem — not only separating voices,
> but deciding how many there are to separate.

---

## The Problem

In a crowded room the human brain follows one voice among many. Colin Cherry
formalised this in 1953 as the **cocktail party problem**, and seventy years later
one half of it is essentially solved: given a mixture of a *known* number of
speakers, modern systems separate it at ~23 dB SI-SDRi.

The other half is open. Real audio does not announce how many people are talking.
A system that must be told "there are three voices here" is not solving the
cocktail party problem — it is solving an easier one.

**Auralis targets the unknown-N case:** given a mixture of N unknown, simultaneous
speakers, produce N separated streams, one per speaker, deciding N itself.

---

## Research direction

The current state of the art is reviewed in
[`docs/research/unknown-n-state-of-the-art.md`](docs/research/unknown-n-state-of-the-art.md).
Two findings shape the plan:

1. **Speaker counting on the standard benchmarks is saturated.** SepEDA (2022),
   SepTDA (2024) and SepNetEDCI (2025) all exceed 95 % counting accuracy on
   WSJ0-{2,3,4,5}Mix; at N=2 they sit at ~99.9 %.
2. **What remains open** is the trade-off between separation quality and
   termination robustness — SepTDA leads by 2–6 dB but its counting collapses to
   83 % at N=5, while SepNetEDCI holds 96 % by giving up that quality — and the
   absence of a shared benchmark for *realistic* conditions. Every published
   number is anechoic, fully overlapped, single-utterance, simulated audio.

The work proceeds in phases, detailed in [`ROADMAP.md`](ROADMAP.md): validate the
environment, **reproduce a published baseline before innovating**, use a
pretrained 3-speaker model as a stepping stone, then build recursive one-and-rest
extraction with a stop criterion. Reproduction targets and the evaluation
protocol are pinned in
[`docs/research/reproduction-targets.md`](docs/research/reproduction-targets.md).

Models are **fine-tuned from pretrained SpeechBrain backbones** rather than
trained from scratch: the hardware ceiling is a single 12 GB GPU, and a method
that does not fit the available compute is not a method.

---

## Status

| | |
|---|---|
| **Current phase** | Reproducing `sepformer-libri2mix` on Libri2Mix test — target 20.6 dB SI-SDRi ± 0.5 |
| **Evaluation harness** | Complete: LibriMix loader, SI-SDRi under optimal assignment, unknown-N scoring, committed per-run records |
| **History** | The previous two-speaker male/female system (custom 301K-parameter DPCRN + gender-based attention, +6.9 dB SI-SDRi on a private synthetic set) is released as tag `v0.4.0` and no longer part of the codebase |

---

## Requirements

- **Python 3.10+**
- **GPU:** required for fine-tuning; inference and evaluation run on CPU
- Pretrained checkpoints are downloaded from Hugging Face on first use

| Library | Purpose |
|---|---|
| `numpy`, `scipy` | Numerical operations, metrics |
| `librosa`, `soundfile` | Audio I/O and resampling |
| `speechbrain>=1.1` | Pretrained separation backbones |
| `torch`, `torchaudio` | Inference and fine-tuning |
| `matplotlib` | Diagnostics and result curves |
| `pytest` | Unit testing |

`pesq` and `pystoi` are optional: the metrics degrade gracefully to SI-SDR alone
when they are not installed.

---

## Installation

```bash
git clone <repo-url>
cd auralis
python3 -m venv .venv
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate      # Windows
pip install -e ".[dev,separation]"
```

Verify the separation stack (interpreter, PyTorch build, GPU, pretrained checkpoint):

```bash
python scripts/check_separation_env.py
```

> **Blackwell GPUs (RTX 50xx, sm_120):** install `torch` from the cu128 nightly
> index rather than the stable wheels, which do not ship sm_120 kernels.
>
> **Apple Silicon:** SpeechBrain 1.1.0 inference runs on CPU, not MPS — its
> `Pretrained` base class only assigns a device type for `cpu` and `cuda`.

### Run tests

```bash
pytest tests/ -q
```

---

## Benchmark & Experiments

Research claims need numbers that are comparable — with published baselines and
with the project's own past runs. Every evaluation goes through one protocol,
pinned in [`docs/research/reproduction-targets.md`](docs/research/reproduction-targets.md):
a frozen split, `min` mixtures at 8 kHz, optimal estimate-to-reference assignment
(never a fixed output order), SI-SDR **improvement** over the unprocessed mixture,
and a committed record of the run.

### Getting the benchmark data

Mixtures can be built on the fly from the official LibriMix metadata plus a local
LibriSpeech tree — nothing extra is stored, and the CSV pins sources and gains so
a run is exactly reproducible:

```bash
git clone https://github.com/JorisCos/LibriMix    # metadata CSVs
# LibriSpeech test-clean from https://www.openslr.org/12/ under data/raw/librispeech/
```

A tree already generated by LibriMix's `generate_librimix.sh` works too, via
`--wav-dir` instead of `--metadata`.

### Running an evaluation

```bash
# Control run — no separation; SI-SDRi must come out at ~0 dB.
# If it does not, the metric plumbing is wrong, not the model.
python scripts/evaluate.py \
    --metadata LibriMix/metadata/Libri2Mix/libri2mix_test.csv \
    --librispeech-root data/raw/librispeech \
    --separator none --sr 8000 --experiment-id control-mixture

# Phase 1 reproduction: pretrained SepFormer on Libri2Mix test
# Target: 20.6 dB SI-SDRi ± 0.5
python scripts/evaluate.py \
    --wav-dir data/raw/Libri2Mix/wav8k/min/test --sr 8000 \
    --separator speechbrain:speechbrain/sepformer-libri2mix \
    --experiment-id sepformer-libri2mix-repro
```

Each run writes `results/<experiment-id>.json` (summary, full config, per-mixture
metrics) and appends a row to [`EXPERIMENTS.md`](EXPERIMENTS.md). Both are
committed: results are research output, not scratch files.

**Unknown-N scoring.** When a system estimates the speaker count itself, metrics
follow the convention used by SepEDA / SepTDA / SepNetEDCI, so the numbers stay
comparable with their tables: over-estimation is scored on the best-matching
subset of estimates, under-estimation substitutes silence for each unmatched
reference, and speaker-count accuracy is reported separately — with a `N->N̂`
confusion breakdown, because a single accuracy figure hides whether errors are
off-by-one or catastrophic.

> **Method.** One variable per experiment, always on the frozen split. PESQ and
> STOI are optional at runtime — SI-SDR and the harness itself are pure numpy,
> so the benchmark runs anywhere, with or without a GPU.

---

---

## Project Structure

```
auralis/
├── README.md
├── ROADMAP.md                 # Phase plan and decision record
├── EXPERIMENTS.md             # Running log of every evaluated run
│
├── docs/research/
│   ├── unknown-n-state-of-the-art.md   # Literature review, candidate directions
│   └── reproduction-targets.md         # Pinned targets and evaluation protocol
│
├── src/
│   ├── data/
│   │   └── librimix.py        # LibriMix loader (metadata CSVs or generated tree)
│   │
│   ├── eval/
│   │   ├── metrics.py         # SI-SDR, optimal assignment, PESQ / STOI
│   │   ├── harness.py         # The evaluation protocol
│   │   └── registry.py        # results/<id>.json + EXPERIMENTS.md row
│   │
│   ├── dsp/
│   │   ├── stft.py            # STFT / ISTFT with centralised parameters
│   │   ├── augment.py         # RIR reverb augmentation (for the robustness sweep)
│   │   └── enhancement.py     # Voice activity gate (stop-criterion candidate)
│   │
│   └── utils.py               # Audio I/O utilities
│
├── scripts/
│   ├── evaluate.py            # Benchmark CLI
│   └── check_separation_env.py # Environment validation
│
├── results/                   # Committed evaluation records, one JSON per run
├── data/raw/                  # LibriSpeech, LibriMix, RIRs (not tracked)
└── tests/
```

---

## Theoretical References

- **Cherry, E. C.** (1953). *Some Experiments on the Recognition of Speech, with One and with Two Ears.* JASA, 25(5). — Original definition of the cocktail party problem.
- **Bregman, A. S.** (1990). *Auditory Scene Analysis.* MIT Press. — Bottom-up / top-down model of auditory attention.
- **Wang, D., & Brown, G. J.** (2006). *Computational Auditory Scene Analysis.* Wiley-IEEE Press.
- **Kolbæk, M. et al.** (2017). *Multitalker Speech Separation with Utterance-Level Permutation Invariant Training.* IEEE/ACM TASLP. — uPIT, the training criterion underlying the known-N systems.
- **Takahashi, N. et al.** (2019). *Recursive Speech Separation for Unknown Number of Speakers.* Interspeech. — OR-PIT; one-and-rest extraction, the baseline for the recursive approach.
- **Chetupalli, S. R., & Habets, E.** (2022). *Speech Separation for an Unknown Number of Speakers Using Transformers With Encoder-Decoder Attractors.* Interspeech. — SepEDA.
- **Lee, D. et al.** (2024). *Boosting Unknown-number Speaker Separation with Transformer Decoder-based Attractor.* — SepTDA, current quality leader.
- **Yang, et al.** (2025). *Speaker Separation for an Unknown Number of Speakers with Encoder-Decoder-Based Contextual Information Module.* Interspeech. — SepNetEDCI; source of the comparison table in the review.

A fuller bibliography, with the numbers each paper reports, is in
[`docs/research/unknown-n-state-of-the-art.md`](docs/research/unknown-n-state-of-the-art.md).

---

## License

[MIT](LICENSE) — free to use, modify and cite, with attribution.

If you use Auralis in academic work, please cite the repository.
