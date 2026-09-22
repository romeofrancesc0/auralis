# G1 — A Robustness Benchmark for Unknown-N Speaker Counting

**Status:** active (chosen 2026-08-10) · **Supersedes:** the attractor/EDA plan in ROADMAP Phase 4

## Research question

Published unknown-N systems report speaker counting accuracy above 95 % — but only on anechoic,
fully overlapped, single-utterance, 8 kHz simulated mixtures. **Where does counting actually break
when the mixture stops looking like that?**

G1 answers this with a controlled sweep over the factors that separate benchmark audio from real
audio: overlap ratio, reverberation, additive noise, and speaker count. See
[`unknown-n-state-of-the-art.md`](unknown-n-state-of-the-art.md) for why this is the open gap.

## What G1 is and is not

**Is:** a protocol, an executable harness, and a set of curves showing counting accuracy and
separation quality as a function of acoustic condition — released so others can rerun it.

**Is not:** a leaderboard of published systems. SepTDA and SepNetEDCI publish no weights, so they
cannot be placed in the table. This limitation is itself part of the motivation: the field's
unknown-N results are currently unreproducible by anyone outside the originating lab.

## Systems under test

| System | Role | N handling |
|---|---|---|
| `sepformer-libri2mix` | Anechoic-trained backbone | Fixed N=2 |
| `sepformer-wsj03mix` | Anechoic-trained backbone | Fixed N=3 |
| `sepformer-whamr` | **Reverb+noise-trained** backbone (13.7 dB SI-SNRi on WHAMR!) | Fixed N=2 |
| Recursive extractor (ours) | The unknown-N system under study | Estimated N̂ via stop criterion |

The recursive extractor wraps a fixed-N backbone in one-and-rest extraction with a replaceable
stop criterion — the Phase 3 deliverable, built here because G1 needs something to measure.

## The oracle-N control

Every cell of the grid is evaluated **twice**: once with N given to the system, once with N
estimated. Without the oracle-N reference, a declining curve cannot distinguish "separation
degrades under reverberation" from "counting fails under reverberation" — and the second is the
claim. The gap between the two curves *is* the cost of not knowing N.

## Factor grid

| Factor | Levels | Source |
|---|---|---|
| Speakers N | 2, 3 | SparseLibriMix 2- and 3-speaker sets |
| Overlap ratio | 0, 0.2, 0.4, 0.6, 0.8, 1.0 | SparseLibriMix (500 mixtures per ratio, 15 s each) |
| Reverberation RT60 | anechoic, 0.2, 0.4, 0.6, 0.8 s | Simulated RIRs (pyroomacoustics), applied via `src/dsp/augment.py` |
| Noise SNR | clean, 10, 5, 0 dB | WHAM! noise (test partition) |

Full grid: 2 × 6 × 5 × 4 = 240 cells, each evaluated under oracle-N and estimated-N.

**Reverberation is simulated, not recorded.** A benchmark axis has to be metric, not ordinal:
simulation gives an exact target RT60 per cell and requires no 4 GB corpus download from anyone
who reruns the sweep. Each source gets an independent RIR (independent speaker positions, shared
room), and the reverberant image of each source — not its dry signal — is the reference, so the
task stays separation rather than joint separation-and-dereverberation.

## Metrics

- **Speaker counting accuracy**, plus the full confusion matrix over N̂. A scalar accuracy hides
  whether errors are off-by-one or catastrophic, and the direction of the error (splitting one
  speaker vs. merging two) is diagnostic.
- **SI-SDRi**, reported two ways: over the subset of mixtures where N̂ = N (clean quality
  comparison, unaffected by counting), and over all mixtures under the standard unknown-N
  convention (best-matching subset when N̂ > N, silent estimate padding when N̂ < N).

The second convention needs a finite value for a silent estimate scored against a live reference,
which is mathematically −∞. The harness uses an explicit, documented floor rather than an epsilon
buried in a denominator; the floor is a parameter and is reported with every result. Exact reconstruction (+∞, reachable only by an
oracle) is capped at 100 dB for the same reason: aggregates and result files must stay finite.

## Work breakdown

| Step | Deliverable | Depends on |
|---|---|---|
| **G1.1** | `src/eval/metrics.py` — SI-SDR/SI-SDRi, PIT assignment, unknown-N scoring, counting confusion. Unit-tested on synthetic signals. | — |
| **G1.2** | Libri2Mix test split + reproduction of 20.6 ±0.5 dB SI-SNRi. Validates the metrics against a published number. | G1.1 |
| **G1.3** | Condition generator: reproducible manifests for every grid cell. | G1.2 |
| **G1.4** | Recursive unknown-N extractor with pluggable stop criterion. | G1.2 |
| **G1.5** | Sweep runner, results, plots. | G1.3, G1.4 |

G1.2–G1.4 are roadmap Phases 1–3 in disguise; G1 consumes them rather than replacing them.

## Compute plan

Pilot-scale cells (tens of mixtures) run on the MacBook CPU to validate the harness — SpeechBrain
1.1.0 cannot use MPS, see the environment note in the README. The full sweep runs on the RTX 5070
once WSL2 + PyTorch nightly cu128 is validated with `scripts/check_separation_env.py --device cuda`.
Sample size per cell is a parameter of the runner, not a constant, so the pilot and the full run
are the same code path.

## Data footprint

Only test material is needed — no training split of anything.

| Asset | Purpose | Notes |
|---|---|---|
| LibriSpeech `test-clean` | Source speech | ~346 MB; `dev-clean` already on disk is not a substitute |
| WHAM! noise (test partition) | Noise axis | Full WHAM! is far larger; only `tt` is required |
| SparseLibriMix | Overlap axis | Generated from the two above via the upstream scripts |
| Libri2Mix test split | G1.2 reproduction only | Full Libri2Mix is ~430 GB — generate the test split alone |
