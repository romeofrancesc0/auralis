# Reproduction Targets and Evaluation Protocol

**Verified:** 2026-08-09 (numbers read from the SpeechBrain model cards on Hugging Face).

Phase 1 of the roadmap says: reproduce a published baseline before innovating. This file fixes
*which* numbers count as a pass, so the check is falsifiable rather than a vibe.

## Pretrained checkpoints and their published numbers

| Checkpoint | Test set | SI-SNRi | SDRi | Sample rate | Outputs |
|---|---|---|---|---|---|
| `speechbrain/sepformer-libri2mix` | Libri2Mix test | **20.6 dB** | 20.9 dB | 8 kHz | 2 |
| `speechbrain/sepformer-wsj02mix` | WSJ0-2Mix test | 22.4 dB | 22.6 dB | 8 kHz | 2 |
| `speechbrain/sepformer-wsj03mix` | WSJ0-3Mix test | 19.8 dB | 20.0 dB | 8 kHz | 3 |
| `speechbrain/resepformer-wsj02mix` | WSJ0-2Mix test | 18.6 dB | 18.9 dB | 8 kHz | 2 |

**Primary Phase 1 target:** `sepformer-libri2mix` on the Libri2Mix test set, 20.6 dB SI-SNRi,
**tolerance ±0.5 dB**. Missing the tolerance means the harness is wrong, not the model — debug the
pipeline before drawing any conclusion from it.

**Phase 2 target:** `sepformer-wsj03mix`, 19.8 dB SI-SNRi on WSJ0-3Mix. Note WSJ0 is LDC-licensed;
if the corpus is not available, Libri3Mix substitutes for the N=3 stepping stone, but then the
published number no longer applies and the run is a smoke test, not a reproduction.

## Protocol

Getting these details wrong is the usual reason a reproduction misses by 1–2 dB.

- **Dataset version:** `min` (mixtures truncated to the shortest source), 8 kHz. The `max` version
  yields different, non-comparable numbers.
- **Improvement, not absolute:** SI-SDRi = SI-SDR(estimate, reference) − SI-SDR(mixture, reference).
  Reporting bare SI-SDR silently inflates results.
- **Permutation:** score under the best permutation of estimates against references (PIT at
  evaluation time), then average over utterances — not over a concatenated signal.
- **No length or gain fixes:** do not peak-normalize or trim estimates before scoring. SI-SDR is
  scale-invariant by construction; other post-processing is not neutral.
- **Full test set:** Libri2Mix test is 3000 mixtures. A subset is acceptable for a smoke test only
  if the sample size and seed are reported alongside the number, and it is never compared to a
  published figure.

### Unknown-N scoring rules

When the system estimates N̂ instead of being told N, separation scores need a convention for
count errors. The standard one (used by SepEDA, SepTDA and SepNetEDCI, which makes their tables
mutually comparable):

- **Over-estimation (N̂ > N):** score the subset of estimates that best matches the references.
- **Under-estimation (N̂ < N):** substitute a silent signal for each unmatched reference.

Report **speaker counting accuracy separately** from SI-SDRi, and give the full confusion matrix
over N̂ when the sweep covers more than two speaker counts — a single accuracy number hides whether
errors are off-by-one or catastrophic.

## Storage and runtime notes

- Full Libri2Mix (all splits, both sample rates) does not fit in the ~118 GB currently free.
  Generate the **test split only** for Phase 1; training splits are only needed once fine-tuning
  starts, on the machine that will actually do the fine-tuning.
- LibriMix generation resamples and mixes from LibriSpeech; the existing `data/raw/LibriSpeech`
  (dev-clean, 349 MB) is not sufficient — Libri2Mix test is built from `test-clean`.
- SepFormer inference on 3000 mixtures is heavy on a CPU/MPS machine. Expect the full Phase 1
  evaluation to belong on the RTX 5070 box; the MacBook is for wiring the harness up and listening
  to outputs.
