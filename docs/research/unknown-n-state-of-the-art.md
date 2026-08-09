# Unknown-N Speech Separation — State of the Art

**Review date:** 2026-08-09 · **Scope:** single-channel speech separation when the number of
speakers N is not known in advance.

## Why this review exists

The roadmap (locked 2026-07-11) proposed *attractor / EDA-style counting* as the main research
contribution of Auralis, with recursive one-and-rest extraction as the baseline. This review checks
whether that plan still describes an open problem.

**It does not, as stated.** Attractor-based counting was published in 2022 (SepEDA), improved in
2024 (SepTDA) and again in 2025 (SepNetEDCI). Speaker counting on the standard benchmarks is
essentially saturated. The plan needs re-aiming — Phases 0–3 are unaffected, Phase 4 is not.

## 1. Taxonomy

Yang et al. (2025) group unknown-N methods into three families; the grouping is a useful map of
where the remaining difficulty sits.

| Family | Mechanism | Stop / counting signal | Weakness |
|---|---|---|---|
| **Fixed maximum N** | One model with N_max output channels, or one decoder / one model per N | Auxiliary speaker-count head, or activity detection on outputs | Breaks entirely above N_max; wasted capacity below it |
| **Iterative / recursive** | Extract one speaker, feed the residual back (OR-PIT) | Energy threshold or binary discriminator on the residual | Runtime linear in N; **error accumulates** across iterations |
| **Attractor** | Generate one attractor vector per speaker, extract all in one pass | Learned stop head on the attractor sequence | Attractors are compact vectors that discard context; training is harder to bootstrap from pretrained weights |

Auralis' v0.4.0 system is a degenerate case of family 1 with N_max = 2.

## 2. Reported results

All numbers below are on WSJ0-{2,3,4,5}Mix, 8 kHz, `min` version, **anechoic, fully overlapped,
one utterance per speaker**. `*` marks the unknown-N condition; parentheses give speaker counting
accuracy. Source: comparison table in Yang et al., Interspeech 2025.

| Method | 2Mix | 3Mix | 4Mix | 5Mix |
|---|---|---|---|---|
| Recursive SS (OR-PIT lineage) | 14.8 | 12.6 | 10.2 | — |
| Gated DPRNN | 20.1 | 16.9 | 12.9 | 10.6 |
| Gated DPRNN\* | 18.6 | 14.6 | 11.5 | 10.4 |
| SepEDA | 21.1 | 18.6 | 14.7 | 12.1 |
| SepEDA\* | 21.1 (99.80) | 18.4 (97.00) | 14.4 (90.17) | 11.6 (96.87) |
| **SepTDA** | **23.6** | **23.5** | **22.0** | **21.0** |
| SepTDA\* | 23.6 (99.90) | 22.1 (95.93) | 19.5 (90.10) | 16.9 (**83.23**) |
| SepNetEDCI | 21.4 | 20.0 | 17.0 | 14.8 |
| SepNetEDCI\* | 21.4 (99.87) | 19.9 (**99.53**) | 16.9 (**97.71**) | 14.3 (**95.74**) |

SI-SDRi in dB. The SepTDA paper reports 24.0 dB on WSJ0-2mix for its own best configuration; the
23.6 dB above is the number as tabulated by Yang et al.

Two facts matter more than the ranking:

1. **Counting is solved in this regime.** Three independent systems exceed 95 % counting accuracy;
   at N=2 everyone is at ~99.9 %. "Count the speakers" is no longer the hard part *here*.
2. **Quality and counting robustness trade off against each other, and nobody has resolved it.**
   SepTDA wins separation by 2–6 dB but its counting collapses as N grows (90.1 % at N=4,
   83.2 % at N=5). SepNetEDCI gives up that quality margin and holds counting at 95.7 % for N=5.
   This tension — labelled by the 2025 survey as separation quality versus termination robustness —
   is the one genuinely unresolved axis on the standard benchmarks.

## 3. The regime nobody benchmarks

Every number in §2 comes from anechoic, 100 %-overlapped, single-utterance, 8 kHz simulated
mixtures. Real cocktail-party audio is reverberant, noisy, sparsely overlapped, and each speaker
contributes several utterances separated by pauses — the exact conditions under which an
energy-based or VAD-based stop criterion is most likely to misfire.

The work that does target that regime, A-DCSS (Interspeech 2025), reports **9.7 dB ΔSI-SDR with
97.9 % counting accuracy** on reverberant multi-utterance mixtures at ~22.5 % average overlap —
roughly *half* the dB figure of the anechoic benchmarks, on data the authors synthesized
themselves (LibriSpeech + WHAM! noise, RT60 0.2–0.6 s).

That is the structural gap: **realistic-condition unknown-N results are not comparable across
papers, because each paper builds its own realistic set.** Public sparsely-overlapped material
exists (SparseLibriMix: 2- and 3-speaker mixtures with overlap ratios swept from 0 to 100 %, WHAM!
noise), but it is used for extraction and diarization far more than for unknown-N counting.

## 4. Candidate directions for Auralis

Ranked by feasibility under the actual hardware ceiling (RTX 5070, 12 GB; cloud GPU only
occasionally).

### G1 — A reproducible robustness benchmark for unknown-N counting *(recommended)*

Measure where speaker counting actually breaks, under a controlled sweep of realistic factors:
overlap ratio (SparseLibriMix, 0–100 %), reverberation (RT60 sweep through the existing
`src/dsp/augment.py` RIR pipeline), additive noise (WHAM!), and N = 2…4. Evaluate published,
pretrained systems plus a recursive baseline, with a fixed public protocol and released code.

- *Why it is a contribution:* it produces the comparison the field currently cannot make. Papers
  report robustness claims on private synthetic sets; nobody reports a shared curve of counting
  accuracy versus overlap and RT60.
- *Why it fits:* dominated by inference, not training. Runs on the 5070, and partly on the M5.
- *Existing assets reused:* `augment.py` (RIR), `enhancement.py::voice_activity_gate`, the
  evaluation habits from v0.4.0.
- *Risk:* it is a measurement contribution, not an architectural one — weaker as a headline, but
  it is the prerequisite for G2 either way.

### G2 — Termination robustness of recursive extraction *(natural follow-up)*

The recursive family's stop criterion is its weakest component and the cheapest to iterate on:
energy threshold vs. VAD vs. learned discriminator, evaluated under reverberation and sparse
overlap, on top of a pretrained SpeechBrain backbone. Needs G1's harness to be measurable at all.

- *Why it fits:* fine-tuning a stop head is cheap; the backbone stays frozen.
- *Risk:* the deflationary-extraction line of work (2025) is moving in this area — check for
  collisions before committing.

### G3 — Closing the quality/counting trade-off

Combine SepTDA-level separation with SepNetEDCI-level counting robustness. Highest novelty,
highest cost: from-scratch training of a SOTA-scale model, i.e. rented cloud GPUs. Not a sensible
first target, and not something to attempt before G1 exists.

**Recommendation:** G1 as the next milestone, G2 as the follow-on, G3 only if compute appears.
This keeps every phase measurable on hardware that is actually available, and it makes the
"attractor" work of Phase 4 optional rather than load-bearing.

## 5. What does not change

Phases 0–3 of the roadmap stand as written: environment validation, reproducing a published
Libri2Mix baseline, the 3-speaker stepping stone, and a recursive N-way extraction baseline. G1
consumes their output directly — a reproduction harness that trusts its own SI-SDRi numbers is
exactly what a robustness sweep needs.

## Sources

- [Advances in Speech Separation: Techniques, Challenges, and Future Trends](https://arxiv.org/html/2508.10830v1) — survey, Aug 2025
- [Speaker Separation for an Unknown Number of Speakers with Encoder-Decoder-Based Contextual Information Module](https://www.isca-archive.org/interspeech_2025/yang25_interspeech.pdf) — SepNetEDCI, Interspeech 2025
- [Boosting Unknown-number Speaker Separation with Transformer Decoder-based Attractor](https://arxiv.org/html/2401.12473v1) — SepTDA, 2024
- [Speech Separation for an Unknown Number of Speakers Using Transformers With Encoder-Decoder Attractors](https://www.isca-archive.org/interspeech_2022/chetupalli22_interspeech.pdf) — SepEDA, Interspeech 2022
- [Recursive speech separation for unknown number of speakers](https://www.isca-archive.org/interspeech_2019/takahashi19_interspeech.pdf) — OR-PIT, Interspeech 2019
- [Coarse-to-Fine Recursive Speech Separation for Unknown Number of Speakers](https://arxiv.org/pdf/2203.16054) — 2022
- [Deflationary Extraction Transformer for Speech Separation with Unknown Number of Talkers](https://doi.org/10.3390/s25164905) — Sensors, 2025
- [Attractor-Based Speech Separation of Multiple Utterances by Unknown Number of Speakers](https://arxiv.org/html/2505.16607) — A-DCSS, Interspeech 2025
- [LibriMix: An Open-Source Dataset for Generalizable Speech Separation](https://arxiv.org/pdf/2005.11262) — SparseLibriMix is introduced here
