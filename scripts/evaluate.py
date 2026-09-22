"""Benchmark a separator on LibriMix and record the run.

The protocol is fixed on purpose: a frozen split, optimal estimate-to-reference
assignment, SI-SDRi against the unprocessed mixture, and a JSON record per run.
Numbers produced this way are comparable with published baselines and with
every past experiment in EXPERIMENTS.md.

Examples:
    # Baseline of the current 2-source separator on the Libri2Mix test split
    python scripts/evaluate.py \\
        --metadata metadata/Libri2Mix/libri2mix_test.csv \\
        --librispeech-root data/raw/librispeech \\
        --separator models/separator.pt \\
        --experiment-id dpcrn-upit-v0.4.0-libri2mix-test

    # Control run: no separation at all — SI-SDRi must come out at ~0 dB
    python scripts/evaluate.py --metadata ... --separator none --experiment-id control-mixture

    # Phase 1 reproduction: pretrained SepFormer on Libri2Mix test (8 kHz, min)
    python scripts/evaluate.py \\
        --wav-dir data/raw/Libri2Mix/wav8k/min/test --sr 8000 \\
        --separator speechbrain:speechbrain/sepformer-libri2mix \\
        --experiment-id sepformer-libri2mix-repro
        # target: 20.6 dB SI-SDRi +/- 0.5 (docs/research/reproduction-targets.md)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.librimix import LibriMixDataset          # noqa: E402
from src.eval.harness import evaluate                  # noqa: E402
from src.eval.registry import save_results             # noqa: E402

logger = logging.getLogger("evaluate")

DPCRN_SR      = 16_000   # the legacy separator's STFT parameters are tied to this rate
SPEECHBRAIN_SR = 8_000   # published sample rate of the sepformer/resepformer checkpoints


def build_separator(path: str, n_sources: int, sr: int = 16_000):
    """Return a `separate(mixture, sr) -> list[np.ndarray]` callable.

    Three forms are accepted:

    * ``none`` — the control: hands back copies of the mixture, so the harness
      must report SI-SDRi ≈ 0 dB. Any deviation means the metric plumbing is
      wrong, not the model.
    * ``speechbrain:<source>`` — a pretrained SpeechBrain separator, e.g.
      ``speechbrain:speechbrain/sepformer-libri2mix``. This is the Phase 1
      reproduction path; those checkpoints are 8 kHz, so pass ``--sr 8000``.
    * anything else — a path to a legacy v0.4.0 DPCRNSeparator checkpoint.
    """
    if path.lower() == "none":
        logger.info("Control separator: returning %d copies of the mixture", n_sources)
        return lambda mixture, sr: [mixture.copy() for _ in range(n_sources)]

    if path.startswith("speechbrain:"):
        return _speechbrain_separator(path.split(":", 1)[1], sr)

    if sr != DPCRN_SR:
        raise ValueError(
            f"The DPCRN separator operates at {DPCRN_SR} Hz (its STFT parameters are "
            f"fixed); --sr {sr} would silently mis-scale every mixture. Re-run at "
            f"{DPCRN_SR} Hz, or use --separator none for a control run at this rate."
        )

    from src.ai.dpcrn import DPCRNSeparator           # imported lazily: needs torch

    separator = DPCRNSeparator.load(path)
    logger.info("Loaded DPCRNSeparator from %s", path)
    return lambda mixture, sr: list(separator.separate(mixture))


def _speechbrain_separator(source: str, sr: int):
    """Wrap a pretrained SpeechBrain separator as a plain callable."""
    import torch

    try:                                              # SpeechBrain >= 1.0
        from speechbrain.inference.separation import SepformerSeparation
    except ImportError:                               # older layout
        from speechbrain.pretrained import SepformerSeparation

    if sr != SPEECHBRAIN_SR:
        logger.warning(
            "%s was published at %d Hz; running at %d Hz gives numbers that are "
            "not comparable with the model card.", source, SPEECHBRAIN_SR, sr,
        )

    model = SepformerSeparation.from_hparams(
        source=source, savedir=f"models/pretrained/{source.replace('/', '_')}",
    )
    logger.info("Loaded SpeechBrain separator: %s", source)

    def separate(mixture: np.ndarray, sr: int) -> list[np.ndarray]:
        batch = torch.from_numpy(mixture.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            estimates = model.separate_batch(batch)   # (1, time, n_src)
        estimates = estimates.squeeze(0).cpu().numpy()
        return [estimates[:, i] for i in range(estimates.shape[-1])]

    return separate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata", help="LibriMix metadata CSV (mixtures built on the fly).")
    source.add_argument("--wav-dir",  help="Pre-generated LibriMix split directory.")

    parser.add_argument("--librispeech-root", default="data/raw/librispeech",
                        help="Directory holding the LibriSpeech subsets (with --metadata).")
    parser.add_argument("--mix-dir", default="mix_clean",
                        help="Mixture subdirectory (with --wav-dir). Default: mix_clean.")
    parser.add_argument("--separator", required=True,
                        help="'speechbrain:<hf-source>' for a pretrained checkpoint, a path "
                             "to a legacy separator.pt, or 'none' for the control run.")
    parser.add_argument("--experiment-id", required=True,
                        help="Identifier for results/<id>.json and the EXPERIMENTS.md row.")
    parser.add_argument("--notes", default="", help="One-line note stored with the run.")
    parser.add_argument("--sr", type=int, default=16_000, choices=[8_000, 16_000])
    parser.add_argument("--mode", default="min", choices=["min", "max"],
                        help="LibriMix length mode. 'min' is the reporting standard.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N mixtures (smoke runs).")
    parser.add_argument("--n-sources", type=int, default=2,
                        help="Sources the control separator emits. Default: 2.")
    parser.add_argument("--no-perceptual", action="store_true",
                        help="Skip PESQ and STOI (much faster, SI-SDR only).")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--log-path", default="EXPERIMENTS.md",
                        help="Experiment log to append to; 'none' to skip the row.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.metadata:
        dataset = LibriMixDataset.from_metadata(
            args.metadata, args.librispeech_root, sr=args.sr,
            mode=args.mode, limit=args.limit,
        )
    else:
        dataset = LibriMixDataset.from_wav_dir(
            args.wav_dir, mix_dir=args.mix_dir, sr=args.sr, limit=args.limit,
        )

    separate = build_separator(args.separator, args.n_sources, sr=args.sr)

    config = {
        "separator":   args.separator,
        "dataset":     args.metadata or args.wav_dir,
        "sr":          args.sr,
        "mode":        args.mode,
        "limit":       args.limit,
        "perceptual":  not args.no_perceptual,
    }

    result = evaluate(
        separate, dataset,
        dataset_name=dataset.name,
        perceptual=not args.no_perceptual,
        config=config,
    )
    save_results(result, args.experiment_id, notes=args.notes,
                 results_dir=Path(args.results_dir),
                 log_path=None if args.log_path.lower() == "none" else Path(args.log_path))

    summary = result.summary()
    print(f"\n{args.experiment_id} — {summary['dataset']}  ({summary['n_items']} mixtures)")
    print(f"  SI-SDRi        {summary['si_sdri_mean']} dB  (± {summary['si_sdri_std']})")
    print(f"  SI-SDR in/out  {summary['si_sdr_in_mean']} → {summary['si_sdr_out_mean']} dB")
    print(f"  PESQ / STOI    {summary['pesq_mean']} / {summary['stoi_mean']}")
    print(f"  Count accuracy {summary['count_accuracy']}")


if __name__ == "__main__":
    main()
