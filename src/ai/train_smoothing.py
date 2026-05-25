"""Training script for GRUSmoother — supervised on IBM frame sequences.

The GRU learns to map noisy per-frame classifier probabilities (from the
MLP+GMM blend) to clean IBM ground-truth labels, effectively learning the
optimal temporal smoothing for this specific acoustic setup.

Training data:
  - Raw attention mask from AttentionModule (no HMM smoothing)
  - IBM labels: per-frame 0=F-dominant, 1=M-dominant from STFT energy

Usage:
    # Minimal:
    python -m src.ai.train_smoothing

    # Full options:
    python -m src.ai.train_smoothing \\
        --classifier models/classifier.joblib \\
        --gmm models/gender_gmm.joblib \\
        --n-samples 200 \\
        --epochs 30 \\
        --out models/smoothing_gru.pt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FIXED_FRAMES: int = 500   # consistent with train_mask_net.py


def build_dataset(
    classifier_path: str,
    gmm_path: str | None,
    n_samples: int,
    snr_db_list: list[float],
    clip_duration: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate (raw_mask, ibm_labels) pairs for GRU training.

    raw_mask  — unsmoothed attention probabilities from AttentionModule
    ibm_labels — per-frame 0=F-dominant, 1=M-dominant ground truth

    Both arrays are padded/trimmed to FIXED_FRAMES.
    """
    from src.ai.attention import AttentionModule
    from src.ai.classifier import SpeakerClassifier
    from src.dsp.dataset import make_samples
    from src.dsp.stft import compute_stft

    classifier = SpeakerClassifier.load(classifier_path)
    gmm = None
    if gmm_path:
        from src.ai.gmm_classifier import GenderGMM
        gmm = GenderGMM.load(gmm_path)
    attention_module = AttentionModule(classifier, gmm=gmm)

    items: list[tuple[np.ndarray, np.ndarray]] = []

    for snr in snr_db_list:
        logger.info("Generating %d samples at SNR=%.1f dB...", n_samples, snr)
        samples = make_samples(
            n_samples=n_samples,
            snr_db=snr,
            clip_duration=clip_duration,
            seed=7 + int(snr * 10),
        )
        for i, sample in enumerate(samples):
            if (i + 1) % 20 == 0:
                logger.info("  %d/%d processed", i + 1, len(samples))

            # Raw mask WITHOUT HMM smoothing — the GRU replaces that step
            raw_mask = attention_module.compute_mask(sample.mixture, sr=sample.sr, smooth=False)

            # IBM labels from STFT energy of clean sources
            energy_f = (np.abs(compute_stft(sample.target)) ** 2).mean(axis=0)
            energy_m = (np.abs(compute_stft(sample.interferer)) ** 2).mean(axis=0)
            n_frames  = min(len(raw_mask), len(energy_f), len(energy_m))
            ibm = (energy_f[:n_frames] <= energy_m[:n_frames]).astype(np.float32)
            raw_mask = raw_mask[:n_frames]

            # Pad/trim to FIXED_FRAMES
            def _pad(a: np.ndarray) -> np.ndarray:
                t = len(a)
                if t >= FIXED_FRAMES:
                    return a[:FIXED_FRAMES]
                return np.pad(a, (0, FIXED_FRAMES - t), constant_values=0.5)

            items.append((_pad(raw_mask), _pad(ibm)))

    logger.info("Dataset ready: %d total samples", len(items))
    return items


def train(
    classifier_path: str,
    gmm_path: str | None,
    n_samples: int,
    snr_db_list: list[float],
    clip_duration: float,
    epochs: int,
    batch_size: int,
    lr: float,
    out: str,
    device: str | None,
) -> None:
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for GRUSmoother training. Install with: pip install torch"
        ) from exc

    from src.ai.smoothing_gru import GRUSmoother

    raw_items = build_dataset(
        classifier_path, gmm_path, n_samples, snr_db_list, clip_duration,
    )

    class _Dataset(Dataset):
        def __init__(self, items: list) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            raw, ibm = self.items[idx]
            x = torch.from_numpy(raw).unsqueeze(-1)    # (T, 1)
            y = torch.from_numpy(ibm).unsqueeze(-1)    # (T, 1) — 0=F, 1=M
            # Invert IBM so target is 1=F-dominant (matches attention mask convention)
            y_f = 1.0 - y
            return x, y_f

    n_val = max(1, len(raw_items) // 10)
    val_items, train_items = raw_items[:n_val], raw_items[n_val:]

    train_loader = DataLoader(_Dataset(train_items), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(_Dataset(val_items),   batch_size=batch_size, shuffle=False)
    logger.info(
        "Train: %d  |  Val: %d  |  Batch: %d",
        len(train_items), len(val_items), batch_size,
    )

    gru = GRUSmoother(device=device)
    logger.info("GRUSmoother params: %d  |  Device: %s", gru._model.n_params, gru._device)

    optimizer = torch.optim.Adam(gru._model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        gru._model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(gru._device), y.to(gru._device)
            optimizer.zero_grad()
            pred = gru._model(x)          # (B, T, 1)
            loss = F.binary_cross_entropy(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        gru._model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(gru._device), y.to(gru._device)
                val_loss += F.binary_cross_entropy(gru._model(x), y).item()
        val_loss /= len(val_loader)

        scheduler.step()
        logger.info(
            "Epoch %3d/%d  train_loss=%.5f  val_loss=%.5f  lr=%.2e",
            epoch, epochs, train_loss, val_loss,
            scheduler.get_last_lr()[0],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            gru.save(out_path)
            logger.info("  ↳ new best — checkpoint saved")

    logger.info("Done. Best val_loss=%.5f  →  %s", best_val_loss, out_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train GRUSmoother on IBM frame sequences from LibriSpeech."
    )
    parser.add_argument(
        "--classifier", default="models/classifier.joblib",
        help="Pre-trained SpeakerClassifier path. Default: models/classifier.joblib.",
    )
    parser.add_argument(
        "--gmm", default=None,
        help="Pre-trained GenderGMM path (.joblib). Optional.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Mix clips per SNR value. Default: 200.",
    )
    parser.add_argument(
        "--snr-db", type=float, nargs="+", default=[-3.0, 0.0, 3.0],
        help="SNR values in dB. Default: -3 0 3.",
    )
    parser.add_argument(
        "--clip-duration", type=float, default=4.0,
        help="Duration in seconds of each audio clip. Default: 4.0.",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Training epochs. Default: 30.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size. Default: 16.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Adam learning rate. Default: 0.001.",
    )
    parser.add_argument(
        "--device", default=None,
        help="PyTorch device (cuda / mps / cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--out", default="models/smoothing_gru.pt",
        help="Output path. Default: models/smoothing_gru.pt.",
    )
    args = parser.parse_args()

    train(
        classifier_path=args.classifier,
        gmm_path=args.gmm,
        n_samples=args.n_samples,
        snr_db_list=args.snr_db,
        clip_duration=args.clip_duration,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out=args.out,
        device=args.device,
    )


if __name__ == "__main__":
    main()
