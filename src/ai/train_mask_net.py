"""Training script for MaskNet — second-stage, requires pre-trained classifier + GMM.

MaskNet learns to refine the NMF-guided IRM using the full DSP+AI pipeline
outputs as input, supervised against the ideal IRM computed from clean sources:

    IRM_target(f, t) = |F(f,t)|² / (|F(f,t)|² + |M(f,t)|² + ε)

Training data is generated on-the-fly from LibriSpeech, reusing the same
make_samples() utility used for MLP training.

Usage:
    # Minimal (assumes pre-trained models at default paths):
    python -m src.ai.train_mask_net

    # Full options:
    python -m src.ai.train_mask_net \\
        --classifier models/classifier.joblib \\
        --gmm models/gender_gmm.joblib \\
        --n-samples 200 \\
        --snr-db -3.0 0.0 3.0 \\
        --epochs 50 \\
        --batch-size 8 \\
        --out models/mask_net.pt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Fixed number of STFT frames used for batching.
# At SR=16000, HOP=128, clip_duration=4.0s → librosa produces ~501 frames.
# We truncate/pad all samples to this length so the DataLoader can collate
# without a custom collate_fn.
FIXED_FRAMES = 500


def _compute_target_irm(target: np.ndarray, interferer: np.ndarray) -> np.ndarray:
    """Wiener-filter IRM target from clean sources.

    IRM(f,t) = |F(f,t)|² / (|F(f,t)|² + |M(f,t)|² + eps)
    """
    from src.dsp.stft import compute_stft
    pf = np.abs(compute_stft(target)) ** 2
    pm = np.abs(compute_stft(interferer)) ** 2
    return (pf / (pf + pm + 1e-8)).astype(np.float32)


def _pad_or_trim(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """Truncate or zero-pad the last axis of arr to exactly n_frames."""
    t = arr.shape[-1]
    if t >= n_frames:
        return arr[..., :n_frames]
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, n_frames - t)]
    return np.pad(arr, pad_width, mode="constant")


def build_dataset(
    classifier_path: str,
    gmm_path: str | None,
    n_samples: int,
    snr_db_list: list[float],
    clip_duration: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Generate (magnitude, attention_weights, nmf_irm, target_irm) tuples.

    Each tuple represents one training sample with all arrays aligned to
    FIXED_FRAMES time steps.
    """
    from src.ai.attention import AttentionModule
    from src.ai.classifier import SpeakerClassifier
    from src.dsp.dataset import make_samples
    from src.dsp.nmf_separation import compute_nmf_irm
    from src.dsp.stft import compute_stft

    classifier = SpeakerClassifier.load(classifier_path)
    gmm = None
    if gmm_path:
        from src.ai.gmm_classifier import GenderGMM
        gmm = GenderGMM.load(gmm_path)
    attention_module = AttentionModule(classifier, gmm=gmm)

    items: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for snr in snr_db_list:
        logger.info("Generating %d samples at SNR=%.1f dB...", n_samples, snr)
        samples = make_samples(
            n_samples=n_samples,
            snr_db=snr,
            clip_duration=clip_duration,
            seed=42 + int(snr * 10),
        )
        for i, sample in enumerate(samples):
            if (i + 1) % 20 == 0:
                logger.info("  %d/%d processed", i + 1, len(samples))

            mix = sample.mixture
            sr = sample.sr

            magnitude = np.abs(compute_stft(mix))           # (F, T_stft)
            attn = attention_module.compute_mask(mix, sr=sr, smooth=True)
            nmf_irm, _ = compute_nmf_irm(mix, attn, sr=sr)  # (F, T_nmf)
            target_irm = _compute_target_irm(sample.target, sample.interferer)

            # Align all arrays to FIXED_FRAMES
            mag_f  = _pad_or_trim(magnitude,  FIXED_FRAMES)  # (F, FIXED)
            irm_f  = _pad_or_trim(nmf_irm,    FIXED_FRAMES)
            tgt_f  = _pad_or_trim(target_irm, FIXED_FRAMES)

            n_attn = min(len(attn), FIXED_FRAMES)
            attn_f = np.zeros(FIXED_FRAMES, dtype=np.float32)
            attn_f[:n_attn] = attn[:n_attn].astype(np.float32)

            items.append((
                mag_f.astype(np.float32),
                attn_f,
                irm_f.astype(np.float32),
                tgt_f.astype(np.float32),
            ))

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
            "PyTorch is required for MaskNet training. Install with: pip install torch"
        ) from exc

    from src.ai.mask_net import MaskNet, build_input

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    raw_items = build_dataset(
        classifier_path, gmm_path, n_samples, snr_db_list, clip_duration,
    )

    class _MaskDataset(Dataset):
        def __init__(self, items: list) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            mag, attn, irm, target = self.items[idx]
            x = build_input(mag, attn, irm)          # (3, F, FIXED)
            return (
                torch.from_numpy(x),
                torch.from_numpy(target),            # (F, FIXED)
            )

    n_val = max(1, len(raw_items) // 10)
    val_items, train_items = raw_items[:n_val], raw_items[n_val:]

    train_loader = DataLoader(_MaskDataset(train_items), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(_MaskDataset(val_items),   batch_size=batch_size, shuffle=False)
    logger.info(
        "Train: %d samples  |  Val: %d samples  |  Batch size: %d",
        len(train_items), len(val_items), batch_size,
    )

    # ------------------------------------------------------------------ #
    # Model, optimizer, scheduler                                          #
    # ------------------------------------------------------------------ #
    net = MaskNet(device=device)
    logger.info("MaskNet params: %d  |  Device: %s", net._model.n_params, net._device)

    optimizer = torch.optim.Adam(net._model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    for epoch in range(1, epochs + 1):
        net._model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(net._device), y.to(net._device)
            optimizer.zero_grad()
            loss = F.mse_loss(net._model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        net._model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(net._device), y.to(net._device)
                val_loss += F.mse_loss(net._model(x), y).item()
        val_loss /= len(val_loader)

        scheduler.step()
        logger.info(
            "Epoch %3d/%d  train_loss=%.5f  val_loss=%.5f  lr=%.2e",
            epoch, epochs, train_loss, val_loss,
            scheduler.get_last_lr()[0],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            net.save(out_path)
            logger.info("  ↳ new best — checkpoint saved")

    logger.info("Done. Best val_loss=%.5f  →  %s", best_val_loss, out_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train MaskNet (second-stage: requires classifier + GMM)."
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
        help="SNR values in dB for data augmentation. Default: -3 0 3.",
    )
    parser.add_argument(
        "--clip-duration", type=float, default=4.0,
        help="Duration in seconds of each audio clip. Default: 4.0.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs. Default: 50.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size. Default: 8.",
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
        "--out", default="models/mask_net.pt",
        help="Output path for the saved model. Default: models/mask_net.pt.",
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
