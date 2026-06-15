"""Training script for DPCRNSeparator — dual-output separation with uPIT.

The separator is trained as a PRIMARY separation stage: input is only the
normalised log-magnitude spectrogram of the mixture, output is one T-F mask
per source. No classifier, GMM, or NMF is needed in the training loop, so
data generation is just load + mix (fast).

Loss: utterance-level permutation invariant training (uPIT) with negative
SI-SDR on the reconstructed waveforms (Kolbæk et al. 2017). For each sample
the permutation of (output, source) pairs with the best SI-SDR is used, so
the network is free to assign either output to either speaker — the
attention module performs the target selection at inference time
(cocktail-party model: bottom-up segregation + top-down selection).

Speaker split: 80% of the LibriSpeech speakers (per gender) are used for
dynamic-mixing training, the remaining 20% build a fixed, seeded validation
set — validation therefore measures generalisation to unseen speakers.

Usage:
    python -m src.ai.train_separator --n-samples 200 --epochs 60 \\
        --batch-size 4 --out models/separator.pt
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np

from src.utils import SAMPLE_RATE, make_mixture_with_sources

logger = logging.getLogger(__name__)

CLIP_SECONDS_DEFAULT = 4.0
TRAIN_SPEAKER_FRACTION = 0.8   # remaining 20% are validation-only speakers
VAL_SEED = 1234                # fixed seed → reproducible validation mixtures


def _load_random_segment(
    rng: random.Random,
    audio_files: list[Path],
    n_samples: int,
    sr: int,
) -> np.ndarray:
    """Load a random fixed-length segment from a random file."""
    from src.utils import load_audio
    path = rng.choice(audio_files)
    audio, _ = load_audio(path, sr=sr)
    if len(audio) <= n_samples:
        return np.pad(audio, (0, n_samples - len(audio)))
    start = rng.randint(0, len(audio) - n_samples)
    return audio[start : start + n_samples]


def _make_mix_pair(
    rng: random.Random,
    female_speakers: list,
    male_speakers: list,
    snr_db_list: list[float],
    n_clip: int,
    sr: int,
    rirs: list | None = None,
    rir_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one (mix, source_f, source_m) triple.

    When ``rirs`` is provided, the whole mixture is reverberated with
    probability ``rir_prob`` (both sources convolved with independent RIRs) so
    that the model sees a mix of clean and reverberant conditions and stays
    robust on anechoic input. Reverb is applied to the dry voices *before*
    mixing, so ``make_mixture_with_sources`` still returns reverberant targets
    satisfying ``mix == src_f + src_m``.
    """
    female = rng.choice(female_speakers)
    male   = rng.choice(male_speakers)
    snr    = rng.choice(snr_db_list)
    f_audio = _load_random_segment(rng, female.audio_files, n_clip, sr)
    m_audio = _load_random_segment(rng, male.audio_files, n_clip, sr)
    if rirs and rng.random() < rir_prob:
        from src.dsp.augment import reverberate_pair
        f_audio, m_audio = reverberate_pair(f_audio, m_audio, rirs, rng, sr)
    return make_mixture_with_sources(f_audio, m_audio, snr_db=snr)


def _worker_init(_worker_id: int) -> None:
    """Silence per-file load logging inside DataLoader worker processes.

    On Windows workers are spawned (fresh interpreter), so logger levels set
    in the main process do not propagate.
    """
    logging.getLogger("src.utils").setLevel(logging.WARNING)


class _DynamicMixDataset:
    """Generates a fresh M/F mixture waveform triple on every __getitem__."""

    def __init__(
        self,
        female_speakers: list,
        male_speakers: list,
        snr_db_list: list[float],
        n_per_epoch: int,
        n_clip: int,
        sr: int,
        rirs: list | None = None,
        rir_prob: float = 0.0,
    ) -> None:
        self.female_speakers = female_speakers
        self.male_speakers = male_speakers
        self.snr_db_list = snr_db_list
        self.n_per_epoch = n_per_epoch
        self.n_clip = n_clip
        self.sr = sr
        self.rirs = rirs
        self.rir_prob = rir_prob

    def __len__(self) -> int:
        return self.n_per_epoch

    def __getitem__(self, idx: int):
        import torch
        rng = random.Random()   # system entropy — unique mix per call
        mix, src_f, src_m = _make_mix_pair(
            rng, self.female_speakers, self.male_speakers,
            self.snr_db_list, self.n_clip, self.sr,
            rirs=self.rirs, rir_prob=self.rir_prob,
        )
        return (
            torch.from_numpy(mix.astype(np.float32)),
            torch.from_numpy(src_f.astype(np.float32)),
            torch.from_numpy(src_m.astype(np.float32)),
        )


def _neg_si_sdr(pred: "torch.Tensor", target: "torch.Tensor", eps: float = 1e-8) -> "torch.Tensor":
    """Negative SI-SDR per batch element. pred/target: (B, n) → (B,)."""
    import torch
    target = target - target.mean(dim=-1, keepdim=True)
    pred   = pred   - pred.mean(dim=-1, keepdim=True)
    dot           = (target * pred).sum(dim=-1, keepdim=True)
    target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    s_target      = (dot / target_energy) * target
    e_noise       = pred - s_target
    si_sdr = 10.0 * torch.log10(
        (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps
    )
    return -si_sdr


def _upit_loss(
    preds: "torch.Tensor",
    src_f: "torch.Tensor",
    src_m: "torch.Tensor",
) -> "torch.Tensor":
    """uPIT negative SI-SDR for 2 sources.

    Args:
        preds: (B, 2, n) estimated source waveforms
        src_f: (B, n) female source
        src_m: (B, n) male source

    Returns:
        scalar — mean over batch of the best-permutation neg-SI-SDR
    """
    import torch
    perm_a = _neg_si_sdr(preds[:, 0], src_f) + _neg_si_sdr(preds[:, 1], src_m)
    perm_b = _neg_si_sdr(preds[:, 0], src_m) + _neg_si_sdr(preds[:, 1], src_f)
    return (torch.minimum(perm_a, perm_b) / 2.0).mean()


def _forward_batch(model, mix: "torch.Tensor") -> "torch.Tensor":
    """mix (B, n) → estimated sources (B, 2, n) via mask × complex mix STFT."""
    from src.ai.dpcrn import istft_torch, normalize_log_mag, stft_torch
    mix_stft = stft_torch(mix)                    # (B, F, T) complex
    x = normalize_log_mag(mix_stft.abs())         # (B, 1, F, T)
    masks = model(x)                              # (B, 2, F, T)
    masked = mix_stft.unsqueeze(1) * masks        # (B, 2, F, T)
    return istft_torch(masked, length=mix.shape[-1])


def _split_speakers(speakers: list) -> tuple[list, list]:
    """Deterministic train/val speaker split (sorted by id, 80/20)."""
    ordered = sorted(speakers, key=lambda s: s.id)
    n_train = max(1, int(len(ordered) * TRAIN_SPEAKER_FRACTION))
    return ordered[:n_train], ordered[n_train:]


def train(
    n_samples: int,
    snr_db_list: list[float],
    clip_duration: float,
    epochs: int,
    batch_size: int,
    lr: float,
    out: str,
    device: str | None,
    num_workers: int,
    rir_dir: str | None = None,
    rir_prob: float = 0.0,
) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for training. Install with: pip install torch"
        ) from exc

    from src.ai.dpcrn import DPCRNSeparator
    from src.dsp.augment import load_rir_index, split_rirs
    from src.dsp.dataset import load_speaker_index

    n_clip = int(SAMPLE_RATE * clip_duration)

    index = load_speaker_index()
    female_train, female_val = _split_speakers(index["F"])
    male_train, male_val = _split_speakers(index["M"])
    logger.info(
        "Speakers — train: %dF/%dM  val (held-out): %dF/%dM",
        len(female_train), len(male_train), len(female_val), len(male_val),
    )

    # RIR augmentation: split rooms train/val (held-out rooms for reverb val).
    rirs = load_rir_index(rir_dir) if rir_dir else []
    rir_train, rir_val = split_rirs(rirs) if rirs else ([], [])
    if rirs and not rir_val:               # too few rooms to hold any out
        rir_val = rir_train
    use_reverb = bool(rir_train) and rir_prob > 0.0
    if use_reverb:
        logger.info(
            "Reverb augmentation ON — rooms: %d train / %d val (held-out)  |  p=%.2f",
            len(rir_train), len(rir_val), rir_prob,
        )

    n_per_epoch = n_samples * len(snr_db_list)
    train_dataset = _DynamicMixDataset(
        female_train, male_train, snr_db_list, n_per_epoch, n_clip, SAMPLE_RATE,
        rirs=rir_train, rir_prob=rir_prob,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init if num_workers > 0 else None,
    )

    # Fixed seeded validation set(s) on held-out speakers.
    n_val = max(8, n_samples // 5)

    def _build_val(rng: random.Random, rir_list: list, rprob: float) -> list:
        items = []
        for _ in range(n_val):
            mix, src_f, src_m = _make_mix_pair(
                rng, female_val, male_val, snr_db_list, n_clip, SAMPLE_RATE,
                rirs=rir_list, rir_prob=rprob,
            )
            items.append((
                torch.from_numpy(mix.astype(np.float32)),
                torch.from_numpy(src_f.astype(np.float32)),
                torch.from_numpy(src_m.astype(np.float32)),
            ))
        return items

    # Clean val always present (anti-regression on anechoic input); reverb val
    # only when augmentation is active, built on unseen rooms.
    clean_loader = DataLoader(
        _build_val(random.Random(VAL_SEED), [], 0.0),
        batch_size=batch_size, shuffle=False,
    )
    reverb_loader = None
    if use_reverb:
        reverb_loader = DataLoader(
            _build_val(random.Random(VAL_SEED + 1), rir_val, 1.0),
            batch_size=batch_size, shuffle=False,
        )

    separator = DPCRNSeparator(device=device)
    model = separator._model
    dev = separator._device
    logger.info("DPCRNSeparator params: %d  |  Device: %s", model.n_params, dev)
    logger.info(
        "Train: %d mixes/epoch (dynamic)  |  Val: %d mixes (fixed, unseen speakers)  |  Batch: %d",
        n_per_epoch, n_val, batch_size,
    )

    def _eval_loss(loader) -> float:
        total = 0.0
        with torch.no_grad():
            for mix, src_f, src_m in loader:
                mix, src_f, src_m = mix.to(dev), src_f.to(dev), src_m.to(dev)
                preds = _forward_batch(model, mix)
                total += _upit_loss(preds, src_f, src_m).item()
        return total / len(loader)

    # Mix-as-estimate baseline: clean-val SI-SDR if the system did nothing
    with torch.no_grad():
        base = []
        for mix, src_f, src_m in clean_loader:
            mix, src_f, src_m = mix.to(dev), src_f.to(dev), src_m.to(dev)
            b = (_neg_si_sdr(mix, src_f) + _neg_si_sdr(mix, src_m)) / 2.0
            base.append(-b.mean().item())
        baseline_sisdr = float(np.mean(base))
    logger.info("Val baseline (mix as estimate): SI-SDR = %.2f dB", baseline_sisdr)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for mix, src_f, src_m in train_loader:
            mix, src_f, src_m = mix.to(dev), src_f.to(dev), src_m.to(dev)
            optimizer.zero_grad()
            preds = _forward_batch(model, mix)
            loss = _upit_loss(preds, src_f, src_m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        clean_val_loss = _eval_loss(clean_loader)
        reverb_val_loss = _eval_loss(reverb_loader) if reverb_loader else None
        # Checkpoint metric balances both domains when reverb val exists, so a
        # gain in reverb cannot be bought by regressing on clean input.
        val_score = (
            (clean_val_loss + reverb_val_loss) / 2.0
            if reverb_val_loss is not None else clean_val_loss
        )

        scheduler.step()
        if reverb_val_loss is not None:
            logger.info(
                "Epoch %3d/%d  train SI-SDR=%.2f  val(clean)=%.2f  val(reverb)=%.2f dB  lr=%.2e",
                epoch, epochs, -train_loss, -clean_val_loss, -reverb_val_loss,
                scheduler.get_last_lr()[0],
            )
        else:
            logger.info(
                "Epoch %3d/%d  train SI-SDR=%.2f dB  val SI-SDR=%.2f dB (SI-SDRi %+.2f)  lr=%.2e",
                epoch, epochs, -train_loss, -clean_val_loss,
                -clean_val_loss - baseline_sisdr, scheduler.get_last_lr()[0],
            )

        if val_score < best_val_loss:
            best_val_loss = val_score
            separator.save(out_path)
            logger.info("  ↳ new best — checkpoint saved")

    logger.info("Done. Best val SI-SDR=%.2f dB  →  %s", -best_val_loss, out_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # Dynamic mixing loads two clips per sample — silence the per-file log spam
    logging.getLogger("src.utils").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Train DPCRNSeparator (dual-output, uPIT SI-SDR)."
    )
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Mixtures per SNR value per epoch (dynamic mixing). Default: 200.",
    )
    parser.add_argument(
        "--snr-db", type=float, nargs="+", default=[-3.0, 0.0, 3.0],
        help="SNR values in dB for data augmentation. Default: -3 0 3.",
    )
    parser.add_argument(
        "--clip-duration", type=float, default=CLIP_SECONDS_DEFAULT,
        help="Duration in seconds of each training clip. Default: 4.0.",
    )
    parser.add_argument(
        "--epochs", type=int, default=60,
        help="Training epochs. Default: 60.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size. Default: 4.",
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
        "--num-workers", type=int, default=0,
        help="DataLoader workers. Default: 0 (safe on Windows).",
    )
    parser.add_argument(
        "--rir-dir", default=None,
        help="Directory of RIR files (.wav/.flac) for reverb augmentation. "
             "If omitted, training uses clean mixtures only.",
    )
    parser.add_argument(
        "--rir-prob", type=float, default=0.5,
        help="Probability a mixture is reverberated when --rir-dir is set. "
             "0.5 keeps the model robust on both clean and reverberant input. "
             "Default: 0.5.",
    )
    parser.add_argument(
        "--out", default="models/separator.pt",
        help="Output path for the saved model. Default: models/separator.pt.",
    )
    args = parser.parse_args()

    train(
        n_samples=args.n_samples,
        snr_db_list=args.snr_db,
        clip_duration=args.clip_duration,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out=args.out,
        device=args.device,
        num_workers=args.num_workers,
        rir_dir=args.rir_dir,
        rir_prob=args.rir_prob,
    )


if __name__ == "__main__":
    main()
