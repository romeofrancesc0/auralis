"""Training script for MaskNet / DPCRN — second-stage, requires pre-trained classifier + GMM.

Both models learn to refine the NMF-guided IRM using the full DSP+AI pipeline
outputs as input, supervised against the ideal IRM computed from clean sources:

    IRM_target(f, t) = |F(f,t)|² / (|F(f,t)|² + |M(f,t)|² + ε)

Dynamic mixing (default):
  Each sample is generated fresh at every __getitem__ call: a random F/M speaker
  pair is selected, a random segment is taken from each clip, and a random SNR is
  drawn from the SNR list.  This means the model sees a unique mixture at every
  step of every epoch, dramatically increasing effective training set diversity
  without downloading additional data.

Loss options:
  mse      — Mean-squared error on the IRM (original, per-bin)
  sisdr    — Negative SI-SDR on reconstructed waveforms (primary metric alignment)
  combined — 0.7 * neg_SI-SDR + 0.3 * MSE (default — metric-aligned + stable)

Usage:
    # Minimal (MaskNet, combined loss, dynamic mixing):
    python -m src.ai.train_mask_net

    # DPCRN with combined loss:
    python -m src.ai.train_mask_net --model-type dpcrn --out models/dpcrn.pt

    # Full options:
    python -m src.ai.train_mask_net \\
        --classifier models/classifier.joblib \\
        --gmm models/gender_gmm.joblib \\
        --n-samples 200 \\
        --snr-db -3.0 0.0 3.0 \\
        --epochs 50 \\
        --batch-size 8 \\
        --loss combined \\
        --out models/mask_net.pt

    # Disable dynamic mixing (legacy static dataset):
    python -m src.ai.train_mask_net --no-dynamic-mixing
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np

from src.dsp.stft import HOP_LENGTH, N_FFT
from src.utils import SAMPLE_RATE

logger = logging.getLogger(__name__)

# Fixed number of STFT frames used for batching.
# At SR=16000, HOP=128, clip_duration=4.0s → librosa produces ~501 frames.
# We truncate/pad all samples to this length so the DataLoader can collate
# without a custom collate_fn.
FIXED_FRAMES: int = 500
FIXED_SAMPLES: int = FIXED_FRAMES * HOP_LENGTH   # 64000 — approx. waveform length


def _si_sdr_loss(pred: "torch.Tensor", target: "torch.Tensor", eps: float = 1e-8) -> "torch.Tensor":
    """Negative SI-SDR (to minimise). Computed per batch element, then averaged.

    SI-SDR = 10 log10(||s_target||² / ||e_noise||²)
    where s_target = (⟨target, pred⟩ / ||target||²) · target
          e_noise  = pred − s_target

    Args:
        pred:   (B, n_samples) predicted waveforms
        target: (B, n_samples) clean target waveforms

    Returns:
        scalar — mean negative SI-SDR over the batch
    """
    import torch
    target = target - target.mean(dim=-1, keepdim=True)
    pred   = pred   - pred.mean(dim=-1, keepdim=True)

    dot            = (target * pred).sum(dim=-1, keepdim=True)
    target_energy  = (target ** 2).sum(dim=-1, keepdim=True) + eps
    s_target       = (dot / target_energy) * target
    e_noise        = pred - s_target

    si_sdr = 10.0 * torch.log10(
        (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps
    )
    return -si_sdr.mean()


def _reconstruct_waveform(
    mask: "torch.Tensor",
    stft_real: "torch.Tensor",
    stft_imag: "torch.Tensor",
    device: "torch.device | str",
) -> "torch.Tensor":
    """Apply mask to complex STFT and reconstruct waveforms via ISTFT.

    Args:
        mask:      (B, F, T) predicted mask in [0, 1]
        stft_real: (B, F, T) real part of mix STFT
        stft_imag: (B, F, T) imaginary part of mix STFT
        device:    target torch device

    Returns:
        (B, FIXED_SAMPLES) reconstructed waveforms
    """
    import torch
    window = torch.hann_window(N_FFT, device=device)
    mix_stft = torch.complex(stft_real, stft_imag)   # (B, F, T)
    masked_stft = mix_stft * mask                     # element-wise: real × complex
    return torch.istft(
        masked_stft,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        length=FIXED_SAMPLES,
    )   # (B, FIXED_SAMPLES)


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


def _load_random_segment(
    rng: random.Random,
    audio_files: list[Path],
    n_samples: int,
    sr: int,
) -> np.ndarray:
    """Load a random segment from a random file in audio_files.

    Unlike _load_clip (which always reads from the start), this function picks
    a uniformly random start offset so that the same audio file can yield many
    distinct training clips.
    """
    from src.utils import load_audio
    path = rng.choice(audio_files)
    audio, _ = load_audio(path, sr=sr)
    if len(audio) <= n_samples:
        return np.pad(audio, (0, n_samples - len(audio)))
    start = rng.randint(0, len(audio) - n_samples)
    return audio[start : start + n_samples]


class _DynamicMixingDataset:
    """PyTorch Dataset that generates a fresh M/F mixture on every __getitem__ call.

    Each call independently samples:
      - a random female speaker from the index
      - a random male speaker from the index
      - a random SNR value from snr_db_list
      - a random clip segment from each speaker's audio files

    This means every epoch exposes the model to a different set of mixtures,
    multiplying effective training data by the number of possible combinations
    of speaker pairs, clip offsets, and SNR values.

    num_workers=0 is required (default in train()) because the sklearn NMF
    object inside compute_nmf_irm() and the attention module are not safe
    to share across forked worker processes.
    """

    def __init__(
        self,
        female_speakers: list,
        male_speakers: list,
        attention_module,
        snr_db_list: list[float],
        n_per_epoch: int,
        clip_duration: float,
        sr: int,
    ) -> None:
        self.female_speakers = female_speakers
        self.male_speakers   = male_speakers
        self.attention_module = attention_module
        self.snr_db_list = snr_db_list
        self.n_per_epoch = n_per_epoch
        self.n_clip = int(sr * clip_duration)
        self.sr = sr

    def __len__(self) -> int:
        return self.n_per_epoch

    def __getitem__(self, idx: int):
        import torch
        from src.ai.mask_net import build_input
        from src.dsp.nmf_separation import compute_nmf_irm
        from src.dsp.stft import compute_stft
        from src.utils import make_mixture

        # System-entropy seed — different result each call, even for the same idx
        rng = random.Random()

        female = rng.choice(self.female_speakers)
        male   = rng.choice(self.male_speakers)
        snr    = rng.choice(self.snr_db_list)
        # Randomly target female (0) or male (1) so DPCRN learns to refine both
        target_gender = rng.choice([0, 1])

        f_audio = _load_random_segment(rng, female.audio_files, self.n_clip, self.sr)
        m_audio = _load_random_segment(rng, male.audio_files,   self.n_clip, self.sr)

        mix = make_mixture(f_audio, m_audio, snr_db=snr)

        stft      = compute_stft(mix)
        magnitude = np.abs(stft)
        # compute_mask returns P(target): P(female) for gender=0, P(male) for gender=1
        attn = self.attention_module.compute_mask(mix, sr=self.sr, smooth=True,
                                                  target_gender=target_gender)

        # NMF IRM and target IRM depend on who we're isolating this step
        nmf_irm, _ = compute_nmf_irm(mix, attn, sr=self.sr, target_gender=target_gender)
        female_irm  = _compute_target_irm(f_audio, m_audio)
        target_irm  = female_irm if target_gender == 0 else (1.0 - female_irm)

        effective_attn = attn  # already P(target), no further inversion needed

        # Target waveform: the voice we want to isolate this step
        tgt_wav_src = f_audio if target_gender == 0 else m_audio

        mag_f  = _pad_or_trim(magnitude,  FIXED_FRAMES)
        irm_f  = _pad_or_trim(nmf_irm,    FIXED_FRAMES)
        tgt_f  = _pad_or_trim(target_irm, FIXED_FRAMES)
        stft_f = _pad_or_trim(stft,       FIXED_FRAMES)

        n_attn = min(len(effective_attn), FIXED_FRAMES)
        attn_f = np.zeros(FIXED_FRAMES, dtype=np.float32)
        attn_f[:n_attn] = effective_attn[:n_attn].astype(np.float32)

        n_samp = len(tgt_wav_src)
        if n_samp >= FIXED_SAMPLES:
            tgt_wav_f = tgt_wav_src[:FIXED_SAMPLES].astype(np.float32)
        else:
            tgt_wav_f = np.pad(tgt_wav_src, (0, FIXED_SAMPLES - n_samp)).astype(np.float32)

        x = build_input(mag_f.astype(np.float32), attn_f, irm_f.astype(np.float32))

        return (
            torch.from_numpy(x),
            torch.from_numpy(tgt_f.astype(np.float32)),
            torch.from_numpy(stft_f.real.astype(np.float32)),
            torch.from_numpy(stft_f.imag.astype(np.float32)),
            torch.from_numpy(tgt_wav_f),
            torch.tensor(target_gender, dtype=torch.long),
        )


def build_dataset(
    classifier_path: str,
    gmm_path: str | None,
    n_samples: int,
    snr_db_list: list[float],
    clip_duration: float,
) -> list[tuple]:
    """Generate training tuples.

    Each tuple:
        (magnitude, attention_weights, nmf_irm, target_irm,
         mix_stft_real, mix_stft_imag, target_waveform)

    All arrays are aligned to FIXED_FRAMES time steps; waveforms to FIXED_SAMPLES.
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

    items: list[tuple] = []

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
            sr  = sample.sr

            stft      = compute_stft(mix)
            magnitude = np.abs(stft)
            attn      = attention_module.compute_mask(mix, sr=sr, smooth=True)
            nmf_irm, _ = compute_nmf_irm(mix, attn, sr=sr)
            target_irm = _compute_target_irm(sample.target, sample.interferer)

            # Align all spectral arrays to FIXED_FRAMES
            mag_f  = _pad_or_trim(magnitude,  FIXED_FRAMES)
            irm_f  = _pad_or_trim(nmf_irm,    FIXED_FRAMES)
            tgt_f  = _pad_or_trim(target_irm, FIXED_FRAMES)
            stft_f = _pad_or_trim(stft,       FIXED_FRAMES)

            n_attn  = min(len(attn), FIXED_FRAMES)
            attn_f  = np.zeros(FIXED_FRAMES, dtype=np.float32)
            attn_f[:n_attn] = attn[:n_attn].astype(np.float32)

            # Target waveform padded to FIXED_SAMPLES
            tgt_wav = sample.target
            n_samp  = len(tgt_wav)
            if n_samp >= FIXED_SAMPLES:
                tgt_wav_f = tgt_wav[:FIXED_SAMPLES].astype(np.float32)
            else:
                tgt_wav_f = np.pad(tgt_wav, (0, FIXED_SAMPLES - n_samp)).astype(np.float32)

            items.append((
                mag_f.astype(np.float32),
                attn_f,
                irm_f.astype(np.float32),
                tgt_f.astype(np.float32),
                stft_f.real.astype(np.float32),
                stft_f.imag.astype(np.float32),
                tgt_wav_f,
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
    loss_type: str,
    model_type: str,
    dynamic_mixing: bool = True,
) -> None:
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for training. Install with: pip install torch"
        ) from exc

    from src.ai.mask_net import MaskNet, build_input

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    if dynamic_mixing:
        from src.ai.attention import AttentionModule
        from src.ai.classifier import SpeakerClassifier
        from src.dsp.dataset import load_speaker_index

        logger.info("Dynamic mixing enabled — fresh mixes generated each step.")

        classifier = SpeakerClassifier.load(classifier_path)
        gmm = None
        if gmm_path:
            from src.ai.gmm_classifier import GenderGMM
            gmm = GenderGMM.load(gmm_path)
        attention_module = AttentionModule(classifier, gmm=gmm)

        index = load_speaker_index()
        female_speakers = index["F"]
        male_speakers   = index["M"]

        n_per_epoch = n_samples * len(snr_db_list)

        train_dataset = _DynamicMixingDataset(
            female_speakers=female_speakers,
            male_speakers=male_speakers,
            attention_module=attention_module,
            snr_db_list=snr_db_list,
            n_per_epoch=n_per_epoch,
            clip_duration=clip_duration,
            sr=SAMPLE_RATE,
        )

        # Fixed small validation set for a reproducible loss curve
        n_val = max(4, n_samples // 5)
        logger.info("Building fixed val set (%d samples)...", n_val)
        val_raw = build_dataset(classifier_path, gmm_path, n_val, snr_db_list[:1], clip_duration)

        class _StaticDataset(Dataset):
            def __init__(self, items: list) -> None:
                self.items = items

            def __len__(self) -> int:
                return len(self.items)

            def __getitem__(self, idx: int):
                mag, attn, irm, target_irm, stft_r, stft_i, tgt_wav = self.items[idx]
                x = build_input(mag, attn, irm)
                return (
                    torch.from_numpy(x),
                    torch.from_numpy(target_irm),
                    torch.from_numpy(stft_r),
                    torch.from_numpy(stft_i),
                    torch.from_numpy(tgt_wav),
                    torch.tensor(0, dtype=torch.long),   # val set always female
                )

        # num_workers=0: NMF and sklearn models are not safe across forked workers
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        val_loader   = DataLoader(_StaticDataset(val_raw), batch_size=batch_size, shuffle=False)
        logger.info(
            "Train: %d samples/epoch (dynamic)  |  Val: %d samples (fixed)  |  Batch: %d",
            n_per_epoch, len(val_raw), batch_size,
        )

    else:
        # ---- Legacy static dataset ----------------------------------------
        raw_items = build_dataset(
            classifier_path, gmm_path, n_samples, snr_db_list, clip_duration,
        )

        class _StaticDataset(Dataset):  # type: ignore[no-redef]
            def __init__(self, items: list) -> None:
                self.items = items

            def __len__(self) -> int:
                return len(self.items)

            def __getitem__(self, idx: int):
                mag, attn, irm, target_irm, stft_r, stft_i, tgt_wav = self.items[idx]
                x = build_input(mag, attn, irm)
                return (
                    torch.from_numpy(x),
                    torch.from_numpy(target_irm),
                    torch.from_numpy(stft_r),
                    torch.from_numpy(stft_i),
                    torch.from_numpy(tgt_wav),
                    torch.tensor(0, dtype=torch.long),   # static dataset always female
                )

        n_val = max(1, len(raw_items) // 10)
        val_items, train_items = raw_items[:n_val], raw_items[n_val:]

        train_loader = DataLoader(_StaticDataset(train_items), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(_StaticDataset(val_items),   batch_size=batch_size, shuffle=False)
        logger.info(
            "Train: %d samples  |  Val: %d samples  |  Batch size: %d",
            len(train_items), len(val_items), batch_size,
        )

    # ------------------------------------------------------------------ #
    # Model, optimizer, scheduler                                          #
    # ------------------------------------------------------------------ #
    if model_type == "dpcrn":
        from src.ai.dpcrn import DPCRN
        net_wrapper = DPCRN(device=device)
        model_obj   = net_wrapper._model
        dev         = net_wrapper._device
        logger.info("DPCRN params: %d  |  Device: %s", model_obj.n_params, dev)
    else:
        net_wrapper = MaskNet(device=device)
        model_obj   = net_wrapper._model
        dev         = net_wrapper._device
        logger.info("MaskNet params: %d  |  Device: %s", model_obj.n_params, dev)

    optimizer = torch.optim.Adam(model_obj.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_sisdr = loss_type in ("sisdr", "combined")
    sisdr_weight = 0.7 if loss_type == "combined" else 1.0
    mse_weight   = 0.3 if loss_type == "combined" else 0.0

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    for epoch in range(1, epochs + 1):
        model_obj.train()
        train_loss = 0.0

        for x, y_irm, stft_r, stft_i, tgt_wav, gender_b in train_loader:
            x, y_irm = x.to(dev), y_irm.to(dev)
            stft_r, stft_i, tgt_wav = stft_r.to(dev), stft_i.to(dev), tgt_wav.to(dev)
            gender_t = gender_b.to(dev)

            optimizer.zero_grad()

            if model_type == "dpcrn":
                pred_mask = model_obj(x)
            else:
                pred_mask = model_obj(x, gender_t)

            loss = torch.tensor(0.0, device=dev)
            if mse_weight > 0:
                loss = loss + mse_weight * F.mse_loss(pred_mask, y_irm)
            if use_sisdr:
                pred_wav = _reconstruct_waveform(pred_mask, stft_r, stft_i, dev)
                loss = loss + sisdr_weight * _si_sdr_loss(pred_wav, tgt_wav)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model_obj.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_irm, stft_r, stft_i, tgt_wav, gender_b in val_loader:
                x, y_irm = x.to(dev), y_irm.to(dev)
                stft_r, stft_i, tgt_wav = stft_r.to(dev), stft_i.to(dev), tgt_wav.to(dev)
                gender_t = gender_b.to(dev)

                if model_type == "dpcrn":
                    pred_mask = model_obj(x)
                else:
                    pred_mask = model_obj(x, gender_t)

                v = torch.tensor(0.0, device=dev)
                if mse_weight > 0:
                    v = v + mse_weight * F.mse_loss(pred_mask, y_irm)
                if use_sisdr:
                    pred_wav = _reconstruct_waveform(pred_mask, stft_r, stft_i, dev)
                    v = v + sisdr_weight * _si_sdr_loss(pred_wav, tgt_wav)
                val_loss += v.item()

        val_loss /= len(val_loader)

        scheduler.step()
        logger.info(
            "Epoch %3d/%d  train_loss=%.5f  val_loss=%.5f  lr=%.2e",
            epoch, epochs, train_loss, val_loss,
            scheduler.get_last_lr()[0],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            net_wrapper.save(out_path)
            logger.info("  ↳ new best — checkpoint saved")

    logger.info("Done. Best val_loss=%.5f  →  %s", best_val_loss, out_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train MaskNet or DPCRN (second-stage: requires classifier + GMM)."
    )
    parser.add_argument(
        "--model-type", choices=["masknet", "dpcrn"], default="masknet",
        help="Model architecture to train. Default: masknet.",
    )
    parser.add_argument(
        "--loss", choices=["mse", "sisdr", "combined"], default="combined",
        help=(
            "Training loss. 'mse': per-bin MSE on IRM. "
            "'sisdr': negative SI-SDR on reconstructed waveforms. "
            "'combined': 0.7*neg_SI-SDR + 0.3*MSE (default)."
        ),
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
    parser.add_argument(
        "--no-dynamic-mixing", action="store_true", default=False,
        help=(
            "Disable dynamic mixing and use a fixed pre-generated dataset (legacy behaviour). "
            "Dynamic mixing is the default and produces better generalisation."
        ),
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
        loss_type=args.loss,
        model_type=args.model_type,
        dynamic_mixing=not args.no_dynamic_mixing,
    )


if __name__ == "__main__":
    main()
