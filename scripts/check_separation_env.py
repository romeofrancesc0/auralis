"""Smoke test for the N-speaker separation environment.

Run this first on any machine that is meant to train or evaluate the separation
track — it is the Phase 0 check of the roadmap. It reports the installed stack,
picks a compute device, and (unless --no-model) pulls a pretrained SepFormer
backbone and runs one forward pass on a synthetic two-source mixture.

A pass means the stack is wired up: correct interpreter, working PyTorch build
for this GPU, SpeechBrain able to fetch and instantiate a checkpoint. It says
nothing about separation quality — that is what the Libri2Mix reproduction in
docs/research/reproduction-targets.md is for.

Usage:
    python scripts/check_separation_env.py
    python scripts/check_separation_env.py --device cuda
    python scripts/check_separation_env.py --no-model     # versions + device only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.utils import make_mixture_with_sources

DEFAULT_MODEL = "speechbrain/sepformer-libri2mix"
DEFAULT_SAVEDIR = Path("pretrained_models")
MODEL_SAMPLE_RATE = 8000
SMOKE_DURATION_S = 3.0
SOURCE_FREQS_HZ = (140.0, 220.0)
MODULATION_HZ = 4.0


def report_versions() -> None:
    """Print the interpreter and library versions that define this environment."""
    import torch

    print(f"python      {sys.version.split()[0]} ({sys.executable})")
    print(f"torch       {torch.__version__}")
    try:
        import speechbrain

        print(f"speechbrain {speechbrain.__version__}")
    except ImportError:
        print("speechbrain NOT INSTALLED — pip install -e '.[separation]'")

    print(f"cuda available  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device     {torch.cuda.get_device_name(0)}")
        major, minor = torch.cuda.get_device_capability(0)
        print(f"compute capab.  sm_{major}{minor}")
    print(f"mps available   {torch.backends.mps.is_available()}")


def select_device(requested: str) -> str:
    """Resolve 'auto' to the best usable device, or validate an explicit choice.

    'auto' never picks MPS: speechbrain 1.1.0 sets `device_type` only for 'cpu'
    and 'cuda' (inference/interfaces.py), so instantiating a Pretrained model on
    'mps' raises AttributeError before any compute happens. Apple Silicon runs
    SpeechBrain inference on CPU until that is fixed upstream.
    """
    import torch

    if requested == "mps":
        print("warning: speechbrain 1.1.0 does not support mps for inference")
        return requested
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def synth_voiced_source(freq_hz: float, duration_s: float, sr: int) -> np.ndarray:
    """Synthesize a crude voiced-speech proxy: a modulated harmonic stack.

    Not speech — just a signal with harmonic structure and an amplitude envelope,
    enough to exercise the model's input path with something that is not noise.
    """
    t = np.arange(int(duration_s * sr), dtype=np.float32) / sr
    harmonics = sum(np.sin(2 * np.pi * freq_hz * k * t) / k for k in range(1, 6))
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * MODULATION_HZ * t))
    return (harmonics * envelope).astype(np.float32)


def run_forward_pass(model_source: str, savedir: Path, device: str) -> None:
    """Load a pretrained separator and separate one synthetic mixture."""
    import torch
    from speechbrain.inference.separation import SepformerSeparation

    print(f"\nloading {model_source} on {device} ...")
    model = SepformerSeparation.from_hparams(
        source=model_source,
        savedir=str(savedir / Path(model_source).name),
        run_opts={"device": device},
    )

    source_a = synth_voiced_source(SOURCE_FREQS_HZ[0], SMOKE_DURATION_S, MODEL_SAMPLE_RATE)
    source_b = synth_voiced_source(SOURCE_FREQS_HZ[1], SMOKE_DURATION_S, MODEL_SAMPLE_RATE)
    mixture, _, _ = make_mixture_with_sources(source_a, source_b, snr_db=0.0)

    batch = torch.from_numpy(mixture).unsqueeze(0).to(device)
    estimates = model.separate_batch(batch)

    print(f"mixture     {tuple(batch.shape)} @ {MODEL_SAMPLE_RATE} Hz")
    print(f"estimates   {tuple(estimates.shape)}  (batch, samples, sources)")
    print(f"n_sources   {estimates.shape[-1]}")
    if not torch.isfinite(estimates).all():
        raise RuntimeError("separator produced non-finite output")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="pretrained checkpoint id")
    parser.add_argument("--savedir", type=Path, default=DEFAULT_SAVEDIR)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="compute device (default: best available)",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="skip the checkpoint download and forward pass",
    )
    args = parser.parse_args()

    report_versions()
    if args.no_model:
        return

    device = select_device(args.device)
    run_forward_pass(args.model, args.savedir, device)
    print("\nOK — separation stack is functional on this machine.")


if __name__ == "__main__":
    main()
