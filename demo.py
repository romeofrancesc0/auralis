"""Quick demo: genera un mix M+F, esegue la pipeline, salva i file da ascoltare.

Usage:
    python demo.py

Output in data/processed/demo/:
    - mix.wav         → il mix originale (voce F + voce M)
    - target.wav      → la voce femminile isolata (ground truth)
    - interferer.wav  → la voce maschile (ground truth)
    - output.wav      → output del sistema (voce F estratta dal mix)
"""
from __future__ import annotations

import logging

from src.ai.attention import AttentionModule
from src.ai.classifier import SpeakerClassifier
from src.dsp.dataset import make_samples
from src.dsp.enhancement import enhance
from src.dsp.nmf_separation import separate_nmf
from src.utils import save_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH = "models/classifier.joblib"
OUT_DIR = "data/processed/demo"


def main() -> None:
    log.info("Generando un campione di test da LibriSpeech...")
    samples = make_samples(n_samples=1, clip_duration=5.0, seed=7)
    sample = samples[0]

    log.info("Speaker target (F): %s | Speaker interferente (M): %s",
             sample.target_speaker_id, sample.interferer_speaker_id)

    # Salva mix e ground truth
    save_audio(f"{OUT_DIR}/mix.wav", sample.mixture, sr=sample.sr)
    save_audio(f"{OUT_DIR}/target.wav", sample.target, sr=sample.sr)
    save_audio(f"{OUT_DIR}/interferer.wav", sample.interferer, sr=sample.sr)

    # Esegue la pipeline
    log.info("Caricando il classificatore da %s...", MODEL_PATH)
    classifier = SpeakerClassifier.load(MODEL_PATH)
    attention = AttentionModule(classifier)

    log.info("Calcolando la maschera di attenzione...")
    mask = attention.compute_mask(sample.mixture, sr=sample.sr)
    log.info("Genere dominante rilevato nel mix: %s", attention.dominant_gender(sample.mixture, sr=sample.sr))

    log.info("Separando con NMF guidata dal classificatore...")
    reconstructed = separate_nmf(sample.mixture, mask, sr=sample.sr)
    output = enhance(reconstructed, sr=sample.sr)

    save_audio(f"{OUT_DIR}/output.wav", output, sr=sample.sr)

    print("\n" + "="*55)
    print("  FILE SALVATI IN data/processed/demo/")
    print("="*55)
    print("  mix.wav         → mix originale (F + M)")
    print("  target.wav      → voce femminile (ground truth)")
    print("  interferer.wav  → voce maschile (ground truth)")
    print("  output.wav      → voce estratta dal sistema")
    print("="*55)
    print("\nAscolta i file nell'ordine sopra per valutare il risultato.")


if __name__ == "__main__":
    main()
