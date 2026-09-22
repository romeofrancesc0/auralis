"""Tests for the LibriMix loader — metadata mixing and directory layout."""
from __future__ import annotations

import csv

import numpy as np
import pytest
import soundfile as sf

from src.data.librimix import LibriMixDataset, fit_lengths

SR = 16_000


def _write_source(path, freq: float, n: int) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(n) / SR
    audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, SR)
    return audio


@pytest.fixture
def librispeech(tmp_path):
    """A miniature LibriSpeech tree with two sources of different lengths."""
    root = tmp_path / "librispeech"
    a = _write_source(root / "train-clean-100/103/1240/103-1240-0000.flac", 220.0, SR)
    b = _write_source(root / "train-clean-100/119/1230/119-1230-0001.flac", 440.0, SR // 2)
    return root, a, b


def _write_metadata(tmp_path, gains=(1.0, 0.5)) -> str:
    csv_path = tmp_path / "libri2mix_test.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mixture_ID", "source_1_path", "source_1_gain",
                         "source_2_path", "source_2_gain", "length"])
        writer.writerow(["103-1240-0000_119-1230-0001",
                         "train-clean-100/103/1240/103-1240-0000.flac", gains[0],
                         "train-clean-100/119/1230/119-1230-0001.flac", gains[1],
                         SR // 2])
    return str(csv_path)


def test_from_metadata_mixes_sources_with_gains(tmp_path, librispeech):
    root, a, b = librispeech
    dataset = LibriMixDataset.from_metadata(_write_metadata(tmp_path), root, sr=SR, mode="min")

    assert len(dataset) == 1
    item = next(iter(dataset))
    assert item.mixture_id == "103-1240-0000_119-1230-0001"
    assert item.n_sources == 2
    np.testing.assert_allclose(item.sources[0], a[: SR // 2], atol=1e-4)
    np.testing.assert_allclose(item.sources[1], b * 0.5, atol=1e-4)
    np.testing.assert_allclose(item.mixture, item.sources[0] + item.sources[1], atol=1e-6)


def test_min_mode_truncates_to_shortest_source(tmp_path, librispeech):
    root, _, _ = librispeech
    item = next(iter(LibriMixDataset.from_metadata(
        _write_metadata(tmp_path), root, sr=SR, mode="min")))
    assert len(item.mixture) == SR // 2


def test_max_mode_pads_to_longest_source(tmp_path, librispeech):
    root, _, _ = librispeech
    item = next(iter(LibriMixDataset.from_metadata(
        _write_metadata(tmp_path), root, sr=SR, mode="max")))
    assert len(item.mixture) == SR
    assert np.all(item.sources[1][SR // 2:] == 0.0)     # padded region is silent


def test_resamples_to_requested_rate(tmp_path, librispeech):
    root, _, _ = librispeech
    item = next(iter(LibriMixDataset.from_metadata(
        _write_metadata(tmp_path), root, sr=8_000, mode="min")))
    assert item.sr == 8_000
    assert len(item.mixture) == pytest.approx(SR // 2 // 2, abs=2)


def test_limit_truncates_the_dataset(tmp_path, librispeech):
    root, _, _ = librispeech
    dataset = LibriMixDataset.from_metadata(
        _write_metadata(tmp_path), root, sr=SR, limit=0)
    assert len(dataset) == 0


def test_missing_source_file_names_the_path(tmp_path, librispeech):
    root, _, _ = librispeech
    csv_path = tmp_path / "broken.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mixture_ID", "source_1_path", "source_1_gain"])
        writer.writerow(["x", "train-clean-100/does/not/exist.flac", 1.0])

    dataset = LibriMixDataset.from_metadata(csv_path, root, sr=SR)
    with pytest.raises(FileNotFoundError, match="exist.flac"):
        next(iter(dataset))


def test_rejects_csv_without_source_columns(tmp_path):
    csv_path = tmp_path / "wrong.csv"
    csv_path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="source_<i>_path"):
        LibriMixDataset.from_metadata(csv_path, None)


def test_from_wav_dir_reads_generated_layout(tmp_path):
    split = tmp_path / "test"
    a = _write_source(split / "s1" / "mix-0.wav", 220.0, SR // 4)
    b = _write_source(split / "s2" / "mix-0.wav", 440.0, SR // 4)
    _write_source(split / "mix_clean" / "mix-0.wav", 330.0, SR // 4)

    item = next(iter(LibriMixDataset.from_wav_dir(split, sr=SR)))
    assert item.mixture_id == "mix-0"
    assert item.n_sources == 2
    np.testing.assert_allclose(item.sources[0], a, atol=1e-4)
    np.testing.assert_allclose(item.sources[1], b, atol=1e-4)


def test_from_wav_dir_requires_mixture_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Mixture directory"):
        LibriMixDataset.from_wav_dir(tmp_path)


def test_fit_lengths_rejects_unknown_mode():
    with pytest.raises(ValueError, match="min.*max"):
        fit_lengths([np.zeros(4)], mode="pad")
