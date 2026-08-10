"""Simulated room impulse responses with a controlled reverberation time.

The G1 benchmark needs reverberation as a *metric* axis: every condition has to
correspond to a known RT60, so the resulting curve reads "counting accuracy
versus reverberation time" rather than "versus a bag of rooms". A recorded RIR
corpus cannot give that — its rooms come with nominal, unevenly spaced RT60s —
so the reverberation axis is simulated with the image-source method instead.
It also means anyone rerunning the benchmark needs no multi-gigabyte download.

Room geometry follows the ranges used in the reverberant unknown-N literature
(A-DCSS, Interspeech 2025): 4-8 m floor, 3-4 m ceiling, RT60 0.2-0.6 s, which
keeps our conditions comparable to the closest published work.

Sources share a room and get independent positions, mirroring the geometry of
several people talking in one space. Applying the RIRs is
``src.dsp.augment.apply_rir``'s job; this module only generates them.

Sabine's formula is an approximation, and inverting it does not deliver the
reverberation time you asked for: measured against the requested value it runs
roughly 18 % long at 0.2 s and 58 % long at 0.8 s, because the image-source
model is truncated at a finite order and Sabine assumes a diffuse field. The
generator therefore calibrates — it adjusts the absorption until the *measured*
decay matches the target — and reports the measured value, which is what any
manifest or plot axis should carry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pyroomacoustics as pra

from src.utils import SAMPLE_RATE

logger = logging.getLogger(__name__)

ROOM_LENGTH_RANGE_M = (4.0, 8.0)
ROOM_WIDTH_RANGE_M = (4.0, 8.0)
ROOM_HEIGHT_RANGE_M = (3.0, 4.0)
# Keep sources and microphone away from the walls: the image-source method is
# least accurate close to a boundary, and speakers do not stand against walls.
WALL_MARGIN_M = 0.5
SOURCE_HEIGHT_RANGE_M = (1.4, 1.9)
MIC_HEIGHT_M = 1.5
MIN_SOURCE_SEPARATION_M = 0.5
MAX_PLACEMENT_ATTEMPTS = 100
# Calibration: accept a room once its measured RT60 is within this relative
# distance of the target, or give up after this many corrections.
RT60_TOLERANCE = 0.05
MAX_CALIBRATION_STEPS = 8


@dataclass(frozen=True)
class RoomResponse:
    """Impulse responses for one room, with the reverberation actually achieved."""

    rirs: tuple[np.ndarray, ...]
    target_rt60_s: float
    measured_rt60_s: float
    room_dim_m: tuple[float, float, float]

    @property
    def is_anechoic(self) -> bool:
        return self.target_rt60_s <= 0.0


def simulate_room_rirs(
    rt60_s: float,
    n_sources: int,
    *,
    sr: int = SAMPLE_RATE,
    rng: np.random.Generator | None = None,
    tolerance: float = RT60_TOLERANCE,
) -> RoomResponse:
    """Generate one RIR per source for a randomly drawn, calibrated room.

    Args:
        rt60_s: target reverberation time in seconds. Zero (or negative) means
            anechoic and returns unit impulses, so callers can treat the dry
            condition as just another level of the reverberation axis.
        n_sources: how many independent source positions to place.
        sr: sample rate of the returned impulse responses.
        rng: seeded generator; pass one to make a benchmark cell reproducible.
        tolerance: relative error at which calibration stops.

    Returns:
        A RoomResponse holding ``n_sources`` impulse responses from one room,
        plus the measured RT60 — always prefer that over ``rt60_s`` when
        labelling data.

    Raises:
        ValueError: if fewer than one source is requested, or if the target
            RT60 is unreachable in the drawn room (Sabine inversion has no
            solution below a fully absorbent room's decay).
    """
    if n_sources < 1:
        raise ValueError("at least one source is required")

    rng = rng if rng is not None else np.random.default_rng()
    room_dim = np.array(
        [
            rng.uniform(*ROOM_LENGTH_RANGE_M),
            rng.uniform(*ROOM_WIDTH_RANGE_M),
            rng.uniform(*ROOM_HEIGHT_RANGE_M),
        ]
    )

    if rt60_s <= 0.0:
        return RoomResponse(
            rirs=tuple(_unit_impulse() for _ in range(n_sources)),
            target_rt60_s=rt60_s,
            measured_rt60_s=0.0,
            room_dim_m=tuple(room_dim),
        )

    mic_position = _sample_position(room_dim, MIC_HEIGHT_M, rng)
    source_positions = _sample_source_positions(room_dim, n_sources, rng)

    best_rirs: tuple[np.ndarray, ...] | None = None
    best_measured = float("nan")
    request = rt60_s

    for step in range(MAX_CALIBRATION_STEPS):
        try:
            rirs = _build_rirs(room_dim, mic_position, source_positions, request, sr)
        except ValueError as error:
            if best_rirs is None:
                raise ValueError(
                    f"RT60 of {rt60_s} s is not reachable in a "
                    f"{room_dim.round(2).tolist()} m room"
                ) from error
            break

        measured = float(np.mean([measure_rt60(rir, sr) for rir in rirs]))
        if not np.isfinite(measured) or measured <= 0.0:
            break

        if best_rirs is None or abs(measured - rt60_s) < abs(best_measured - rt60_s):
            best_rirs, best_measured = rirs, measured

        if abs(measured - rt60_s) / rt60_s <= tolerance:
            break

        # Measured RT60 is close to proportional to the requested one, so a
        # multiplicative correction converges in a couple of steps.
        request *= rt60_s / measured
        if step == MAX_CALIBRATION_STEPS - 1:
            logger.warning(
                "RT60 calibration stopped at %.3f s for a %.3f s target",
                best_measured,
                rt60_s,
            )

    if best_rirs is None:
        raise ValueError(f"could not simulate a room with RT60 {rt60_s} s")

    return RoomResponse(
        rirs=best_rirs,
        target_rt60_s=rt60_s,
        measured_rt60_s=best_measured,
        room_dim_m=tuple(room_dim),
    )


def _build_rirs(
    room_dim: np.ndarray,
    mic_position: np.ndarray,
    source_positions: list[np.ndarray],
    rt60_request_s: float,
    sr: int,
) -> tuple[np.ndarray, ...]:
    """Compute the RIRs of a shoebox whose absorption targets ``rt60_request_s``."""
    absorption, max_order = pra.inverse_sabine(rt60_request_s, room_dim)
    room = pra.ShoeBox(
        room_dim,
        fs=sr,
        materials=pra.Material(absorption),
        max_order=max_order,
    )
    room.add_microphone(mic_position)
    for position in source_positions:
        room.add_source(position)
    room.compute_rir()
    return tuple(
        np.asarray(room.rir[0][index], dtype=np.float32)
        for index in range(len(source_positions))
    )


def measure_rt60(rir: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Estimate the RT60 of an impulse response by Schroeder backward integration.

    Used to verify that a simulated RIR actually decays at its target rate —
    a benchmark axis is only metric if the requested value is the delivered one.
    """
    return float(pra.experimental.measure_rt60(np.asarray(rir), fs=sr))


def _unit_impulse(length: int = 1) -> np.ndarray:
    impulse = np.zeros(length, dtype=np.float32)
    impulse[0] = 1.0
    return impulse


def _sample_position(
    room_dim: np.ndarray, height_m: float, rng: np.random.Generator
) -> np.ndarray:
    return np.array(
        [
            rng.uniform(WALL_MARGIN_M, room_dim[0] - WALL_MARGIN_M),
            rng.uniform(WALL_MARGIN_M, room_dim[1] - WALL_MARGIN_M),
            height_m,
        ]
    )


def _sample_source_positions(
    room_dim: np.ndarray, n_sources: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Place sources at least MIN_SOURCE_SEPARATION_M apart.

    Two speakers at nearly the same point would produce near-identical RIRs,
    which is not a reverberation condition but a degenerate mixture.
    """
    positions: list[np.ndarray] = []
    for _ in range(n_sources):
        for attempt in range(MAX_PLACEMENT_ATTEMPTS):
            candidate = _sample_position(
                room_dim, rng.uniform(*SOURCE_HEIGHT_RANGE_M), rng
            )
            if all(
                np.linalg.norm(candidate - placed) >= MIN_SOURCE_SEPARATION_M
                for placed in positions
            ):
                positions.append(candidate)
                break
            if attempt == MAX_PLACEMENT_ATTEMPTS - 1:
                logger.warning(
                    "could not separate source %d by %.2f m after %d attempts; "
                    "accepting a closer position",
                    len(positions),
                    MIN_SOURCE_SEPARATION_M,
                    MAX_PLACEMENT_ATTEMPTS,
                )
                positions.append(candidate)
    return positions
