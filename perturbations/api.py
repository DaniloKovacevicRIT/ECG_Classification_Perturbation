"""
User-facing API surface for ECG perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .noise import band_limited_noise, baseline_wander


@dataclass
class PerturbationConfig:
    ptype: str
    strength: float
    center_time: float
    window_seconds: Optional[float] = None
    target_class: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_extra(self, key: str, default: Any = None) -> Any:
        return self.extra.get(key, default)


def apply_perturbation(
    x: np.ndarray,
    *,
    fs: float,
    config: PerturbationConfig,
    model: Optional[Any] = None,
    r_peaks: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply the configured perturbation to `x`.
    """

    ptype = config.ptype.lower()
    strength = float(np.clip(config.strength, 0.0, 1.0))
    center_time = float(config.center_time)
    window_seconds = config.window_seconds

    if ptype == "baseline_wander":
        return baseline_wander(
            x,
            fs=fs,
            strength=strength,
            center_time=center_time,
            window_seconds=window_seconds,
            alpha=config.get_extra("alpha"),
            freq_range=config.get_extra("f_range"),
            rng=rng,
        )
    if ptype == "band_noise":
        return band_limited_noise(
            x,
            fs=fs,
            strength=strength,
            center_time=center_time,
            window_seconds=window_seconds,
            beta=config.get_extra("beta"),
            band=config.get_extra("band"),
            rng=rng,
        )

    raise ValueError(f"Unsupported perturbation type '{config.ptype}' for current implementation.")

