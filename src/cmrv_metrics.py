"""Core CMRV estimands.

The functions in this module implement the formula-level audit objects described
in the manuscript. They do not include a fitted clinical model, patient data, or
a claim that finite-sample model-class contrasts are oracle information.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


TIER_ORDER = ("I0", "I1", "I2", "I3")


def _one_dimensional(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def binary_log_loss(
    outcomes: Sequence[float], probabilities: Sequence[float], *, clip: float = 1e-6
) -> np.ndarray:
    """Return per-observation binary log loss after prespecified clipping."""
    y = _one_dimensional(outcomes, "outcomes")
    p = _one_dimensional(probabilities, "probabilities")
    if y.shape != p.shape:
        raise ValueError("outcomes and probabilities must have the same shape")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("outcomes must contain only 0 and 1")
    if not 0.0 < clip < 0.5:
        raise ValueError("clip must lie strictly between 0 and 0.5")
    p = np.clip(p, clip, 1.0 - clip)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def implemented_fusion_value(
    outcomes: Sequence[float],
    clinical_probabilities: Sequence[float],
    fusion_probabilities: Sequence[float],
    *,
    clip: float = 1e-6,
) -> float:
    """Estimate IFV as mean paired clinical loss minus mean paired fusion loss."""
    clinical = binary_log_loss(outcomes, clinical_probabilities, clip=clip)
    fusion = binary_log_loss(outcomes, fusion_probabilities, clip=clip)
    if clinical.shape != fusion.shape:
        raise ValueError("clinical and fusion probabilities must have the same shape")
    return float(np.mean(clinical - fusion))


def potential_information_value(
    oracle_clinical_risk: float, oracle_fusion_risk: float, *, tolerance: float = 1e-12
) -> float:
    """Return oracle PIV; valid only when both risks are oracle-optimal risks."""
    value = float(oracle_clinical_risk) - float(oracle_fusion_risk)
    if value < -tolerance:
        raise ValueError("oracle PIV cannot be negative beyond numerical tolerance")
    return max(value, 0.0)


def fusion_implementation_gap(piv: float, ifv: float) -> float:
    """Return signed FIG = PIV - IFV without truncation."""
    return float(piv) - float(ifv)


def normalized_value(value_nats: float, outcome_entropy_nats: float) -> float:
    """Normalize a signed value by positive outcome entropy."""
    entropy = float(outcome_entropy_nats)
    if not np.isfinite(entropy) or entropy <= 0.0:
        raise ValueError("outcome_entropy_nats must be finite and positive")
    return float(value_nats) / entropy


def information_saturation(values_by_tier: Mapping[str, float]) -> float:
    """Return mean(I0, I1) minus mean(I2, I3)."""
    missing = set(TIER_ORDER).difference(values_by_tier)
    if missing:
        raise ValueError(f"missing tiers: {sorted(missing)}")
    values = {tier: float(values_by_tier[tier]) for tier in TIER_ORDER}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("tier values must be finite")
    low = (values["I0"] + values["I1"]) / 2.0
    high = (values["I2"] + values["I3"]) / 2.0
    return low - high


def saturation_point(
    piv_by_tier: Mapping[str, float], *, fraction_of_i0: float = 0.25
) -> str | None:
    """Return the first tier at or below fraction_of_i0, or None if absent."""
    if not 0.0 <= fraction_of_i0 <= 1.0:
        raise ValueError("fraction_of_i0 must lie between 0 and 1")
    missing = set(TIER_ORDER).difference(piv_by_tier)
    if missing:
        raise ValueError(f"missing tiers: {sorted(missing)}")
    baseline = float(piv_by_tier["I0"])
    if not np.isfinite(baseline) or baseline <= 0.0:
        return None
    threshold = fraction_of_i0 * baseline
    for tier in TIER_ORDER:
        value = float(piv_by_tier[tier])
        if not np.isfinite(value):
            raise ValueError("tier values must be finite")
        if value <= threshold:
            return tier
    return None
