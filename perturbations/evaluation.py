"""
Evaluation utilities for ECG perturbations.

These helpers standardize prediction, attack logging, metrics, and aggregation so
that different perturbation families can share a single analysis pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from .api import PerturbationConfig, apply_perturbation
from .config import CLASS_TO_INDEX, CLASS_NAMES


@dataclass
class AttackResult:
    """
    Structured representation of a single perturbation attempt on one sample.
    """

    record_id: Optional[Any]
    ptype: str
    strength: float
    center_time: float
    window_seconds: Optional[float]
    target_class: Optional[str]
    target_mode: Optional[str]
    config_extra: Dict[str, Any]
    y_true: np.ndarray
    y_hat_clean: np.ndarray
    y_hat_adv: np.ndarray
    y_proba_clean: np.ndarray
    y_proba_adv: np.ndarray
    delta_norm_l2: float
    delta_norm_linf: float
    delta_smoothness: Optional[float]
    untargeted_success: bool
    targeted_success: Optional[bool]


def predict_proba_single(model, x: np.ndarray) -> np.ndarray:
    """
    Run the model on a single sample shaped (samples, leads) and return probs.
    """

    x = np.asarray(x, dtype=np.float32)
    return np.asarray(model.predict(x[None, ...], verbose=0)[0])


def predict_proba_batch(model, X: np.ndarray) -> np.ndarray:
    """
    Run the model on a batch shaped (N, samples, leads) and return probs.
    """

    X = np.asarray(X, dtype=np.float32)
    return np.asarray(model.predict(X, verbose=0))


def binarize_predictions(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert probability vectors to binary predictions using the given threshold.
    """

    proba = np.asarray(proba)
    return (proba >= threshold).astype(int)


def compute_smoothness(delta: np.ndarray) -> float:
    """
    Smoothness metric defined as the mean squared first-order difference.
    """

    diffs = np.diff(delta, axis=0)
    return float(np.mean(np.square(diffs)))


def run_attack_on_sample(
    model,
    x_clean: np.ndarray,
    y_true: np.ndarray,
    config: PerturbationConfig,
    *,
    fs: float,
    rng: Optional[np.random.Generator] = None,
    threshold: float = 0.5,
    compute_delta_smoothness: bool = True,
) -> Tuple[AttackResult, np.ndarray]:
    """
    Apply the configured perturbation, evaluate predictions, and log results.
    """

    x_clean = np.asarray(x_clean, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=int)

    x_adv = apply_perturbation(
        x_clean,
        fs=fs,
        config=config,
        model=model,
        y_true=y_true,
        rng=rng,
    )

    proba_clean = predict_proba_single(model, x_clean)
    proba_adv = predict_proba_single(model, x_adv)
    y_hat_clean = binarize_predictions(proba_clean, threshold=threshold)
    y_hat_adv = binarize_predictions(proba_adv, threshold=threshold)

    delta = x_adv - x_clean
    delta_norm_l2 = float(np.linalg.norm(delta))
    delta_norm_linf = float(np.max(np.abs(delta)))
    delta_smoothness = (
        compute_smoothness(delta) if compute_delta_smoothness else None
    )

    untargeted_success = bool(np.any(y_hat_clean != y_hat_adv))
    targeted_success: Optional[bool] = None
    if config.target_class is not None:
        idx = CLASS_TO_INDEX[config.target_class]
        mode = config.extra.get("target_mode", "force") if config.extra else "force"
        if mode == "suppress":
            targeted_success = bool(y_hat_clean[idx] == 1 and y_hat_adv[idx] == 0)
        else:
            targeted_success = bool(y_hat_adv[idx] == 1)

    result = AttackResult(
        record_id=config.extra.get("record_id") if config.extra else None,
        ptype=config.ptype,
        strength=config.strength,
        center_time=config.center_time,
        window_seconds=config.window_seconds,
        target_class=config.target_class,
        target_mode=config.extra.get("target_mode") if config.extra else None,
        config_extra=config.extra or {},
        y_true=y_true,
        y_hat_clean=y_hat_clean,
        y_hat_adv=y_hat_adv,
        y_proba_clean=proba_clean,
        y_proba_adv=proba_adv,
        delta_norm_l2=delta_norm_l2,
        delta_norm_linf=delta_norm_linf,
        delta_smoothness=delta_smoothness,
        untargeted_success=untargeted_success,
        targeted_success=targeted_success,
    )

    return result, x_adv


def compute_untargeted_asr(results: Sequence[AttackResult]) -> float:
    """
    Fraction of attacks where the binary prediction changed.
    """

    if not results:
        return 0.0
    return float(
        np.mean([1.0 if res.untargeted_success else 0.0 for res in results])
    )


def compute_targeted_success(
    results: Sequence[AttackResult],
    *,
    class_name: str,
    mode: str = "force",
) -> Tuple[float, int]:
    """
    Compute targeted success for a specific class and mode ('force' or 'suppress').
    """

    filtered = [
        res
        for res in results
        if res.target_class == class_name
        and (res.target_mode or "force") == mode
    ]
    if not filtered:
        return 0.0, 0

    idx = CLASS_TO_INDEX[class_name]
    if mode == "suppress":
        successes = [
            1.0 if (res.y_hat_clean[idx] == 1 and res.y_hat_adv[idx] == 0) else 0.0
            for res in filtered
        ]
    else:
        successes = [1.0 if res.y_hat_adv[idx] == 1 else 0.0 for res in filtered]
    return float(np.mean(successes)), len(filtered)


def results_to_dataframe(
    results: Sequence[AttackResult],
    *,
    include_probabilities: bool = False,
    include_vectors: bool = False,
) -> pd.DataFrame:
    """
    Convert a list of AttackResult objects into a DataFrame.
    """

    rows: List[Dict[str, Any]] = []
    for res in results:
        row = {
            "record_id": res.record_id,
            "ptype": res.ptype,
            "strength": res.strength,
            "center_time": res.center_time,
            "window_seconds": res.window_seconds,
            "target_class": res.target_class,
            "target_mode": res.target_mode,
            "delta_norm_l2": res.delta_norm_l2,
            "delta_norm_linf": res.delta_norm_linf,
            "delta_smoothness": res.delta_smoothness,
            "untargeted_success": res.untargeted_success,
            "targeted_success": res.targeted_success,
        }
        if include_probabilities:
            row["y_proba_clean"] = res.y_proba_clean
            row["y_proba_adv"] = res.y_proba_adv
        if include_vectors:
            row["y_true"] = res.y_true
            row["y_hat_clean"] = res.y_hat_clean
            row["y_hat_adv"] = res.y_hat_adv
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_norms(results: Sequence[AttackResult]) -> pd.DataFrame:
    """
    Group by (ptype, strength) and summarize L2/Linf norms.
    """

    df = results_to_dataframe(results)
    if df.empty:
        return df
    grouped = (
        df.groupby(["ptype", "strength"])[["delta_norm_l2", "delta_norm_linf"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    return grouped


def summarize_smoothness(results: Sequence[AttackResult]) -> pd.DataFrame:
    """
    Group by (ptype, strength) and summarize smoothness metrics.
    """

    df = results_to_dataframe(results)
    if df.empty or "delta_smoothness" not in df:
        return df
    grouped = (
        df.groupby(["ptype", "strength"])["delta_smoothness"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return grouped


def compute_global_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute per-class ROC AUC and thresholded precision/recall/F1 metrics.
    """

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    metrics: Dict[str, Any] = {}

    try:
        auc_per_class = sk_metrics.roc_auc_score(
            y_true, y_proba, average=None, multi_class="ovr"
        )
    except ValueError:
        auc_per_class = [np.nan] * y_true.shape[1]
    metrics["auc_per_class"] = dict(zip(CLASS_NAMES, auc_per_class))
    metrics["auc_macro"] = float(
        np.nanmean(list(metrics["auc_per_class"].values()))
    )

    y_pred = (y_proba >= threshold).astype(int)
    precision, recall, f1, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics["precision_per_class"] = dict(zip(CLASS_NAMES, precision))
    metrics["recall_per_class"] = dict(zip(CLASS_NAMES, recall))
    metrics["f1_per_class"] = dict(zip(CLASS_NAMES, f1))
    metrics["f1_macro"] = float(np.mean(f1))
    return metrics
