"""
Visualization utilities for ECG perturbations.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CLASS_NAMES
from .evaluation import AttackResult, results_to_dataframe


def _ensure_axes(ax=None, nrows=1, ncols=1, figsize=(10, 6)):
    if ax is not None:
        return None, ax
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    return fig, axes


def plot_triptych(
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    *,
    fs: float,
    lead_idx: int = 1,
    title: Optional[str] = None,
    zoom: Optional[Tuple[float, float]] = None,
    share_y: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot clean, perturbed, and difference signals for a single lead.
    """

    times = np.arange(x_clean.shape[0]) / fs
    lead_idx = int(lead_idx)
    x_clean_lead = x_clean[:, lead_idx]
    x_adv_lead = x_adv[:, lead_idx]
    delta_lead = x_adv_lead - x_clean_lead

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax_clean, ax_adv, ax_delta = axes

    ax_clean.plot(times, x_clean_lead, color="tab:blue")
    ax_clean.set_ylabel("Amplitude")
    ax_clean.set_title(f"Lead {lead_idx + 1} – Clean")

    ax_adv.plot(times, x_adv_lead, color="tab:orange")
    ax_adv.set_ylabel("Amplitude")
    ax_adv.set_title(f"Lead {lead_idx + 1} – Perturbed")

    if share_y:
        ymin = min(ax_clean.get_ylim()[0], ax_adv.get_ylim()[0])
        ymax = max(ax_clean.get_ylim()[1], ax_adv.get_ylim()[1])
        ax_clean.set_ylim(ymin, ymax)
        ax_adv.set_ylim(ymin, ymax)

    ax_delta.plot(times, delta_lead, color="tab:green")
    ax_delta.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_delta.set_ylabel("Difference")
    ax_delta.set_xlabel("Time (s)")
    ax_delta.set_title("Perturbed − Clean")
    max_delta = np.max(np.abs(delta_lead))
    ax_delta.text(
        0.01,
        0.9,
        f"max |Δ| = {max_delta:.4f}",
        transform=ax_delta.transAxes,
        fontsize=9,
    )

    if zoom:
        ax_clean.set_xlim(zoom)
        ax_adv.set_xlim(zoom)
        ax_delta.set_xlim(zoom)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig, axes


def plot_overlay_with_diff(
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    *,
    fs: float,
    lead_idx: int = 1,
    zoom: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot clean vs perturbed traces overlayed plus a difference panel.
    """

    times = np.arange(x_clean.shape[0]) / fs
    lead_idx = int(lead_idx)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax_overlay, ax_diff = axes

    ax_overlay.plot(times, x_clean[:, lead_idx], label="Clean", color="tab:blue")
    ax_overlay.plot(
        times, x_adv[:, lead_idx], label="Perturbed", color="tab:orange", alpha=0.8
    )
    ax_overlay.set_ylabel("Amplitude")
    ax_overlay.legend(loc="upper right")
    ax_overlay.set_title(f"Lead {lead_idx + 1} – Overlay")

    delta = x_adv[:, lead_idx] - x_clean[:, lead_idx]
    ax_diff.plot(times, delta, color="tab:green")
    ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_diff.set_ylabel("Difference")
    ax_diff.set_xlabel("Time (s)")
    ax_diff.set_title("Perturbed − Clean")

    if zoom:
        ax_overlay.set_xlim(zoom)
        ax_diff.set_xlim(zoom)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig, axes


def plot_asr_vs_strength(
    results: Sequence[AttackResult],
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot untargeted ASR as a function of strength for each perturbation type.
    """

    df = results_to_dataframe(results)
    if df.empty:
        raise ValueError("No results provided for ASR plotting.")
    grouped = (
        df.groupby(["ptype", "strength"])["untargeted_success"]
        .mean()
        .reset_index()
    )

    fig, ax = _ensure_axes(ax, figsize=(8, 5))
    pivot = grouped.pivot(index="strength", columns="ptype", values="untargeted_success")
    for ptype in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[ptype],
            marker="o",
            label=ptype,
        )

    ax.set_xlabel("Strength")
    ax.set_ylabel("Untargeted ASR")
    ax.set_ylim(0, 1)
    ax.legend(title="Perturbation")
    if title:
        ax.set_title(title)
    return fig, ax


def plot_norm_distributions(
    results: Sequence[AttackResult],
    *,
    value: str = "delta_norm_l2",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Boxplot of norm/smoothness metrics grouped by perturbation type and strength.
    """

    df = results_to_dataframe(results)
    if df.empty or value not in df.columns:
        raise ValueError(f"No '{value}' values available for plotting.")

    df["label"] = df.apply(
        lambda row: f"{row['ptype']} (s={row['strength']})", axis=1
    )

    labels = df["label"].unique()
    data = [df[df["label"] == label][value].dropna() for label in labels]

    fig, ax = _ensure_axes(ax, figsize=(10, 5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel(value)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_classwise_metric_bars(
    metrics: Dict[str, Dict[str, float]],
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    metric_name: str = "AUC",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot per-class metrics for different evaluation conditions.
    """

    conditions = list(metrics.keys())
    classes = CLASS_NAMES
    x = np.arange(len(classes))
    width = 0.8 / max(1, len(conditions))

    fig, ax = _ensure_axes(ax, figsize=(10, 5))
    for idx, condition in enumerate(conditions):
        scores = [metrics[condition].get(cls, np.nan) for cls in classes]
        offset = (idx - (len(conditions) - 1) / 2) * width
        ax.bar(x + offset, scores, width=width, label=condition)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15)
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, 1)
    ax.legend(title="Condition")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax

