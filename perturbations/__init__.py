from .adv_smooth import smooth_adversarial_perturbation
from .api import PerturbationConfig, apply_perturbation
from .evaluation import (
    AttackResult,
    binarize_predictions,
    compute_global_metrics,
    compute_targeted_success,
    compute_untargeted_asr,
    predict_proba_batch,
    predict_proba_single,
    results_to_dataframe,
    run_attack_on_sample,
    summarize_norms,
    summarize_smoothness,
)
from .morphology import detect_r_peaks, local_amplitude_scaling, local_time_warp
from .noise import band_limited_noise, baseline_wander
from visualization import (
    plot_asr_vs_strength,
    plot_classwise_metric_bars,
    plot_norm_distributions,
    plot_overlay_with_diff,
    plot_triptych,
)

__all__ = [
    "PerturbationConfig",
    "apply_perturbation",
    "baseline_wander",
    "band_limited_noise",
    "local_amplitude_scaling",
    "local_time_warp",
    "detect_r_peaks",
    "smooth_adversarial_perturbation",
    "AttackResult",
    "predict_proba_single",
    "predict_proba_batch",
    "binarize_predictions",
    "run_attack_on_sample",
    "results_to_dataframe",
    "summarize_norms",
    "summarize_smoothness",
    "compute_untargeted_asr",
    "compute_targeted_success",
    "compute_global_metrics",
    "plot_triptych",
    "plot_overlay_with_diff",
    "plot_asr_vs_strength",
    "plot_norm_distributions",
    "plot_classwise_metric_bars",
]
