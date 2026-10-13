"""Evidence-grounded metrics summary; visual scene descriptions come from stored Qwen output."""
from __future__ import annotations

from preprocessing.timestamp_utils import format_timestamp as ft

DESCRIPTION_METHOD = "template (metrics only; Qwen explanation stored separately)"


def build_explanation(ev, config: dict, model_info: dict, severity, motion, n_evidence: int,
                      window_count_total: int) -> str:
    """Summarize measured model outputs without inventing an action category."""
    threshold = float(config["threshold"])
    temporal = model_info.get("temporal_transformer", {}) or {}
    temporal_status = temporal.get("status", "not configured")
    lines = [
        f"**Candidate interval:** {ft(ev.start_s)}–{ft(ev.end_s)} ({ev.duration_s:.2f}s).",
        f"**Visual model evidence:** {model_info.get('display_name', 'X3D-M')} produced a peak binary violence score "
        f"of {ev.peak_prob:.3f} near {ft(ev.peak_time_s)} and a mean of {ev.mean_prob:.3f} across {ev.n_windows} "
        f"positive window(s). Configured event threshold: {threshold:.2f}.",
        "**Action category:** Not available from the binary X3D checkpoint. See the separately stored Qwen explanation "
        "for a visual description when generation succeeded.",
        f"**Temporal sequence model:** {temporal_status}.",
        f"**Evidence frames:** {n_evidence} frame(s) were saved from the source video for review.",
    ]
    if motion is not None and motion.mean is not None:
        lines.append(
            f"**Motion statistic:** Mean frame-difference intensity {motion.mean:.1f}/255 across {motion.n_pairs} "
            f"frame pair(s); {motion.n_shot_cuts_excluded} detected cut(s) were excluded from this motion calculation. "
            "This measurement does not establish violence."
        )
    rationale = "; ".join(severity.rationale) or "No rule rationale was recorded."
    lines.append(
        f"**Heuristic review priority / Severity estimate:** {severity.label} ({severity.method}); based on duration, score bursts, and motion. "
        f"This is not a violence-severity estimate and is not validated on cartoons. {rationale}"
    )
    lines.append("**Review:** Check the original clip and evidence frames; automated candidates can be false positives.")
    return "\n\n".join(lines)
