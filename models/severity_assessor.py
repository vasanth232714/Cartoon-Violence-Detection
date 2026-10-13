"""Heuristic review-priority estimate (not a trained model and not violence severity).

Legacy numeric labels are retained for stored-report compatibility: Mild, Moderate, High.
Model probability is deliberately NOT an input: confidence in "violence" is not severity.

Inputs measured from the video / predictions:
  duration_s        length of the candidate event
  n_positive_runs   separate runs of positive windows (repetition proxy)
  motion_mean       mean absolute grayscale frame difference (0-255), shot cuts excluded

Points:  duration  <4s:0   4-<10s:1   >=10s:2
         repetition 1 run:0  2 runs:1  >=3 runs:2
         motion     <8:0    8-20:1     >20:2
Total 0-6 -> score 1 (0-1 pts), 2 (2-3 pts), 3 (>=4 pts).

Thresholds are initial, unvalidated defaults that must be calibrated against a human-annotated
severity rubric. Client-listed factors NOT measurable here: involved characters, dangerous objects,
action type (the binary model only separates violent / non-violent).
"""
from __future__ import annotations

from dataclasses import dataclass, field

METHOD = "review_priority_rule_v1 (unvalidated; not violence severity)"
LABELS = {0: "No violence", 1: "Mild", 2: "Moderate", 3: "High"}


@dataclass
class SeverityResult:
    score: int | None
    label: str
    method: str = METHOD
    points: dict = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)


def assess_severity(duration_s: float, n_positive_runs: int, motion_mean: float | None) -> SeverityResult:
    if duration_s is None or duration_s <= 0 or n_positive_runs is None or n_positive_runs < 1:
        return SeverityResult(None, "Not assessed", rationale=["Insufficient event information."])
    if motion_mean is None:
        return SeverityResult(None, "Not assessed",
                              rationale=["Motion intensity could not be measured from the video frames."])
    d = 0 if duration_s < 4 else (1 if duration_s < 10 else 2)
    r = 0 if n_positive_runs <= 1 else (1 if n_positive_runs == 2 else 2)
    m = 0 if motion_mean < 8 else (1 if motion_mean <= 20 else 2)
    total = d + r + m
    score = 1 if total <= 1 else (2 if total <= 3 else 3)
    return SeverityResult(
        score, LABELS[score],
        points={"duration": d, "repetition": r, "motion": m, "total": total},
        rationale=[
            f"Duration {duration_s:.1f}s -> {d} pt(s)",
            f"{n_positive_runs} separate positive run(s) -> {r} pt(s)",
            f"Mean frame-difference motion {motion_mean:.1f}/255 -> {m} pt(s)",
            f"Total {total}/6 -> severity {score} ({LABELS[score]})",
        ],
    )
