# Evaluation plan

## Annotation (needed for any real metric)
Per video, JSON: `{"events":[{"start":31.0,"end":39.0,"severity":3}]}` (seconds; severity 1-3, optional).
Severity rubric to document in the paper (proposal, to be agreed with the client): 0 none; 1 mild aggression (threat, chase, single
light shove); 2 single clear physical attack; 3 repeated/high-intensity attacks, weapon use or sustained combat. At least two
independent annotators on a subset; report inter-annotator agreement (e.g. Cohen's kappa) before using labels.

## Splits
Split by **video/episode** (never by window) into train/val/test; keep a cartoon-style-disjoint test set if possible.

## Metrics (implemented in `evaluation/`)
Window-level: accuracy, precision, recall, F1, ROC-AUC (window positive if overlap with a GT event >= 1 s, configurable).
Event-level: precision/recall/F1 at tIoU >= 0.3 and 0.5 with greedy one-to-one matching.
Severity (matched events): accuracy, macro-F1, confusion matrix, MAE.
Description metrics: not implemented until a captioning model and reference captions exist; add human evaluation.

## Experiments
| ID | Experiment | Status |
|---|---|---|
| BASE | Pretrained X3D-M, zero-shot on cartoons | Runnable now; **no results exist** |
| E1 | CNN-only (per-frame) | Not implemented |
| E2 | CNN + LSTM | Architecture only |
| E3 | CNN + Transformer | Architecture only |
| E4 | E3 + learned severity | Not implemented |

Report threshold selection on validation data only; report runtime measured on the target CPU.
