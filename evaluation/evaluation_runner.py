"""Compare stored analysis runs with human ground truth.

Annotation JSON (one file per video):
  {"events": [{"start": 31.0, "end": 39.0, "severity": 3}, ...]}      # severity optional (1-3)

Window label = positive if the window overlaps any ground-truth event by >= --min-overlap seconds.
Results aggregate across all --pair arguments. Metrics are only meaningful with a real, sufficiently
large annotated set; a single video must not be reported as a test accuracy.

Usage (PowerShell):
  python -m evaluation.evaluation_runner --pair 3:annotations/ep1.json --pair 4:annotations/ep2.json
"""
from __future__ import annotations

import argparse
import json

from config.settings import DB_PATH
from database.repository import Repository
from evaluation import metrics as M


def evaluate(repo: Repository, pairs: list[tuple[int, dict]], iou_thresholds=(0.3, 0.5), min_overlap=1.0) -> dict:
    y_true, y_pred, scores = [], [], []
    ev_tot = {t: [0, 0, 0] for t in iou_thresholds}
    sev_true, sev_pred = [], []
    skipped = []
    for run_id, ann in pairs:
        run = repo.get_run(run_id)
        if run is None or run["status"] not in ("completed", "completed_with_warnings"):
            skipped.append({"run_id": run_id, "reason": "run missing or not complete"})
            continue
        gt = [(float(e["start"]), float(e["end"])) for e in ann.get("events", [])]
        gt_sev = [e.get("severity") for e in ann.get("events", [])]
        thr = json.loads(run["config_json"])["threshold"]
        for w in repo.list_windows(run_id):
            if w["p_violent"] is None:
                continue
            ov = max([max(0.0, min(w["end_s"], g[1]) - max(w["start_s"], g[0])) for g in gt] or [0.0])
            y_true.append(int(ov >= min_overlap))
            y_pred.append(int(w["p_violent"] >= thr))
            scores.append(w["p_violent"])
        events = repo.list_events(run_id)
        pred = [(e["start_s"], e["end_s"]) for e in events]
        for t in iou_thresholds:
            matches, fp, fn = M.match_events(pred, gt, t)
            ev_tot[t][0] += len(matches)
            ev_tot[t][1] += fp
            ev_tot[t][2] += fn
        matches, _, _ = M.match_events(pred, gt, min(iou_thresholds))
        for pi, gi, _ in matches:
            if gt_sev[gi] is not None and events[pi]["severity_score"] is not None:
                sev_true.append(int(gt_sev[gi]))
                sev_pred.append(int(events[pi]["severity_score"]))

    out: dict = {"n_runs_used": len(pairs) - len(skipped), "skipped": skipped, "n_windows": len(y_true)}
    if y_true:
        p, r, f = M.precision_recall_f1(y_true, y_pred)
        out["window_level"] = {"accuracy": M.accuracy(y_true, y_pred), "precision": p, "recall": r, "f1": f,
                               "roc_auc": M.roc_auc(y_true, scores),
                               "n_positive_windows_gt": int(sum(y_true))}
    out["event_level"] = {}
    for t, (tp, fp, fn) in ev_tot.items():
        p, r, f = M.event_prf(tp, fp, fn)
        out["event_level"][f"tIoU>={t}"] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}
    if sev_true:
        out["severity_on_matched_events"] = {
            "n": len(sev_true), "accuracy": M.accuracy(sev_true, sev_pred), "macro_f1": M.macro_f1(sev_true, sev_pred, [1, 2, 3]),
            "mae": M.mae(sev_true, sev_pred), "confusion_matrix_labels_1_2_3": M.confusion_matrix(sev_true, sev_pred, [1, 2, 3]).tolist()}
    else:
        out["severity_on_matched_events"] = "not computed: no matched events with ground-truth severity"
    out["note"] = ("Description metrics (BLEU/ROUGE/CIDEr/BERTScore) are not computed: no captioning model is "
                   "integrated and no reference captions are supplied.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True, help="RUN_ID:annotations.json")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--min-overlap", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    repo = Repository(a.db)
    pairs = []
    for spec in a.pair:
        rid, path = spec.split(":", 1)
        with open(path, encoding="utf-8") as f:
            pairs.append((int(rid), json.load(f)))
    res = evaluate(repo, pairs, min_overlap=a.min_overlap)
    text = json.dumps(res, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
