"""Pure-numpy metrics. Each returns None when its inputs cannot support a valid value
(e.g. ROC-AUC with a single class) instead of inventing a number."""
from __future__ import annotations

import numpy as np


def _check(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if y_true.size == 0:
        return None, None
    return y_true, y_pred


def accuracy(y_true, y_pred):
    t, p = _check(y_true, y_pred)
    return None if t is None else float((t == p).mean())


def precision_recall_f1(y_true, y_pred, positive=1):
    t, p = _check(y_true, y_pred)
    if t is None:
        return None, None, None
    tp = int(((t == positive) & (p == positive)).sum())
    fp = int(((t != positive) & (p == positive)).sum())
    fn = int(((t == positive) & (p != positive)).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec is not None and rec is not None and (prec + rec) > 0 else None
    return prec, rec, f1


def roc_auc(y_true, scores):
    """Mann-Whitney formulation with average ranks for ties. None unless both classes are present."""
    t = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if t.shape != s.shape:
        raise ValueError("y_true and scores must have the same length")
    pos, neg = int((t == 1).sum()), int((t == 0).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[t == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def confusion_matrix(y_true, y_pred, labels):
    t, p = _check(y_true, y_pred)
    if t is None:
        return None
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b in zip(t, p):
        m[idx[a], idx[b]] += 1
    return m


def macro_f1(y_true, y_pred, labels):
    f1s = [precision_recall_f1(y_true, y_pred, positive=l)[2] for l in labels]
    f1s = [0.0 if f is None else f for f in f1s]
    return None if not len(np.asarray(y_true)) else float(np.mean(f1s))


def mae(y_true, y_pred):
    t, p = _check(y_true, y_pred)
    return None if t is None else float(np.abs(t.astype(float) - p.astype(float)).mean())


def temporal_iou(a, b) -> float:
    """a, b = (start, end) in seconds."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_events(pred, gt, iou_thr=0.5):
    """Greedy one-to-one matching by descending tIoU. Returns (matches[(pi, gi, iou)], fp, fn)."""
    pairs = sorted(((temporal_iou(p, g), pi, gi) for pi, p in enumerate(pred) for gi, g in enumerate(gt)),
                   reverse=True)
    used_p, used_g, matches = set(), set(), []
    for iou, pi, gi in pairs:
        if iou < iou_thr:
            break
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, iou))
    return matches, len(pred) - len(used_p), len(gt) - len(used_g)


def event_prf(n_tp: int, n_fp: int, n_fn: int):
    prec = n_tp / (n_tp + n_fp) if n_tp + n_fp else None
    rec = n_tp / (n_tp + n_fn) if n_tp + n_fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    return prec, rec, f1
