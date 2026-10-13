"""Learn the decision rule from labelled cartoon windows and report an honest estimate.

  python -m training.fit --cache training_cache --holdout ep9.mp4,ep10.mp4

Steps: (1) out-of-fold (leave-video-out) adapter probabilities, raw vs adapted AUC; (2) grid-search of the post-processing
parameters on those out-of-fold scores; (3) train the final adapter; (4) evaluate on --holdout videos that were never used
for training or tuning, next to the raw baseline and the old hand-set preset; (5) write checkpoints/cartoon_adapter.npz
and config/tuned_settings.json.

Without --holdout, every printed number for the tuned settings is optimistic (tuned on the same data). Say so in any paper.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation import metrics as M
from models.adapter import CartoonAdapter
from models.transformer_refiner import RefinerParams
from pipeline.postprocess import PostParams, run_postprocess

HAND_SET_PRESET = PostParams(threshold=0.55, threshold_low=None, merge_gap_s=1.0, min_event_s=1.5, refine=True,
                             suppress_cuts=True, filter_spikes=True)
RAW_BASELINE = PostParams(threshold=0.40, threshold_low=None, merge_gap_s=1.0, min_event_s=0.0, refine=False,
                          suppress_cuts=False, filter_spikes=False)


@dataclass
class VideoCache:
    name: str
    starts: np.ndarray
    ends: np.ndarray
    raw: np.ndarray
    feats: np.ndarray
    cut: np.ndarray
    labels: np.ndarray
    events: list
    window_s: float
    stride_s: float


def load_cache(cache_dir: str | Path) -> list[VideoCache]:
    out = []
    for f in sorted(Path(cache_dir).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        out.append(VideoCache(meta["name"], z["starts"], z["ends"], z["raw"].astype(float), z["feats"].astype(np.float32),
                              z["cut"].astype(bool), z["labels"], [tuple(e) for e in z["events"].tolist()],
                              meta["key"]["window_s"], meta["key"]["stride_s"]))
    return out


def _vec(v: VideoCache) -> np.ndarray:
    return v.feats  # already [2048 features + logit margin]


def _xy(videos: list[VideoCache]):
    X = np.concatenate([_vec(v)[v.labels >= 0] for v in videos]) if videos else np.empty((0, 0))
    y = np.concatenate([v.labels[v.labels >= 0] for v in videos]) if videos else np.empty(0)
    return X, y


def _pooled_auc(videos, probs):
    y = np.concatenate([v.labels[v.labels >= 0] for v in videos])
    p = np.concatenate([pr[v.labels >= 0] for v, pr in zip(videos, probs)])
    return M.roc_auc(y.astype(int), p)


def oof_probabilities(videos: list[VideoCache], C: float, n_folds: int = 5):
    """Leave-video-out (or grouped k-fold) adapter probabilities. Falls back to raw p where a fold cannot be trained."""
    n = len(videos)
    k = n if n <= 12 else n_folds
    folds = [list(range(i, n, k)) for i in range(k)]
    out = [v.raw.copy() for v in videos]
    fell_back = 0
    for held in folds:
        train = [videos[i] for i in range(n) if i not in held]
        X, y = _xy(train)
        if len(y) == 0 or len(np.unique(y)) < 2:
            fell_back += len(held)
            continue
        ad = CartoonAdapter.fit(X, y, C=C)
        for i in held:
            out[i] = ad.predict_proba(_vec(videos[i]))
    return out, fell_back


def evaluate_params(videos: list[VideoCache], probs: list[np.ndarray], params: PostParams, iou: float = 0.3) -> dict:
    tp = fp = fn = tn = 0
    etp = efp = efn = 0
    for v, pr in zip(videos, probs):
        _, events = run_postprocess(v.starts, v.ends, pr, v.cut, params)
        pred = [(e.start_s, e.end_s) for e in events]
        mid = (v.starts + v.ends) / 2
        flagged = np.array([any(s <= m <= e for s, e in pred) for m in mid]) if len(mid) else np.array([], bool)
        m = v.labels >= 0
        yt, yp = v.labels[m] == 1, flagged[m]
        tp += int((yt & yp).sum()); fp += int((~yt & yp).sum()); fn += int((yt & ~yp).sum()); tn += int((~yt & ~yp).sum())
        matches, efp_v, efn_v = M.match_events(pred, v.events, iou)
        etp += len(matches); efp += efp_v; efn += efn_v
    wp, wr, wf = M.event_prf(tp, fp, fn)
    ep, er, ef = M.event_prf(etp, efp, efn)
    f0 = lambda x: 0.0 if x is None else x
    return {"window": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": wp, "recall": wr, "f1": wf,
                       "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1)},
            "event": {"tp": etp, "fp": efp, "fn": efn, "precision": ep, "recall": er, "f1": ef, "tiou": iou},
            "score": 0.5 * f0(wf) + 0.5 * f0(ef)}


def default_grid() -> list[PostParams]:
    grid = []
    for sigma, refine in [(None, False), (0.75, True), (1.25, True), (2.0, True)]:
        for cuts, spikes in itertools.product([False, True], [False, True]):
            for thr in np.arange(0.20, 0.91, 0.05):
                for low_gap in (None, 0.10, 0.20):
                    low = None if low_gap is None else round(float(thr) - low_gap, 2)
                    if low is not None and low <= 0.05:
                        continue
                    for min_ev, gap in itertools.product((0.0, 2.0, 4.0), (0.0, 2.0)):
                        grid.append(PostParams(
                            threshold=round(float(thr), 2), threshold_low=low, merge_gap_s=gap, min_event_s=min_ev,
                            refine=refine, suppress_cuts=cuts, filter_spikes=spikes,
                            refiner=RefinerParams(sigma=sigma or 1.25)))
    return grid


def tune(videos, probs, grid=None, top: int = 5):
    grid = grid or default_grid()
    scored = [(evaluate_params(videos, probs, p)["score"], i, p) for i, p in enumerate(grid)]
    scored.sort(key=lambda t: (-t[0], t[1]))  # ties -> earlier (simpler/looser) grid entry; deterministic
    return scored[0][2], scored[:top]


def _fmt(r: dict) -> str:
    w, e = r["window"], r["event"]
    g = lambda x: "n/a" if x is None else f"{x:.3f}"
    return (f"window P={g(w['precision'])} R={g(w['recall'])} F1={g(w['f1'])} acc={w['accuracy']:.3f} "
            f"(FP={w['fp']} FN={w['fn']}) | event@tIoU{e['tiou']} P={g(e['precision'])} R={g(e['recall'])} "
            f"F1={g(e['f1'])} (FP={e['fp']} FN={e['fn']})")


def run_fit(videos: list[VideoCache], holdout_names: set[str], adapter_out: Path | None, settings_out: Path | None,
            log=print) -> dict:
    dev = [v for v in videos if v.name not in holdout_names]
    hold = [v for v in videos if v.name in holdout_names]
    report: dict = {"n_dev_videos": len(dev), "n_holdout_videos": len(hold), "created": time.strftime("%Y-%m-%d %H:%M")}
    if len(dev) < 2:
        raise ValueError("Need at least 2 labelled videos outside the holdout set.")
    X, y = _xy(dev)
    log(f"dev: {len(dev)} videos, {len(y)} labelled windows ({int(y.sum())} positive)")
    if len(np.unique(y)) < 2:
        raise ValueError("Labelled data contains only one class; need both violent and non-violent windows.")

    raw_auc = _pooled_auc(dev, [v.raw for v in dev])
    best_C, best_auc, oof = None, -1.0, None
    for C in (0.001, 0.01, 0.1, 1.0):
        probs, fb = oof_probabilities(dev, C)
        auc = _pooled_auc(dev, probs)
        log(f"  adapter C={C}: out-of-fold AUC={auc if auc is None else round(auc, 4)} (raw model AUC={raw_auc if raw_auc is None else round(raw_auc, 4)})")
        if auc is not None and auc > best_auc:
            best_C, best_auc, oof = C, auc, probs
    use_adapter = best_auc is not None and raw_auc is not None and best_auc >= raw_auc + 0.01
    report.update(raw_auc_oof=raw_auc, adapter_auc_oof=best_auc if oof else None, adapter_C=best_C,
                  adapter_used=bool(use_adapter))
    log("adapter " + ("IMPROVES held-out AUC -> will be used" if use_adapter else
                      "does not beat the raw model by >= 0.01 AUC -> NOT used (raw scores kept)"))
    tune_probs = oof if use_adapter else [v.raw for v in dev]

    best, top = tune(dev, tune_probs)
    for sc, _, p in top:
        log(f"  top grid: score={sc:.3f} thr={p.threshold} low={p.threshold_low} gap={p.merge_gap_s} minev={p.min_event_s} "
            f"refine={p.refine}/{p.refiner.sigma} cuts={p.suppress_cuts} spikes={p.filter_spikes}")
    report["best_params"] = best.to_dict()
    report["dev"] = {"raw_baseline": evaluate_params(dev, [v.raw for v in dev], RAW_BASELINE),
                     "hand_set_preset": evaluate_params(dev, [v.raw for v in dev], HAND_SET_PRESET),
                     "tuned_oof (OPTIMISTIC: tuned on this data)": evaluate_params(dev, tune_probs, best)}

    adapter = None
    if use_adapter:
        adapter = CartoonAdapter.fit(X, y, C=best_C, meta={
            "n_videos": len(dev), "cv_auc_adapter": round(float(best_auc), 4), "cv_auc_raw": round(float(raw_auc), 4),
            "trained_at": report["created"],
            "note": "cv_* = leave-video-out AUC on the labelled dev set"})
    if hold:
        hp = [adapter.predict_proba(_vec(v)) if adapter else v.raw for v in hold]
        report["holdout"] = {"raw_baseline": evaluate_params(hold, [v.raw for v in hold], RAW_BASELINE),
                             "hand_set_preset": evaluate_params(hold, [v.raw for v in hold], HAND_SET_PRESET),
                             "tuned": evaluate_params(hold, hp, best)}
    for section in ("dev", "holdout"):
        if section in report:
            log(f"--- {section.upper()} ---")
            for k, r in report[section].items():
                log(f"  {k:45s} {_fmt(r)}")
    if not hold:
        log("NOTE: no --holdout videos: the tuned numbers above are optimistic. Keep a separate test set for any claim.")

    if adapter_out is not None:
        if adapter:
            adapter.save(adapter_out)
            log(f"saved adapter -> {adapter_out}")
        elif Path(adapter_out).is_file():
            log(f"NOTE: existing {adapter_out} left untouched (not retrained this run)")
    if settings_out is not None:
        w, s = videos[0].window_s, videos[0].stride_s
        Path(settings_out).parent.mkdir(parents=True, exist_ok=True)
        with open(settings_out, "w", encoding="utf-8") as f:
            json.dump({"post": best.to_dict(), "window_seconds": w, "stride_seconds": s, "report": {
                k: report[k] for k in report if k not in ("best_params",)}}, f, indent=2, default=float)
        log(f"saved tuned settings -> {settings_out}")
    return report


def main() -> None:
    from config.settings import ADAPTER_PATH, TUNED_SETTINGS_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="training_cache")
    ap.add_argument("--holdout", default="", help="comma-separated video file names kept out of training AND tuning")
    ap.add_argument("--adapter-out", default=str(ADAPTER_PATH))
    ap.add_argument("--settings-out", default=str(TUNED_SETTINGS_PATH))
    ap.add_argument("--dry-run", action="store_true", help="evaluate only; write nothing")
    a = ap.parse_args()
    videos = load_cache(a.cache)
    if not videos:
        raise SystemExit(f"No cache files in {a.cache}. Run training.build_cache first.")
    hold = {h.strip() for h in a.holdout.split(",") if h.strip()}
    run_fit(videos, hold, None if a.dry_run else Path(a.adapter_out), None if a.dry_run else Path(a.settings_out))


if __name__ == "__main__":
    main()
