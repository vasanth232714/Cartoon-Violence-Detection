"""Train the window-level temporal Transformer on cached, video-labeled X3D outputs.

Usage: python -m training.train_temporal_transformer --cache training_cache \\
       --holdout ep09.mp4,ep10.mp4 --output checkpoints/temporal_transformer.pt

The split is by video. The holdout videos are never used for gradients or model selection.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from training.fit import load_cache


def _chunks(features: np.ndarray, labels: np.ndarray, max_len: int, stride: int):
    starts = list(range(0, max(len(features) - max_len, 0) + 1, stride))
    tail = max(0, len(features) - max_len)
    if not starts or starts[-1] != tail:
        starts.append(tail)
    for start in starts:
        end = min(start + max_len, len(features))
        valid = labels[start:end] >= 0
        if np.any(valid):
            yield start, features[start:end], labels[start:end], valid


def train(cache_dir: str | Path, holdout_names: set[str], output: str | Path,
          epochs: int = 12, max_len: int = 128, log=print) -> dict:
    import torch
    import torch.nn.functional as F

    from models.temporal_transformer import build_window_temporal_transformer
    torch.manual_seed(41)
    np.random.seed(41)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    videos = load_cache(cache_dir)
    available_names = {v.name for v in videos}
    missing_holdout = holdout_names - available_names
    if missing_holdout:
        raise ValueError("Holdout video(s) missing from cache: " + ", ".join(sorted(missing_holdout)))
    dev = [v for v in videos if v.name not in holdout_names]
    holdout = [v for v in videos if v.name in holdout_names]
    if len(dev) < 2 or not holdout:
        raise ValueError("Provide at least two development videos and one separate --holdout video.")
    if len({int(y) for v in dev for y in v.labels if y >= 0}) < 2:
        raise ValueError("Development videos need both violent and non-violent labeled windows.")
    if any(v.feats.shape[1] + 1 != 2050 for v in videos):
        raise ValueError("Cache feature width does not match the expected X3D-M 2049 features.")
    if any(v.window_s != dev[0].window_s or v.stride_s != dev[0].stride_s for v in videos):
        raise ValueError("All cached videos must use identical window and stride settings.")

    if epochs < 1:
        raise ValueError("epochs must be at least one")
    max_len = max(8, int(max_len))
    train_chunks = []
    for video in dev:
        seq = np.concatenate([video.feats.astype(np.float32), video.raw[:, None].astype(np.float32)], axis=1)
        train_chunks.extend((video.name, *chunk) for chunk in
                            _chunks(seq, video.labels, max_len, max(1, max_len * 3 // 4)))
    if not train_chunks:
        raise ValueError("No labeled sequence chunks were found in the development cache.")

    model_cfg = {"input_dim": 2050, "d_model": 128, "n_layers": 2, "n_heads": 4, "max_len": max_len}
    model = build_window_temporal_transformer(**model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    pos = sum(int((y[valid] == 1).sum()) for _, _, _, y, valid in train_chunks if valid.any())
    neg = sum(int((y[valid] == 0).sum()) for _, _, _, y, valid in train_chunks if valid.any())
    pos_weight = torch.tensor([min(20.0, max(1.0, neg / max(pos, 1)))], dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_chunks)))
        random.Random(1000 + epoch).shuffle(order)
        losses = []
        for idx in order:
            _, _, x, y, valid = train_chunks[idx]
            xt = torch.from_numpy(x).unsqueeze(0)
            yt = torch.from_numpy(np.maximum(y, 0).astype(np.float32)).unsqueeze(0)
            mask = torch.from_numpy(~valid).unsqueeze(0)
            logits = model(xt, padding_mask=mask)
            loss_values = F.binary_cross_entropy_with_logits(
                logits, yt, reduction="none", pos_weight=pos_weight
            )
            loss = loss_values[~mask].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        log(f"epoch {epoch + 1}/{epochs}: development loss={np.mean(losses):.4f}")

    model.eval()
    all_y, all_p = [], []
    with torch.inference_mode():
        for video in holdout:
            seq = np.concatenate([video.feats.astype(np.float32), video.raw[:, None].astype(np.float32)], axis=1)
            preds = np.zeros(len(seq), dtype=np.float32)
            counts = np.zeros(len(seq), dtype=np.float32)
            for start, x, _, _ in _chunks(seq, video.labels, max_len, max(1, max_len * 3 // 4)):
                p = torch.sigmoid(model(torch.from_numpy(x).unsqueeze(0)))[0].numpy()
                preds[start:start + len(x)] += p
                counts[start:start + len(x)] += 1
            probs = preds / np.maximum(counts, 1)
            known = video.labels >= 0
            all_y.extend(video.labels[known].astype(int).tolist())
            all_p.extend(probs[known].tolist())

    y = np.asarray(all_y, dtype=int)
    p = np.asarray(all_p, dtype=float)
    pred = p >= 0.5
    tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum()); tn = int(((y == 0) & ~pred).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    metrics = {"holdout_videos": sorted(holdout_names), "threshold": 0.5,
               "window_precision": precision, "window_recall": recall,
               "window_f1": 2 * precision * recall / max(precision + recall, 1e-12),
               "window_false_positive_rate": fp / max(fp + tn, 1),
               "tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "n_dev_videos": len(dev), "n_holdout_videos": len(holdout)}
    log("holdout (window-level, fixed 0.5 threshold): " + json.dumps(metrics, sort_keys=True))

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": model_cfg, "state_dict": model.state_dict(),
                "metadata": {"metrics": metrics, "trained_on_videos": sorted(v.name for v in dev),
                             "window_seconds": dev[0].window_s, "stride_seconds": dev[0].stride_s,
                             "architecture": "X3D feature + score sequence Transformer",
                             "trained_at": __import__("datetime").datetime.now().isoformat(timespec="seconds")}}, out)
    log(f"saved trained temporal checkpoint: {out}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="training_cache")
    ap.add_argument("--holdout", required=True, help="Comma-separated cached video filenames kept out of training")
    ap.add_argument("--output", default="checkpoints/temporal_transformer.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=128)
    a = ap.parse_args()
    hold = {s.strip() for s in a.holdout.split(",") if s.strip()}
    train(a.cache, hold, a.output, a.epochs, a.max_len)


if __name__ == "__main__":
    main()
