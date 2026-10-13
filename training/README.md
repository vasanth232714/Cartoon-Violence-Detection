# Making the detector accurate on cartoons (measured, not assumed)

The pretrained X3D-M was trained on real-world fights (RWF-2000). On cartoons its scores are shifted: slapstick, speed
lines, hard cuts and effects look "violent" (false positives) while stylised hits can look calm (false negatives). Hand-picking
a higher threshold trades one error for the other. The fix is to **learn the decision from your own labelled cartoon footage**
and to **measure it on videos the tuning never saw**.

## 1. Label videos
Create `labelled/labels.json` next to the videos (any length; whole-clip labels are fine for short clips):
```json
{"videos": [
  {"file": "ep01.mp4", "events": [{"start": 31.0, "end": 39.0}, {"start": 70.5, "end": 75.0}]},
  {"file": "ep02_calm.mp4", "events": []},
  {"file": "clip_fight.mp4", "label": "violent"},
  {"file": "clip_chat.mp4", "label": "non_violent"}
]}
```
Seconds, to about +/-1 s. Include **hard negatives**: slapstick that is *not* the violence you want flagged, chases, dances,
explosions/effects, fast camera moves, many scene cuts. Cover several shows/art styles. Windows that only touch an event
by < 1 s are treated as ambiguous and ignored.

Rough guide (not a guarantee): a few dozen distinct events and an hour or more of footage across several styles before
numbers are worth trusting; **keep at least 3 videos entirely out as a holdout**.

## 2. Cache model outputs (one CPU pass per video; resumable)
```powershell
python -m training.build_cache --labels labelled\labels.json --root labelled --out training_cache
```
## 3. Learn and evaluate
```powershell
python -m training.fit --cache training_cache --holdout ep09.mp4,ep10.mp4
```
It prints, for the holdout videos: the **raw model**, the **old hand-set preset** and the **tuned system** side by side
(window P/R/F1 with FP/FN counts, event P/R/F1 at tIoU 0.3). It writes
`checkpoints/cartoon_adapter.npz` (only if it beats the raw model by >= 0.01 out-of-fold AUC) and `config/tuned_settings.json`.
Restart Streamlit; the **"Tuned on your labelled cartoons"** preset becomes the default.
Use `--dry-run` to evaluate without writing anything.

## What is learned
* **Adapter**: L2 logistic head on the frozen 2048-d X3D-M feature (+ original logit margin). Cheap on CPU.
* **Post-processing** (threshold, hysteresis low-threshold, merge gap, min event length, smoothing width, cut discount,
  weak-single-window filter): grid-searched on *out-of-fold* scores (leave-video-out) so tuning does not see its own labels.
* **Hysteresis**: events start at the main threshold but extend over adjacent windows above a lower one - the standard way
  to gain recall at event edges without more false alarms.
* **Flip TTA** (optional, ~2x slower): averages clip and mirror image.

## Honest limits
* Numbers are only as good as your labels and the diversity of your videos. Without `--holdout`, tuned numbers are optimistic.
* Boundaries stay window-resolution (4 s / 2 s stride by default). Rebuild the cache with `--stride 1` for finer boundaries.
* This learns a *binary* violent / non-violent boundary. It does not give action categories or true severity.
* The deterministic "refinement" is a fixed smoothing kernel. The sequence Transformer below is a separate learned model.

## Train the integrated sequence Transformer
The app's optional temporal model consumes each cached window's 2049-dimensional X3D feature vector and raw X3D
violence score, then predicts a score for every window using its neighboring sequence context. It requires a
video-level holdout split and cached windows with both classes in the development videos.

```powershell
python -m training.build_cache --labels labelled\labels.json --root labelled --out training_cache
python -m training.train_temporal_transformer --cache training_cache --holdout ep09.mp4,ep10.mp4 --output checkpoints\temporal_transformer.pt
```

This script reports window-level holdout precision, recall, F1, and false-positive rate at a fixed 0.5 threshold.
The holdout videos are excluded from gradient updates. Keep additional videos for a final evaluation if you tune
the model or threshold. The app detects the saved checkpoint at `checkpoints/temporal_transformer.pt`; without it,
the sequence model is not used and the report records that status.
