# Methodology (what the code does today)

1. **Windows.** Duration/FPS from OpenCV. Windows of `window_seconds` (default 4) every `stride_seconds` (default 2); the last
   window is anchored to the end of the video so the tail is always analysed. 16 frames are sampled uniformly per window.
2. **Preprocessing.** Frames are resized directly to 224x224 (no crop), RGB, scaled to [0,1], normalised mean 0.45 / std 0.225
   (model card). *Assumption:* the model card states the sizes/normalisation but not the resize policy (direct resize vs. short-side
   resize + crop). This is unverified and may matter.
3. **Model.** pytorchvideo `x3d_m`, `blocks[5].proj = Linear(2048, 2)`, weights `final/final_x3d_realtime.pt`. Output index 1 = violent
   (model card). Score = softmax probability, uncalibrated.
4. **Events.** Window positive if p >= threshold (default 0.4, the value documented for this checkpoint on RWF-2000 validation data).
   Positive windows that overlap or lie within `merge_gap_seconds` are merged. Boundaries are candidate boundaries at window resolution.
   `n_positive_runs` = number of separate runs of positive windows inside an event (repetition proxy).
5. **Evidence.** Up to N frames evenly spaced in the event plus the peak-window centre frame, read from the original video.
6. **Motion.** Mean absolute grayscale difference (160x90) between frames sampled ~6/s; differences > 45 are treated as shot cuts and excluded.
7. **Severity rubric v1 (0-3, client scale).** Points: duration (<4 s:0, 4-<10:1, >=10:2) + repetition (1 run:0, 2:1, >=3:2) +
   motion (<8:0, 8-20:1, >20:2). Total 0-1 -> Mild(1), 2-3 -> Moderate(2), >=4 -> High(3). Model probability is **not** used.
   Thresholds are placeholders to be calibrated against human annotation. Output states its method; human severity is stored separately.
8. **Explanation.** Template from model output, timestamps, measured motion and available frames. No generated scene narrative.

## Unverified / open items
* Whether `final_x3d_realtime.pt` loads strictly into the documented architecture (loader fails loudly if not).
* Training-time resize policy and frame-sampling stride of the checkpoint.
* Behaviour on cartoon content - the central research question; requires labelled cartoon data.
* TimeSformer (facebookresearch/TimeSformer) and ActionFormer: **not inspected**; requirements and CPU feasibility unknown.
* The other R(2+1)D-18 checkpoint (`final_best.pt`, 32 frames, thr 0.5) is listed on the card but **not integrated**: its wrapper
  architecture is not documented in the card.
