# Optional VideoMAE comparison

## Model-card facts checked

Model: [HappyGook/videomae-violence-detector](https://huggingface.co/HappyGook/videomae-violence-detector), checked 2026-10-14.

- The card describes a binary violent / non-violent classifier, fine-tuned from VideoMAE base.
- Its quick-start samples 16 uniformly spaced RGB frames and maps output class index `1` to `Violent` and index `0` to `Non-Violent`.
- The technical section specifies 16 frames at 224×224 pixels.
- The quick-start uses `AutoProcessor.from_pretrained(...)`; the application also uses the checkpoint's own `AutoProcessor`. At load time, the downloaded config is inspected, the label mapping and frame count are validated, and the processor's resize/crop/rescale/normalization settings are recorded in the run metadata.
- The model card lists UCF-Crime and a Bus Violence Dataset as training sources. Its reported held-out result is for the bus dataset (n=280), not cartoons. It provides no cartoon evaluation.
- The model weights file is listed as approximately 345 MB; first opt-in use downloads the model and processor files into the Hugging Face cache.

If the downloaded config does not expose an unambiguous violent and non-violent mapping, or does not expect 16 frames, the comparison is marked failed rather than guessing at the class indices.

## Application behavior

- X3D remains the baseline and keeps producing the run's saved events.
- The optional VideoMAE pass is off by default. When enabled, it uses the exact same sliding-window boundaries and frame indices as X3D, then runs separately over a second bounded-memory decode pass.
- VideoMAE receives RGB frames through its checkpoint `AutoProcessor`; the processor owns its resize/crop, rescale, and normalization. The app validates the expected 16-frame input and records processor settings.
- The optional model runs on CPU, then is released before later report actions. This avoids GPU memory contention with Qwen; the tradeoff is slower inference and additional CPU RAM use.
- Each model retains independent probabilities and thresholds. No score averaging or fusion occurs. Candidate intervals, window agreement, and disagreements are comparisons only, not ground-truth accuracy.
- Qwen remains the existing user-triggered Phase 2 report action.

## Validation performed

- Python compile check: passed for `config`, `database`, `models`, `pipeline`, `ui`, and `tests`.
- A local Transformers 4.49.0 processor smoke check produced `pixel_values` with shape `(1, 16, 3, 224, 224)` from the documented 16 RGB frames.
- Final unit suite: **100 tests passed** (`python -m unittest discover -s tests`, 43.747 seconds).
- New tests cover the documented class mapping, rejection of ambiguous mappings, CPU inference plumbing, separate database persistence/cascade, and optional pipeline integration without changing X3D scores.
- The suite does not download or run the 345 MB VideoMAE checkpoint. Real checkpoint loading/inference and visual UI behavior remain unverified until a user enables the optional model on a machine with network access and sufficient CPU memory.
