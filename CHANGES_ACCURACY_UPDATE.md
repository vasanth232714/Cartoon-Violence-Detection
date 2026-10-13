# Accuracy update - what changed and why

## Integrity fixes (these affected results and your report)
1. `models/transformer_refiner.py`: removed the randomly-initialised, untrained Transformer that contributed 20% of every
   refined score (results were not reproducible, and it silently vanished when torch was missing). Replaced by a
   deterministic numpy temporal-attention smoother. It is NOT a trained Transformer and the UI/report now say so.
2. `pipeline/comparison_pipeline.py`, `ui/comparison_page.py`, `generate_project_report_pdf.py`: removed the "projected"
   accuracies (94.5% / 95.2%), error rates, R² "benchmarks" and the claim that multimodal fusion cut error from 7.0% to 4.8%.
   None were measured. Unmeasured cells now read "not measured". **Re-generate FINAL_YEAR_PROJECT_REPORT.pdf**, and also
   review the generator's other "Transformer" chapters (15 mentions) so they match what is implemented.
3. `database/` was missing from the zip you sent; restored, with `delete_run` / `clear_all_history` that your UI calls.
4. Bug: `calibrate_window_probabilities` crashed on numpy arrays of shot-cut flags.

## Accuracy features
* `pipeline/postprocess.py` - one shared post-processing path used by the app and by tuning (so tuned = runtime behaviour).
* Hysteresis event detection (`threshold_low`) in `pipeline/event_segmenter.py`.
* `models/adapter.py` + `models/violence_classifier.py` - cartoon-trained logistic head on frozen X3D-M features; optional flip TTA.
* `training/` - `build_cache`, `fit` (leave-video-out CV, grid search, holdout report), see `training/README.md`.
* `config/settings.py` - new fields (`threshold_low`, `use_adapter`, `use_tta`, ...), `AnalysisConfig.from_tuned()`.
* `ui/upload_page.py` - "Tuned on your labelled cartoons" preset, hysteresis / adapter / TTA controls.

## Verification
Run in a sandbox without torch/network: 78 tests, 72 pass, 5 skipped (need torch + checkpoint), 1 error (`yt_dlp` not installed
there). NOT verified here: the feature-capture hook and adapter path on the real X3D-M (needs torch), the Streamlit UI, and
real accuracy (needs your labelled data).
