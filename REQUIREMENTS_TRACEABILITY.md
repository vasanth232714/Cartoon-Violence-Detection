# Requirements Traceability (client document -> implementation)

Status values: Not Started · In Progress · Implemented · Blocked. "Verified" says how far testing actually went.
Dates: written 2026-10-10. Real-model inference and the Streamlit UI have **not** been executed by the author (see README).

| ID | Client requirement (section) | Implementation | Module | Testing | Status | Verified |
|---|---|---|---|---|---|---|
| R01 | Accept MP4/AVI/MOV/MKV video (4) | Extension + readability + size validation | `preprocessing/video_loader.py` | `test_preprocessing` | Implemented | mp4, avi tested; mov/mkv depend on OpenCV codecs, untested |
| R02 | Frame extraction, timestamps preserved (5) | Single-pass decode, idx/fps timestamps | `preprocessing/frame_extractor.py` | `test_preprocessing`, `test_timestamps` | Implemented | Tested on synthetic video; VFR drift possible |
| R03 | Resize, normalise, sample, short clips (5) | 16 uniform frames/window, 224x224, mean .45/std .225 | `clip_sampler.py` | `test_preprocessing` | Implemented | Yes |
| R04 | Adaptive/key-frame sampling, shot segmentation (4,5) | Not done; uniform sampling only (shot cuts only used to exclude motion outliers) | - | - | Not Started | - |
| R05 | CNN visual feature extraction (6) | Baseline: X3D-M (3D CNN). Per-frame ResNet-18 extractor inside the untrained CNN+Transformer/LSTM | `models/model_loader.py`, `temporal_transformer.py` | `test_real_model` (skipped here) | In Progress | Real checkpoint load not run |
| R06 | Transformer temporal modelling (7,8) | Architecture defined, untrained, refused at load time | `models/temporal_transformer.py` | forward-shape test (skipped here) | Blocked | Needs cartoon-labelled data + training |
| R07 | Violence detection non-violent/violent (9) | X3D-M violent-class probability + threshold 0.4 (model card) | `violence_classifier.py`, `event_segmenter.py` | `test_event_segmentation` (fake probs) | In Progress | Real inference not run |
| R08 | 4-class taxonomy (non-violent/aggressive/physical/severe) (9) | Binary baseline cannot output it | - | - | Blocked | Needs annotated data |
| R09 | Severity 0-3 (10) | Rule-based rubric v1 (duration, repetition, motion); labelled unvalidated | `severity_assessor.py` | `test_event_segmentation` | In Progress | Heuristic only; client factors "characters, dangerous objects, action type" not measurable |
| R10 | Duration & frequency of violent events (1.12) | Event durations; counts per run | `event_segmenter.py`, report | `test_event_segmentation` | Implemented | Yes (window-resolution boundaries) |
| R11 | Event segmentation with start/end, probability, confidence, frames (14) | Merge positive windows; configurable thr/gap | `event_segmenter.py` | tests | Implemented | Yes |
| R12 | Evidence + important frames with timestamps (12,18) | Frames from original video + peak-window frame | `evidence_selector.py` | `test_pipeline` | Implemented | Yes |
| R13 | Natural-language scene descriptions (11) | None; no captioning model integrated (template explanation only, disclosed) | `explanation_generator.py` | - | Blocked | Needs verified video-language model |
| R14 | Character/object/action/interaction detection (1.4) | Not available | - | - | Blocked | - |
| R15 | Explainable AI layer (15) | Template explanation separating prediction / evidence / uncertainty / severity basis. No saliency/attention XAI | `explanation_generator.py` | `test_pipeline` | In Progress | Not genuine model-internals XAI |
| R16 | Chronological timeline (13) | Plotly probability timeline + event spans + table | `ui/timeline.py`, `report_page.py` | none automated | Implemented | UI not launched |
| R17 | Dashboard stats (16) | DB-driven metrics, distributions, history | `ui/dashboard.py` | DB stats test | Implemented | UI not launched |
| R18 | Video analysis page, activity summary (17) | Report page; 3-way normal/aggressive/violent split unavailable (binary), shows % inside candidate intervals | `report_page.py` | - | In Progress | UI not launched |
| R19 | Event-level investigation (19) | Detail view, evidence, frame viewer, review actions | `ui/event_investigation.py` | `test_database` (reviews) | Implemented | UI not launched |
| R20 | PDF report (20) | Secondary export built from stored results (master prompt: not the main output) | `pipeline/report_pdf.py` | `test_report_and_eval` | Implemented | PDF bytes validated, not visually inspected |
| R21 | Backend FastAPI + React frontend (21) | **Deviation:** Streamlit per your instruction; no REST API | `app.py` | - | Not Started | Needs your approval or confirmation |
| R22 | SQLite storage (21) | Videos, runs, windows, events, evidence, reviews, config/model metadata | `database/` | `test_database` | Implemented | Yes |
| R23 | Datasets / sample dataset / sample analysis (23,31) | No cartoon dataset obtained; no sample video shipped | `research/dataset_notes.md` | - | Blocked | Licensing + annotation required |
| R24 | Detection metrics: Acc/P/R/F1/ROC-AUC (24) | Implemented as pure functions + runner | `evaluation/` | `test_metrics`, `test_report_and_eval` | Implemented | Verified on synthetic arrays only; no real results exist |
| R25 | Temporal metrics: tIoU, segment P/R (24) | Implemented | `evaluation/metrics.py` | tests | Implemented | Same |
| R26 | Severity metrics: Acc, macro-F1, confusion, MAE (24) | Implemented (on matched events) | `evaluation_runner.py` | partial | Implemented | No severity ground truth exists |
| R27 | Description metrics BLEU/ROUGE/CIDEr/BERTScore (24) | Not implemented (no captions, no references) | - | - | Not Started | - |
| R28 | Ablation CNN / CNN+LSTM / CNN+Transformer / +severity (25) | Architectures E2/E3 defined; no experiments run | `temporal_transformer.py`, Research page | - | Blocked | No data/training |
| R29 | Upload, analyse, view real actions; errors; loading states (31) | Progress bar tied to real work; error handling; interrupted/failed/incomplete statuses | `ui/upload_page.py`, `inference_pipeline.py` | `test_pipeline` | Implemented | UI not launched |
| R30 | Software-only, no hardware (33) | Yes | - | - | Implemented | - |
