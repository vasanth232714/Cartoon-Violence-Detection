# Multimodal Cartoon Violence Review Pipeline

Streamlit application that analyzes an uploaded or linked video end to end. It combines overlapping
X3D-M video windows, an optional trained temporal Transformer, multilingual speech transcription,
verbal-language candidates, AudioSet sound candidates, and optional Qwen2.5-VL explanations for
selected event clips. Reports, transcripts, evidence frames, explanations, and human reviews are stored
in SQLite.

> **Read first (honest status).** The bundled X3D-M checkpoint was trained on real-world fighting footage,
> not cartoons. A temporal Transformer architecture and training script are integrated, but this project
> does not include cartoon-labeled data or a trained temporal checkpoint. Until one is trained, inference
> uses X3D window scores with the selected deterministic post-processing. Qwen explanations describe
> sampled event clips; they are not detector labels and can be wrong. Audio and language outputs are
> independent candidates. Severity remains an unvalidated rule-based estimate. See `training/README.md`.

## Train the temporal Transformer
Label cartoon windows using `training/README.md`, build the X3D feature cache, then run
`python -m training.train_temporal_transformer --cache training_cache --holdout ep09.mp4,ep10.mp4`.
Keep entire videos held out. The app loads `checkpoints/temporal_transformer.pt` automatically when present.
The holdout metrics measure window classification only; they are not a claim of production accuracy.

## What was verified, and what was not

| Item | State |
|---|---|
| Model facts (X3D-M, 16 frames, thr 0.4, label order, MIT, RWF-2000) | Read from the Hugging Face model card |
| Preprocessing, windows, events, evidence, severity, SQLite, metrics, PDF, real X3D load/inference, temporal sequence shape | **95 tests passed** in the project `.venv` |
| Pipeline integration tests | Use a **fake brightness classifier fixture** (`tests/helpers.py`) - proves plumbing only |
| Loading the real X3D checkpoint + real inference | **Verified** against the checkpoint in `checkpoints/final/`; cartoon accuracy remains unvalidated |
| Live ASR, toxicity, AudioSet AST, and Qwen generation | **Not run**; Faster-Whisper is absent and Qwen/audio model weights were not downloaded |
| Streamlit UI | Syntax-checked; the app was not launched during this change |
| Pinned dependency versions (`requirements.txt`) | Not verified in a clean environment |
| CPU speed | **Not measured.** The app measures and displays it per run |

Checkpoint key compatibility is a known risk: the model card says the authors' wrapper classes may be needed.
The loader uses `strict=True` and fails loudly (printing sample keys) rather than running with random weights.

## Pipeline
* X3D-M scores overlapping 16-frame windows across the full video, including the tail.
* A trained sequence Transformer can refine the X3D window scores when a checkpoint exists.
* Faster-Whisper transcribes the full audio track. English lexicons flag specific phrases; a multilingual toxicity
  model provides a broad toxicity candidate for English, French, Spanish, Italian, Portuguese, Turkish, and Russian.
  Whisper transcription supports more languages than the toxicity classifier; abuse detection is not equally validated
  across languages.
* AudioSet AST independently identifies gunfire-like, scream-like, shouting, explosion, crash, and impact sound candidates.
* Qwen2.5-VL-3B-Instruct runs as a separate Phase 2 action from the Report, after Phase 1 has saved all visual,
  speech, and sound candidates. It can explain flagged clips or independently review the full source in 30-second
  chunks to surface possible X3D misses. This is a pretrained second opinion, not a validated cartoon violence model;
  it never replaces or confirms X3D events. The first use may download a large model and can be slow on CPU. It uses
  the checkpoint's bfloat16 weights, reduced frame resolution, and is guarded at about 8 GiB of available RAM; if RAM is low, Phase 1 results
  remain available and Qwen can be retried later.
* The report keeps visual, speech, and sound evidence separate, shows original timestamps and the transcript, and stores
  Qwen text under each event. Human review does not overwrite model output.

## Architecture
```
Video -> X3D windows -> trained sequence Transformer (if available) -> event candidates
      -> full-video ASR + toxicity candidates + AudioSet sound candidates
      -> event evidence -> SQLite -> report/review -> optional Qwen candidate explanations + full-video second opinion
```
Folders follow the requested structure (`models/`, `preprocessing/`, `pipeline/`, `database/`, `ui/`, `evaluation/`,
`tests/`, `research/`). Extras: `ui/common.py`, `ui/history_page.py`, `ui/research_page.py`, `pipeline/report_pdf.py`,
`tests/helpers.py`. Client's FastAPI+React stack was replaced by Streamlit per your instruction (flagged in the traceability file).

## Windows setup (PowerShell)
Use 64-bit **Python 3.10 or 3.11**.
```powershell
cd cartoon-violence-detection
py -3.11 -m venv .venv
# if activation is blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If `pytorchvideo==0.1.5` fails to install: `pip install git+https://github.com/facebookresearch/pytorchvideo.git`
(requires Git). PyPI Windows wheels of `torch` are CPU builds; no CUDA is needed.

### Model checkpoint (MIT license, source: huggingface.co/visionlab-ai/school-violence-detection-models)
Either press **Download checkpoint** in the app (explicit network call), or:
```powershell
python -c "from models.model_loader import download_checkpoint; print(download_checkpoint('x3d_m_realtime'))"
```
It is saved to `checkpoints/final/final_x3d_realtime.pt` (git-ignored). If `torch.load(weights_only=True)` refuses the
file and you trust it: `$env:CVD_TRUST_CHECKPOINT = "1"`.

### Run
```powershell
streamlit run app.py
```
### Tests
```powershell
python -m unittest discover -s tests -t . -v
```
Real-model tests run automatically once torch, pytorchvideo and the checkpoint exist.

## Using the app
1. **Upload & Analyze** - upload a video, check metadata, adjust settings if needed, press **Analyze video**.
   Phase 1 scans the full video and audio, then saves candidates and evidence without loading Qwen. Identical re-analysis is detected; re-running keeps the old run.
2. **Report** - review the complete candidate table and timeline. When ready, use **Explain all remaining candidate clips with Qwen** to run Phase 2. Pick an event (or Previous/Next) for evidence frames, video player, explanation, and review form.
3. **History** - reopen any stored result (no inference rerun); you can delete the original video file while keeping results.
4. **Dashboard** - counts and distributions computed from the database (empty state when no data).

## Data locations
SQLite `data/database/app.db` · uploads `data/uploads/` · evidence `data/evidence/run_<id>/` · checkpoints `checkpoints/`.
Override with `CVD_DATA_DIR`, `CVD_DB_PATH`, `CVD_CHECKPOINT_DIR` (see `.env.example`).

## CPU expectations
Each window is one X3D-M forward pass; a 5-10 minute video with defaults produces roughly 150-300 windows. Total time can exceed
the video length; the Report shows measured `inference_mean_ms_per_window`, total time and time/video-length ratio.
No real-time claim is made.

## Troubleshooting
* *Checkpoint keys do not match* - the checkpoint needs the authors' wrapper; get `model_registry.py` from their source repo.
* *`torch`/`pytorchvideo` not importable* - reinstall in the venv; confirm Python 3.10/3.11.
* *AVI/MKV does not play in the browser* - use the frame viewer; analysis is unaffected (OpenCV decodes it).
* *Run shows `interrupted`* - the page was refreshed during analysis. Start again; the partial run is kept for audit.
* *Corrupted video* - rejected during validation with a message.

## Evaluation
```powershell
python -m evaluation.evaluation_runner --pair 3:annotations\ep1.json --pair 4:annotations\ep2.json --out results.json
```
Annotation format and protocol: `research/evaluation_plan.md`. Do not report a single video as test accuracy.

## Accuracy and language limits
Thresholds only change candidate sensitivity. They do not make the real-world X3D weights cartoon-accurate.
English phrase lists do not generalize to other languages. The multilingual toxicity classifier is broad and may flag
profanity without context; it does not determine threats. Sound labels describe audio and do not prove a visual action.
Use the per-event review form and the held-out data protocol before making accuracy claims.
