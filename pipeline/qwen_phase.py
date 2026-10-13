"""Second-phase Qwen explanations for candidates already saved by a video scan."""
from __future__ import annotations

import gc
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from database.repository import Repository
from models.vlm_explainer import parse_qwen_visual_review
from pipeline.inference_pipeline import _qwen_explain_interval

logger = logging.getLogger(__name__)
ProgressCB = Callable[[float, str], None]


def explain_run_candidates(
    repo: Repository,
    run_id: int,
    video_path: str | Path,
    progress: ProgressCB | None = None,
) -> dict:
    """Explain every not-yet-explained event; detector results remain saved if Qwen fails."""
    progress = progress or (lambda fraction, message: None)
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"The source video is unavailable: {source}")

    events = repo.list_events(run_id)
    selected = [event for event in events
                if event["qwen_status"] != "generated" or not event["qwen_explanation"]
                or not parse_qwen_visual_review(event["qwen_explanation"])["structured"]]
    if not selected:
        return {"generated": 0, "failed": 0, "remaining": 0, "error": None}

    run = repo.get_run(run_id)
    try:
        model_info = json.loads(run["model_info_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        model_info = {}
    model_info["qwen_explanations"] = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "status": "running in report phase",
        "selected_events": len(selected),
        "generated_events": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    repo.update_run_model_info(run_id, model_info)

    runtime: dict = {"model": None, "processor": None, "attempted": False, "error": None}
    generated = 0
    failed = 0
    error_message = None
    try:
        for index, event in enumerate(selected):
            event_no = int(event["event_no"])
            progress(index / len(selected), f"Explaining event {event_no} ({index + 1}/{len(selected)})")
            try:
                try:
                    details = json.loads(event["severity_json"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    details = {}
                label = str(event["label"] or "")
                visual = "Violence (binary model output" in label or bool(details.get("visual_model"))
                audio_findings = details.get("audio_events") or []
                verbal_abuse = details.get("verbal_abuse") or []
                context = {
                    "start_s": event["start_s"],
                    "end_s": event["end_s"],
                    "peak_s": event["peak_time_s"],
                    "peak_prob": event["peak_prob"] if visual else None,
                    "visual_score_available": visual,
                    "severity_label": event["severity_label"],
                }
                explanation = _qwen_explain_interval(
                    str(source), event["start_s"], event["end_s"], context,
                    verbal_abuse, audio_findings, runtime,
                )
                repo.update_event_qwen(event["id"], explanation, "generated")
                generated += 1
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                warning = f"Qwen explanation unavailable for event {event_no}: {error_message}"
                repo.update_event_qwen(event["id"], None, "unavailable")
                repo.append_run_warning(run_id, warning)
                failed += 1
                logger.warning("Qwen explanation failed for event %s: %s", event_no, exc)
                # A failed model load affects every remaining interval. Leave those events pending
                # so the user can retry after freeing RAM or repairing the local model install.
                if runtime.get("error"):
                    break
            progress((index + 1) / len(selected), f"Finished event {event_no} ({index + 1}/{len(selected)})")
    finally:
        runtime["model"] = None
        runtime["processor"] = None
        gc.collect()

    remaining = sum(1 for event in repo.list_events(run_id)
                    if event["qwen_status"] != "generated" or not event["qwen_explanation"])
    model_info["qwen_explanations"] = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "status": "completed" if remaining == 0 else ("partial" if generated else "needs retry"),
        "selected_events": len(selected),
        "generated_events": generated,
        "failed_events": failed,
        "remaining_events": remaining,
        "last_attempt_at": datetime.now(timezone.utc).isoformat(),
        "last_error": error_message,
    }
    repo.update_run_model_info(run_id, model_info)
    return {"generated": generated, "failed": failed, "remaining": remaining, "error": error_message}


def _parse_video_audit_response(text: str, chunk_duration_s: float, offset_s: float) -> dict:
    """Parse a best-effort structured answer and clamp model timestamps to the source chunk."""
    answer = (text or "").strip()
    start = answer.find("{")
    end = answer.rfind("}")
    payload = None
    if start >= 0 and end > start:
        try:
            payload = json.loads(answer[start:end + 1])
        except json.JSONDecodeError:
            payload = None

    valid_assessments = {"violence_observed", "no_violence_observed", "uncertain"}
    assessment = str((payload or {}).get("visual_assessment", "uncertain")).strip().lower()
    if assessment not in valid_assessments:
        assessment = "uncertain"

    clean_events = []
    raw_events = (payload or {}).get("events", [])
    if not isinstance(raw_events, list):
        raw_events = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        try:
            rel_start = float(item.get("start_offset_s"))
            rel_end = float(item.get("end_offset_s"))
        except (TypeError, ValueError):
            continue
        rel_start = min(max(rel_start, 0.0), chunk_duration_s)
        rel_end = min(max(rel_end, rel_start), chunk_duration_s)
        if rel_end <= rel_start:
            continue
        clean_events.append({
            "start_s": round(offset_s + rel_start, 2),
            "end_s": round(offset_s + rel_end, 2),
            "category": str(item.get("category") or "unclear")[:120],
            "observed_action": str(item.get("observed_action") or "")[:500],
        })

    return {
        "assessment": assessment,
        "evidence": str((payload or {}).get("evidence") or "")[:1000],
        "events": clean_events,
        "raw_response": answer[:1600],
        "structured": payload is not None,
    }


def audit_full_video_with_qwen(
    repo: Repository,
    run_id: int,
    video_path: str | Path,
    duration_s: float,
    progress: ProgressCB | None = None,
    chunk_seconds: float = 30.0,
) -> dict:
    """Ask pretrained Qwen to independently review the full video in short chunks.

    These are separate, unconfirmed VLM observations. They do not overwrite X3D scores or
    automatically accept/reject visual candidates.
    """
    from models.vlm_explainer import (
        QWEN_MIN_AVAILABLE_RAM_BYTES,
        _available_physical_memory_bytes,
        explain_clip_qwen,
        extract_subclip,
        load_qwen_vl,
    )

    progress = progress or (lambda fraction, message: None)
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"The source video is unavailable: {source}")
    if duration_s <= 0:
        raise ValueError("A positive video duration is required for the Qwen audit.")

    available = _available_physical_memory_bytes()
    if available is not None and available < QWEN_MIN_AVAILABLE_RAM_BYTES:
        free_gib = available / 1024**3
        required_gib = QWEN_MIN_AVAILABLE_RAM_BYTES / 1024**3
        raise MemoryError(
            f"Qwen full-video review needs about {required_gib:.0f} GiB of available RAM; "
            f"only {free_gib:.1f} GiB is available. X3D candidates remain unchanged."
        )

    run = repo.get_run(run_id)
    try:
        model_info = json.loads(run["model_info_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        model_info = {}

    chunks = []
    cursor = 0.0
    while cursor < duration_s:
        end_s = min(cursor + chunk_seconds, duration_s)
        chunks.append((cursor, end_s))
        cursor = end_s

    existing = model_info.get("qwen_full_video_audit") or {}
    audit_chunks = list(existing.get("chunks", [])) if existing.get("chunks") else []
    # Resume a partial audit without repeating completed source intervals.
    completed_by_start = {round(float(row.get("start_s", -1)), 2): row for row in audit_chunks
                          if row.get("status") == "completed"}
    model_info["qwen_full_video_audit"] = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "status": "running",
        "sampling_fps": 1.0,
        "chunk_seconds": chunk_seconds,
        "total_chunks": len(chunks),
        "completed_chunks": len(completed_by_start),
        "chunks": audit_chunks,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "limitations": "Pretrained general VLM second opinion; not validated for cartoon violence and not a human confirmation.",
    }
    repo.update_run_model_info(run_id, model_info)

    runtime: dict = {"model": None, "processor": None}
    error_message = None
    try:
        progress(0.0, "Loading Qwen's pretrained bfloat16 weights and video processor; the first load can take several minutes")
        logger.info("Loading pretrained Qwen bfloat16 weights for full-video review")
        runtime["model"], runtime["processor"] = load_qwen_vl()
        logger.info("Qwen model is loaded; scanning %d video segments", len(chunks))
        for index, (start_s, end_s) in enumerate(chunks):
            key = round(start_s, 2)
            if key in completed_by_start:
                progress((index + 1) / len(chunks), f"Qwen audit already complete for {start_s:.0f}–{end_s:.0f}s")
                continue

            progress(index / len(chunks), f"Qwen reviewing full-video segment {index + 1}/{len(chunks)} ({start_s:.0f}–{end_s:.0f}s)")
            sub_path = extract_subclip(source, start_s, end_s, padding_s=0.0)
            try:
                prompt = f"""Review only the visible actions in this animated or live-action video segment.
This segment covers original video time {start_s:.2f} to {end_s:.2f} seconds. The sampled frames are spaced
at about 1 frame per second. Do not infer an action from sound, music, color, motion alone, or the X3D detector.
Rain, umbrellas, ordinary play, running, dancing, and harmless falls are not violence by themselves. A weapon or
war setting alone is not an observed violent act; report visible shooting, striking, injury, or destructive acts
only when the frames support it. Cartoons may use fantasy effects; state uncertainty when sampled frames do not
show enough evidence. Do not invent details. Timestamps must be seconds relative to the start of THIS segment.

Return only JSON with this shape:
{{"visual_assessment":"violence_observed|no_violence_observed|uncertain",
 "evidence":"brief visible evidence or why uncertain",
 "events":[{{"start_offset_s":0.0,"end_offset_s":2.0,"category":"shooting|fight|strike|injury|destruction|other",
 "observed_action":"what is visibly happening"}}]}}
Use an empty events array if no violent act is visible. Keep intervals as narrow as the sampled frames support."""
                response = explain_clip_qwen(
                    sub_path, fps=1.0, max_tokens=320, prompt=prompt,
                    model=runtime["model"], processor=runtime["processor"],
                )
                parsed = _parse_video_audit_response(response, end_s - start_s, start_s)
                row = {"start_s": round(start_s, 2), "end_s": round(end_s, 2),
                       "status": "completed", **parsed}
            finally:
                if Path(sub_path).is_file():
                    Path(sub_path).unlink(missing_ok=True)

            audit_chunks.append(row)
            completed_by_start[key] = row
            audit = model_info["qwen_full_video_audit"]
            audit["chunks"] = audit_chunks
            audit["completed_chunks"] = len(completed_by_start)
            audit["updated_at"] = datetime.now(timezone.utc).isoformat()
            repo.update_run_model_info(run_id, model_info)
            progress((index + 1) / len(chunks), f"Qwen reviewed {index + 1}/{len(chunks)} video segments")
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        logger.warning("Qwen full-video review failed for run %s: %s", run_id, error_message)
    finally:
        runtime["model"] = None
        runtime["processor"] = None
        gc.collect()

    audit = model_info["qwen_full_video_audit"]
    audit["status"] = "completed" if len(completed_by_start) == len(chunks) else ("partial" if completed_by_start else "failed")
    audit["completed_chunks"] = len(completed_by_start)
    audit["last_attempt_at"] = datetime.now(timezone.utc).isoformat()
    audit["last_error"] = error_message
    audit["events"] = [item for chunk in audit_chunks if chunk.get("status") == "completed"
                       for item in chunk.get("events", [])]
    audit["event_count"] = len(audit["events"])
    repo.update_run_model_info(run_id, model_info)
    return {"status": audit["status"], "completed_chunks": len(completed_by_start),
            "total_chunks": len(chunks), "event_count": audit["event_count"], "error": error_message}
