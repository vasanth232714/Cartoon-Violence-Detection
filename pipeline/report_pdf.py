"""Optional PDF export (secondary to the interactive Streamlit report). Built only from stored results."""
from __future__ import annotations

import io
import json
from pathlib import Path

from database.repository import Repository
from preprocessing.timestamp_utils import format_timestamp as ft


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_pdf(repo: Repository, run_id: int) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("PDF export requires the reportlab dependency from requirements.txt") from exc
    run = repo.get_run(run_id)
    video = repo.get_video(run["video_id"])
    events = repo.list_events(run_id)
    audio_segments = repo.list_audio_segments(run_id)
    cfg = json.loads(run["config_json"])
    info = json.loads(run["model_info_json"])
    ss = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                            bottomMargin=2 * cm, title=f"Analysis report - {video['original_name']}")
    S = []
    S.append(Paragraph("Cartoon Violence Candidate-Event Report", ss["Title"]))
    S.append(Paragraph(_esc(f"Video: {video['original_name']} | duration {ft(video['duration_s'])} | "
                            f"{video['width']}x{video['height']} @ {video['fps']:.2f} fps"), ss["Normal"]))
    S.append(Paragraph(_esc(f"Run #{run_id} | status: {run['status']} | processing time "
                            f"{(run['elapsed_s'] or 0):.1f}s | app v{run['app_version']}"), ss["Normal"]))
    S.append(Paragraph(_esc(f"Model: {info.get('display_name')} (trained on {info.get('training_domain')}). "
                            f"Threshold {cfg['threshold']}, window {cfg['window_seconds']}s, stride "
                            f"{cfg['stride_seconds']}s, merge gap {cfg['merge_gap_seconds']}s."), ss["Normal"]))
    S.append(Spacer(1, 8))
    S.append(Paragraph(
        "<b>Limitations:</b> X3D is a binary classifier trained on real-world footage and is not validated on "
        "cartoons. Its score is not calibrated cartoon-violence confidence. Sound, language, and Qwen outputs are "
        "candidates or second opinions. Review priority is an unvalidated duration/motion heuristic, not violence "
        "severity. Human review is required.", ss["Normal"]))
    if audio_segments:
        S.append(Spacer(1, 10))
        S.append(Paragraph("Full-video transcript and audio candidates", ss["Heading2"]))
        for segment in audio_segments:
            if segment["kind"] == "transcript":
                text = (f"{ft(segment['start_s'])}–{ft(segment['end_s'])} "
                        f"[{segment['language'] or 'language unknown'}]: {segment['text'] or ''}")
                S.append(Paragraph(_esc(text), ss["BodyText"]))
            elif segment["kind"] in ("verbal_abuse", "audio_event"):
                text = (f"{ft(segment['start_s'])}–{ft(segment['end_s'])} "
                        f"{segment['category'] or segment['kind']}: {segment['text'] or ''}")
                S.append(Paragraph(_esc(text), ss["BodyText"]))
    S.append(Spacer(1, 10))
    if run["status"] not in ("completed", "completed_with_warnings"):
        S.append(Paragraph(_esc(f"WARNING: run status is '{run['status']}' - results may be incomplete."), ss["Heading3"]))
    if not events:
        S.append(Paragraph("No event candidate crossed the configured threshold.", ss["Normal"]))
    else:
        rows = [["#", "Start", "End", "Dur (s)", "X3D score", "Review priority", "Review"]]
        for e in events:
            rv = repo.latest_review(e["id"])
            rows.append([e["event_no"], ft(e["start_s"]), ft(e["end_s"]), f"{e['duration_s']:.1f}",
                         f"{e['peak_prob']:.2f}", e["severity_label"], rv["decision"] if rv else "unreviewed"])
        t = Table(rows, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        S.append(t)
        for e in events:
            S.append(Spacer(1, 12))
            S.append(Paragraph(f"Event {e['event_no']}: {ft(e['start_s'])} - {ft(e['end_s'])}", ss["Heading3"]))
            S.append(Paragraph(_esc(f"Finding: {e['label']} | uncalibrated X3D class score {e['peak_prob']:.3f} | heuristic review priority {e['severity_label']} (not violence severity)"), ss["BodyText"]))
            for para in (e["explanation"] or "").split("\n\n"):
                S.append(Paragraph(_esc(para).replace("**", ""), ss["BodyText"]))
            if e["qwen_explanation"]:
                S.append(Paragraph("Qwen2.5-VL clip explanation", ss["Heading4"]))
                for para in e["qwen_explanation"].split("\n\n"):
                    S.append(Paragraph(_esc(para), ss["BodyText"]))
            details = json.loads(e["severity_json"] or "{}")
            for span in details.get("verbal_abuse", []):
                transcript = span.get("censored_transcript") or span.get("transcript") or ""
                info = (f"Speech candidate {ft(span.get('start_s', 0))}–{ft(span.get('end_s', 0))} "
                        f"[{span.get('language') or 'unknown'}], {', '.join(span.get('categories', []))}: {transcript}")
                S.append(Paragraph(_esc(info), ss["BodyText"]))
            for sound in details.get("audio_events", []):
                info = (f"Sound candidate {ft(sound.get('start_s', 0))}–{ft(sound.get('end_s', 0))}: "
                        f"{sound.get('category', 'unknown')} ({float(sound.get('confidence', 0)):.3f})")
                S.append(Paragraph(_esc(info), ss["BodyText"]))
            imgs = []
            for f in repo.list_evidence(e["id"])[:3]:
                if Path(f["path"]).is_file():
                    imgs.append(Image(f["path"], width=5 * cm, height=3.4 * cm, kind="proportional"))
            if imgs:
                S.append(Table([imgs]))
                S.append(Paragraph("Evidence at " + ", ".join(ft(f["timestamp_s"]) for f in
                                                              repo.list_evidence(e["id"])[:3]), ss["Italic"]))
    audit = info.get("qwen_full_video_audit") or {}
    if audit:
        S.append(Spacer(1, 10))
        S.append(Paragraph("Independent full-video Qwen review (unconfirmed)", ss["Heading2"]))
        S.append(Paragraph(
            _esc(f"Status: {audit.get('status', 'unknown')} · "
                 f"segments: {audit.get('completed_chunks', 0)}/{audit.get('total_chunks', 0)}. "
                 "Qwen is not validated for cartoon violence; its output is a second opinion for human review."),
            ss["BodyText"]))
        for chunk in audit.get("chunks", []) or []:
            if chunk.get("status") != "completed":
                continue
            assessment = str(chunk.get("assessment", "uncertain")).replace("_", " ")
            S.append(Paragraph(
                _esc(f"{ft(chunk.get('start_s', 0))}–{ft(chunk.get('end_s', 0))}: {assessment}. "
                     f"{chunk.get('evidence') or 'No evidence text returned.'}"), ss["BodyText"]))
        for candidate in audit.get("events", []) or []:
            S.append(Paragraph(
                _esc(f"Qwen visual candidate {ft(candidate.get('start_s', 0))}–{ft(candidate.get('end_s', 0))}: "
                     f"{candidate.get('category', 'unclear')} — {candidate.get('observed_action', '')}"),
                ss["BodyText"]))
    doc.build(S)
    return buf.getvalue()
