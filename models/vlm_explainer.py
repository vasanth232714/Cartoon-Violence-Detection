"""Qwen2.5-VL visual review of saved candidate clips.

Qwen provides a structured visible-violence assessment and explanation after the X3D scan.
Its output is an unvalidated model second opinion, not human confirmation.
"""
from __future__ import annotations

import logging
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_MIN_AVAILABLE_RAM_BYTES = 8 * 1024**3

DEFAULT_VLM_PROMPT = """
Analyze this CARTOON/ANIMATED clip only from visible evidence. Animation may contain fantasy,
comedy, exaggerated motion, and unrealistic effects. Do not invent events.

Return:
1. SHORT SUMMARY
2. OBSERVED ACTIONS
3. POSSIBLE CATEGORY: fight/assault, punching/kicking, weapon use, apparent injury/fall,
   blood-like visual effect, property destruction, threat/chase, none, or unclear.
4. TIMESTAMPED EVIDENCE: approximate timestamps relative to the start of this uploaded clip.
5. WHY IT MAY HAVE BEEN FLAGGED: cite visible evidence, not imagined off-screen events.
6. UNCERTAINTY: distinguish what is visible from inference. Do not claim a character was killed
   unless the frames clearly establish it. Say "unclear from sampled frames" when appropriate.
7. HUMAN REVIEW: whether review is recommended and why.

Do not invent blood, injuries, weapons, causes, or exact timestamps. Do not report a violence
probability unless one was supplied by a separate detector.
""".strip()


def build_event_prompt(event: dict[str, Any], verbal_abuse: list[dict] | None = None,
                       audio_findings: list[dict] | None = None,
                       clip_offset_s: float = 0.0) -> str:
    """Add clearly labeled detector context without turning it into visual ground truth."""
    context = [
        "Machine-generated supporting context (may be wrong; do not treat it as a visual observation):",
        f"Original event interval: {float(event.get('start_s', 0)):.2f}s to {float(event.get('end_s', 0)):.2f}s.",
        f"This supplied clip begins at original video time {clip_offset_s:.2f}s.",
    ]
    if event.get("visual_score_available", True) and event.get("peak_prob") is not None:
        context.append(
            f"X3D binary class score: {float(event['peak_prob']):.3f} peak (uncalibrated for cartoons); "
            f"heuristic review priority: {event.get('severity_label', 'Not assessed')} (not violence severity)."
        )
    else:
        context.append("Visual violence detector did not flag this interval; the interval was selected from audio analysis only.")
    speech = verbal_abuse or []
    if speech:
        context.append("Verbal-abuse candidates (ASR transcript preserved in original language):")
        for span in speech[:8]:
            context.append(
                f"- {span.get('start_s', 0):.2f}-{span.get('end_s', 0):.2f}s; "
                f"language={span.get('language') or 'unknown'}; "
                f"categories={', '.join(span.get('categories') or ['toxicity candidate'])}; "
                f"text={span.get('transcript') or span.get('text') or ''!r}; "
                f"score={span.get('confidence', 0):.3f}"
            )
    else:
        context.append("Verbal abuse: no candidate was reported by the configured audio pipeline.")
    sounds = audio_findings or []
    if sounds:
        context.append("Audio-event model candidates (sound category is not proof of a real-world event):")
        for item in sounds[:8]:
            context.append(
                f"- {item.get('start_s', 0):.2f}-{item.get('end_s', 0):.2f}s; "
                f"category={item.get('category', 'unknown')}; score={item.get('confidence', 0):.3f}"
            )
    else:
        context.append("Audio events: no sound-event candidate was reported by the configured pipeline.")
    return DEFAULT_VLM_PROMPT + "\n\n" + "\n".join(context)


def extract_subclip(
    video_path: str | Path,
    start_s: float,
    end_s: float,
    output_path: str | Path | None = None,
    padding_s: float = 0.5,
) -> str:
    """Extracts a short subclip slice for a flagged event using ffmpeg.

    Pads the start/end slightly for temporal context. Returns the output file path.
    """
    video_path = str(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    from config.settings import ensure_ffmpeg
    ffmpeg_bin = ensure_ffmpeg() or "ffmpeg"

    s = max(float(start_s) - float(padding_s), 0.0)
    dur = max(float(end_s) - float(start_s) + 2 * float(padding_s), 1.0)

    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"qwen_subclip_{os.getpid()}_{hash((video_path, start_s, end_s)) & 0xffffffff}.mp4")
    output_path = str(output_path)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel", "error",
        "-ss", f"{s:.2f}",
        "-t", f"{dur:.2f}",
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        output_path,
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        if res.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except Exception as exc:
        logger.warning("ffmpeg subclip extraction failed: %s", exc)

    # Fallback to OpenCV if ffmpeg failed
    return _extract_subclip_opencv(video_path, s, dur, output_path)


def _extract_subclip_opencv(video_path: str, start_s: float, duration_s: float, output_path: str) -> str:
    """Fallback subclip extraction using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video via OpenCV: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start_s * fps)
    end_frame = int((start_s + duration_s) * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    curr_frame = start_frame
    while curr_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        curr_frame += 1

    cap.release()
    out.release()
    return output_path


def _available_physical_memory_bytes() -> int | None:
    """Return currently available system RAM without adding a platform dependency."""
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
            return None
        if hasattr(os, "sysconf"):
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None
    return None


def load_qwen_vl(model_id: str = MODEL_ID):
    """Load Qwen with its pretrained checkpoint dtype to avoid an unnecessary FP32 expansion.

    The official checkpoint uses bfloat16. Keeping that dtype roughly halves the model-weight
    memory compared with forcing float32. A RAM guard still avoids starting a load that is
    likely to terminate the app on a memory-constrained Windows machine.
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    available = _available_physical_memory_bytes()
    if available is not None and available < QWEN_MIN_AVAILABLE_RAM_BYTES:
        gib = available / 1024**3
        raise MemoryError(
            f"Qwen2.5-VL-3B CPU loading needs about {QWEN_MIN_AVAILABLE_RAM_BYTES / 1024**3:.0f} GiB of available RAM "
            f"with the checkpoint's bfloat16 weights; "
            f"only {gib:.1f} GiB is available. The video candidates are still saved. "
            "Retry Qwen review after freeing memory or on a machine with more available RAM."
        )

    logger.info("Loading Qwen2.5-VL model (%s) on CPU...", model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    model.eval()
    return model, processor


def explain_clip_qwen(
    video_path: str | Path,
    fps: float = 0.5,
    max_tokens: int = 320,
    prompt: str = DEFAULT_VLM_PROMPT,
    model: Any = None,
    processor: Any = None,
) -> str:
    """Executes Qwen2.5-VL video inference on a clip and returns the text explanation."""
    import torch

    path = str(video_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video clip not found at: {path}")

    if model is None or processor is None:
        model, processor = load_qwen_vl(MODEL_ID)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        video_fps=float(fps),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to("cpu")
    elif isinstance(inputs, dict):
        inputs = {k: v.to("cpu") if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    answer = processor.batch_decode(
        output_ids[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return answer.strip()


def parse_qwen_visual_review(explanation_text: str) -> dict[str, Any]:
    """Parse the current JSON visual verdict and retain safe defaults for older outputs."""
    result: dict[str, Any] = {
        "visual_assessment": "uncertain", "observed_action": "", "violence_types": [],
        "evidence": "", "false_alarm_reason": "", "uncertainty": "",
        "audio_context": "", "verbal_context": "", "structured": False,
    }
    if not explanation_text:
        return result

    decoder = json.JSONDecoder()
    payload = None
    for match in re.finditer(r"\{", explanation_text):
        try:
            candidate, _ = decoder.raw_decode(explanation_text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        return result

    assessment = str(payload.get("visual_assessment", "")).strip().lower()
    if assessment not in {"violence_observed", "no_violence_observed", "uncertain"}:
        return result
    result["visual_assessment"] = assessment
    result["structured"] = True
    for key in ("observed_action", "evidence", "false_alarm_reason", "uncertainty", "audio_context", "verbal_context"):
        value = payload.get(key, "")
        result[key] = str(value).strip()[:1600] if value is not None else ""
    categories = payload.get("violence_types", [])
    if isinstance(categories, str):
        categories = [categories]
    if isinstance(categories, list):
        result["violence_types"] = [str(value).strip()[:80] for value in categories[:12] if value]
    return result


def parse_structured_explanation(explanation_text: str) -> dict[str, str]:
    """Parse current structured Qwen reviews and older numbered explanations."""
    sections: dict[str, str] = {
        "summary": "",
        "observed_actions": "",
        "category": "",
        "why_flagged": "",
        "timestamped_evidence": "",
        "visual_evidence": "",
        "audio_evidence": "",
        "verbal_abuse": "",
        "uncertainty": "",
        "human_review": "",
    }

    if not explanation_text:
        return sections

    visual_review = parse_qwen_visual_review(explanation_text)
    if visual_review["structured"]:
        assessment = visual_review["visual_assessment"].replace("_", " ").title()
        sections.update({
            "summary": visual_review["observed_action"] or f"Qwen visual assessment: {assessment}.",
            "observed_actions": visual_review["observed_action"],
            "category": ", ".join(visual_review["violence_types"]),
            "why_flagged": visual_review["false_alarm_reason"],
            "timestamped_evidence": visual_review["evidence"],
            "visual_evidence": visual_review["evidence"],
            "audio_evidence": visual_review["audio_context"],
            "verbal_abuse": visual_review["verbal_context"],
            "uncertainty": visual_review["uncertainty"],
            "human_review": "Qwen is a pretrained second opinion. Verify this result against the original clip.",
        })
        return sections

    patterns = [
        ("summary", r"1\.\s*(?:SHORT SUMMARY|SUMMARY)[:\s]*(.*?)(?=2\.\s*|$)"),
        ("observed_actions", r"2\.\s*(?:OBSERVED ACTIONS|ACTIONS)[:\s]*(.*?)(?=3\.\s*|$)"),
        ("category", r"3\.\s*(?:POSSIBLE CATEGORY|CATEGORY)[:\s]*(.*?)(?=4\.\s*|$)"),
        ("why_flagged", r"5\.\s*(?:WHY IT MAY HAVE BEEN FLAGGED|WHY FLAGGED)[:\s]*(.*?)(?=6\.\s*|$)"),
        ("timestamped_evidence", r"4\.\s*(?:TIMESTAMPED EVIDENCE|EVIDENCE)[:\s]*(.*?)(?=5\.\s*|$)"),
        ("uncertainty", r"6\.\s*(?:UNCERTAINTY)[:\s]*(.*?)(?=7\.\s*|$)"),
        ("human_review", r"7\.\s*(?:HUMAN REVIEW)[:\s]*(.*?)$"),
    ]

    for key, pat in patterns:
        m = re.search(pat, explanation_text, re.DOTALL | re.IGNORECASE)
        if m:
            sections[key] = m.group(1).strip()

    # Backward-compatible parsing for explanations produced by the former 7-section prompt.
    legacy_patterns = [
        ("why_flagged", r"4\.\s*(?:WHY IT MAY HAVE BEEN FLAGGED|WHY FLAGGED)[:\s]*(.*?)(?=5\.\s*|$)"),
        ("timestamped_evidence", r"5\.\s*(?:TIMESTAMPED EVIDENCE|EVIDENCE)[:\s]*(.*?)(?=6\.\s*|$)"),
    ]
    for key, pat in legacy_patterns:
        if not sections[key]:
            m = re.search(pat, explanation_text, re.DOTALL | re.IGNORECASE)
            if m:
                sections[key] = m.group(1).strip()
    evidence = sections["timestamped_evidence"]
    for key, label in (("visual_evidence", "VISUAL"), ("audio_evidence", "AUDIO"), ("verbal_abuse", "VERBAL ABUSE")):
        match = re.search(rf"{label}\s*:\s*(.*?)(?=\n\s*(?:VISUAL|AUDIO|VERBAL ABUSE)\s*:|$)", evidence,
                          re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = match.group(1).strip()

    # If regex parsing didn't match numbered structure, store entire text in summary
    if not any(sections.values()):
        sections["summary"] = explanation_text.strip()

    return sections


def build_offline_rule_explanation(
    event: dict[str, Any],
    video_meta: dict[str, Any] | None = None,
) -> str:
    """Return a factual availability note; never fabricate a visual caption from motion scores."""
    start_s = float(event.get("start_s", 0.0))
    end_s = float(event.get("end_s", 0.0))
    peak_p = float(event.get("peak_prob", 0.0))
    return f"""1. SHORT SUMMARY
Qwen2.5-VL explanation is unavailable. X3D produced an unconfirmed visual candidate from {start_s:.2f}s to {end_s:.2f}s (uncalibrated peak class score {peak_p:.3f}).

2. OBSERVED ACTIONS
Not assessed: no vision-language model output is available.

3. POSSIBLE CATEGORY
Unclear. The configured visual detector outputs a binary score and does not identify action categories.

4. TIMESTAMPED EVIDENCE
VISUAL: Candidate interval {start_s:.2f}s to {end_s:.2f}s; inspect the stored evidence frames.

5. WHY IT MAY HAVE BEEN FLAGGED
The binary violence score crossed the configured event threshold. This score is not a category label.

6. UNCERTAINTY
Visible actions and intent cannot be established from the binary score alone.

7. HUMAN REVIEW
Recommended: review the original clip and evidence frames.""".strip()
