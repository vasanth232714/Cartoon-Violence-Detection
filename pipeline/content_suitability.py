"""Fast title and transcript clues plus transparent evidence-combination rules."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

VOCAB_PATH = Path(__file__).resolve().parents[1] / "config" / "content_suitability.json"


@lru_cache(maxsize=1)
def _vocabulary() -> dict:
    with VOCAB_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def normalize_text(value: str) -> str:
    """Case-fold, remove punctuation, and collapse whitespace without losing Tamil letters."""
    value = (value or "").casefold().replace("&", " and ").replace("_", " ")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def classify_title(title: str, source: str = "filename") -> dict:
    vocab = _vocabulary()
    normalized = normalize_text(title)
    matches = []
    for phrase in vocab["nursery_rhyme_phrases"]:
        candidate = normalize_text(phrase)
        if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", normalized, flags=re.UNICODE):
            matches.append(phrase)
    concerning_matches = []
    for phrase in vocab.get("concerning_title_phrases", []):
        candidate = normalize_text(phrase)
        if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", normalized, flags=re.UNICODE):
            concerning_matches.append(phrase)
    conflict_matches = []
    for phrase in vocab.get("potential_conflict_phrases", []):
        candidate = normalize_text(phrase)
        if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", normalized, flags=re.UNICODE):
            conflict_matches.append(phrase)
    strong = bool(matches)
    mixed_context = strong and bool(conflict_matches)
    if mixed_context:
        context_result = "Kids-content and military/conflict clues both matched — human review needed."
        title_suitability = "Needs video/audio review"
    elif conflict_matches:
        context_result = "Potential military/conflict content — video/audio review needed."
        title_suitability = "Needs video/audio review"
    elif concerning_matches:
        context_result = "Potential violence or mature-content clue — review needed."
        title_suitability = "Needs video/audio review"
    elif strong:
        context_result = "Positive kids-content indicator — this does not guarantee the video is safe."
        title_suitability = "Kids-content indicator; detailed analysis pending"
    else:
        context_result = "No configured kids-content or conflict title clue matched."
        title_suitability = "Needs video/audio review"
    title_sentiment = {"status": "unavailable", "polarity": None, "label": "Unavailable"}
    try:
        from textblob import TextBlob
        polarity = float(TextBlob(title).sentiment.polarity)
        label = "Positive" if polarity > 0.05 else "Negative" if polarity < -0.05 else "Neutral"
        title_sentiment = {"status": "completed", "polarity": polarity, "label": label}
    except ImportError:
        title_sentiment["status"] = "unavailable (TextBlob not installed)"
    except Exception as exc:
        title_sentiment["status"] = f"unavailable ({type(exc).__name__})"
    return {
        "title": title,
        "normalized_title": normalized,
        "source": source,
        "matched_phrases": matches,
        "strong_match": strong,
        "concerning_phrases": concerning_matches,
        "has_concerning_title_phrase": bool(concerning_matches),
        "conflict_phrases": conflict_matches,
        "has_conflict_context": bool(conflict_matches),
        "has_mixed_kids_conflict_context": mixed_context,
        "context_result": context_result,
        "title_suitability": title_suitability,
        "title_sentiment": title_sentiment,
        "preliminary_result": (
            context_result
            if conflict_matches else
            "Title contains a concerning violence or mature-content phrase — review required."
            if concerning_matches else
            "Positive kids-content indicator — detailed analysis pending; not a safety conclusion."
            if strong else "No strong known kids' rhyme title match."
        ),
        "disclaimer": "A title or filename is only a clue and cannot establish child safety.",
    }


def x3d_threshold_from_text_score(score: float) -> float:
    """Map the preliminary 0–10 text score to an X3D 0–1 candidate threshold."""
    value = float(score)
    if not 0.0 <= value <= 10.0:
        raise ValueError("preliminary text score must be between 0 and 10")
    # AnalysisConfig requires a strictly open interval; avoid invalid endpoints.
    return round(max(0.01, min(0.99, value / 10.0)), 2)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if not normalized_text or not normalized_phrase:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text, flags=re.UNICODE))


def _negated(text: str, phrase: str, language: str, vocab: dict) -> bool:
    normalized = normalize_text(text)
    target = normalize_text(phrase)
    match = re.search(rf"(?<!\w){re.escape(target)}(?!\w)", normalized, flags=re.UNICODE)
    if not match:
        return False
    tokens_before = normalized[:match.start()].split()[-4:]
    tokens_after = normalized[match.end():].split()[:3]
    negations = [normalize_text(item) for item in vocab["negations"].get(language, [])]
    return any(token in negations for token in tokens_before + tokens_after)


def scan_transcript_segments(segments: Iterable[dict]) -> list[dict]:
    """Apply configured phrase rules to ASR text; these are rules, never model scores."""
    vocab = _vocabulary()
    findings = []
    for segment in segments:
        text = str(segment.get("text") or "")
        language = str(segment.get("language") or "").split("-", 1)[0].lower()
        language_key = "ta" if language in ("ta", "tam", "tamil") else "en" if language in ("", "en", "eng") else None
        if language_key is None:
            continue
        for category, language_phrases in vocab["abuse_phrases"].items():
            for phrase in language_phrases.get(language_key, []):
                if _contains_phrase(text, phrase) and not _negated(text, phrase, language_key, vocab):
                    findings.append({
                        "start_s": float(segment.get("start_s", 0.0)),
                        "end_s": float(segment.get("end_s", 0.0)),
                        "phrase": phrase,
                        "category": category,
                        "language": language_key,
                        "source": "configured phrase rule",
                        "score": None,
                    })
    return findings


def analyze_text_suitability(
    *, title_context: dict, transcript_segments: Iterable[dict] = (),
    verbal_findings: Iterable[dict] = (), toxicity_scores: Iterable[float] = (),
    toxicity_threshold: float = 0.85, transcript_status: str = "pending",
    default_language: str | None = None,
) -> dict:
    """Calculate a transparent preliminary text score; it is not a safety or violence model.

    Correlated kids-title and lyric matches share one positive contribution. Keyword abuse
    and toxicity are grouped by taking the stronger risk contribution, while repeated matches
    of the same abuse category do not stack. Sentiment is bounded and suppressed when abuse
    evidence exists so negative wording is not counted twice.
    """
    vocab = _vocabulary()
    rules = vocab.get("text_score", {})
    segments = list(transcript_segments)
    findings = list(verbal_findings)
    toxic = [float(value) for value in toxicity_scores]
    title_match = bool(title_context.get("strong_match"))
    source = str(title_context.get("source", "filename"))
    title_positive = (float(rules.get("title_match_youtube", 1.2)) if source == "youtube"
                      else float(rules.get("title_match_filename", 0.65))) if title_match else 0.0

    transcript_text = " ".join(str(row.get("text") or "") for row in segments)
    lyric_matches = [phrase for phrase in vocab.get("nursery_rhyme_phrases", [])
                     if _contains_phrase(transcript_text, phrase)]
    lyric_positive = float(rules.get("transcript_kids_phrase", 1.2)) if lyric_matches else 0.0
    title_risk = bool(title_context.get("concerning_phrases"))
    title_conflict = bool(title_context.get("conflict_phrases"))
    title_requires_review = title_risk or title_conflict
    title_risk_penalty = (
        float(rules.get("concerning_title_youtube_penalty", 1.5)) if source == "youtube"
        else float(rules.get("concerning_title_filename_penalty", 0.75))
    ) if title_risk else 0.0
    # Treat any number of conflict keywords as one contextual signal, not as
    # independent proof or a stack of violence findings.
    title_conflict_penalty = float(rules.get("conflict_title_penalty", 2.2)) if title_conflict else 0.0
    # Violence and conflict wording in one title are correlated contextual
    # clues, so apply only the stronger configured title penalty.
    title_context_penalty = max(title_risk_penalty, title_conflict_penalty)
    # Concerning title phrases dominate positive title/lyric context clues.
    kids_context_contribution = 0.0 if title_requires_review else max(title_positive, lyric_positive)

    english_text = []
    for row in segments:
        language = str(row.get("language") or default_language or "").split("-", 1)[0].lower()
        if language in ("en", "eng", "english") and str(row.get("text") or "").strip():
            english_text.append(str(row["text"]).strip())
    polarity = None
    sentiment_status = "no_english_transcript"
    if english_text:
        try:
            from textblob import TextBlob
            # Bound the input size to keep this auxiliary feature lightweight.
            text_for_sentiment = " ".join(english_text)[:20000]
            polarity = float(TextBlob(text_for_sentiment).sentiment.polarity)
            sentiment_status = "completed"
        except ImportError:
            sentiment_status = "unavailable (TextBlob not installed)"
        except Exception as exc:
            sentiment_status = f"unavailable ({type(exc).__name__})"

    category_penalties = rules.get("abuse_penalties", {})
    abuse_categories = set()
    for finding in findings:
        value = " ".join((str(finding.get("category") or ""), str(finding.get("text") or ""),
                          str(finding.get("phrase") or ""))).casefold()
        if "threat" in value or "kill" in value or "hurt" in value or "shoot" in value:
            abuse_categories.add("threat")
        elif "profan" in value or "swear" in value:
            abuse_categories.add("profanity")
        elif "insult" in value or "bully" in value:
            abuse_categories.add("insult")
        else:
            abuse_categories.add("other")
    lexical_penalty = sum(float(category_penalties.get(key, category_penalties.get("other", 1.5)))
                          for key in abuse_categories)
    lexical_penalty = min(lexical_penalty, float(category_penalties.get("max_total", 4.0)))
    toxicity_flag = any(score >= toxicity_threshold for score in toxic)
    toxicity_penalty = float(category_penalties.get("toxicity", 2.0)) if toxicity_flag else 0.0
    # Toxicity and explicit phrase rules can describe the same words; do not add them twice.
    abuse_penalty = max(lexical_penalty, toxicity_penalty)

    sentiment_contribution = 0.0
    sentiment_note = "Sentiment is contextual only; it does not measure child safety."
    if polarity is not None and not abuse_categories and not toxicity_flag and not title_requires_review:
        scale = float(rules.get("sentiment_scale", 0.25))
        limit = float(rules.get("sentiment_max_abs", 0.25))
        sentiment_contribution = max(-limit, min(limit, polarity * scale))
    elif polarity is not None:
        sentiment_note += " Its score contribution was suppressed because separate review context was already found."

    base = float(rules.get("base_score", 6.0))
    raw_score = base + kids_context_contribution + sentiment_contribution - abuse_penalty - title_context_penalty
    score = round(max(0.0, min(10.0, raw_score)), 2)
    classification = "Higher concern / review required"
    for band in rules.get("bands", []):
        if score >= float(band["minimum"]):
            classification = str(band["label"])
            break
    completed_transcript = transcript_status == "completed" and bool(segments)
    uncertainty = []
    if not completed_transcript:
        uncertainty.append("Transcript is missing or incomplete; this score is title-context only or partial.")
    if title_context.get("title") and not title_match:
        uncertainty.append("No configured kids-content title phrase matched.")
    if sentiment_status.startswith("unavailable"):
        uncertainty.append("TextBlob sentiment is unavailable; no sentiment value was inferred.")
    return {
        "score": score,
        "classification": classification,
        "base_score": base,
        "matched_title_phrases": list(title_context.get("matched_phrases", [])),
        "matched_concerning_title_phrases": list(title_context.get("concerning_phrases", [])),
        "matched_conflict_title_phrases": list(title_context.get("conflict_phrases", [])),
        "matched_transcript_kids_phrases": lyric_matches,
        "transcript_status": "available" if completed_transcript else "unavailable",
        "sentiment": {"status": sentiment_status, "polarity": polarity,
                      "contribution": round(sentiment_contribution, 3), "note": sentiment_note},
        "abuse_findings": findings,
        "abuse_categories": sorted(abuse_categories),
        "toxicity_scores": toxic,
        "contributions": {
            "base": round(base, 3),
            "title_match_raw_signal": round(title_positive, 3),
            "concerning_title_phrase_penalty_signal": round(-title_risk_penalty, 3),
            "potential_conflict_title_penalty_signal": round(-title_conflict_penalty, 3),
            "title_context_penalty_applied": round(-title_context_penalty, 3),
            "transcript_kids_phrase_raw_signal": round(lyric_positive, 3),
            "kids_context_combined_max_title_or_lyrics": round(kids_context_contribution, 3),
            "title_source_weight": source if title_match else "no title match",
            "sentiment_context_only": round(sentiment_contribution, 3),
            "abusive_language_grouped_penalty": round(-abuse_penalty, 3),
            "abusive_phrase_penalty_signal_before_grouping": round(-lexical_penalty, 3),
            "multilingual_toxicity_penalty_signal_before_grouping": round(-toxicity_penalty, 3),
        },
        "uncertainty": uncertainty,
        "rules_version": "text_suitability_rules_v1",
        "validated": False,
    }


def build_suitability_assessment(
    *, title_context: dict, visual_status: str, scored_visual_windows: int,
    audio_status: str, audio_metadata: dict, speech_status: str,
    speech_metadata: dict, transcript_count: int, x3d_events: list[dict],
    audio_findings: list[dict], verbal_findings: list[dict], toxicity_scores: list[float],
    toxicity_threshold: float = 0.85, toxicity_status: str = "unavailable",
    run_status: str = "completed",
) -> dict:
    """Combine independent evidence with conservative precedence; never fuse model scores."""
    sources = [
        {"source": "Title / filename", "status": "available" if title_context.get("title") else "unavailable",
         "detail": "; ".join(filter(None, [
             "Kids phrases: " + ", ".join(title_context.get("matched_phrases", []))
             if title_context.get("matched_phrases") else "No known rhyme phrase matched.",
             "Potential conflict phrases: " + ", ".join(title_context.get("conflict_phrases", []))
             if title_context.get("conflict_phrases") else "",
             "Concerning phrases: " + ", ".join(title_context.get("concerning_phrases", []))
             if title_context.get("concerning_phrases") else "",
         ]))},
        {"source": "X3D visual", "status": "available" if visual_status == "completed" and scored_visual_windows else "unavailable",
         "detail": f"{len(x3d_events)} candidate event(s) from {scored_visual_windows} scored window(s)."},
        {"source": "Sound events", "status": "available" if audio_status == "completed" and audio_metadata.get("status") == "completed" else "unavailable",
         "detail": f"{len(audio_findings)} sound candidate(s); model status: {audio_metadata.get('status', audio_status)}."},
        {"source": "Speech / verbal abuse", "status": "available" if speech_status == "completed" and transcript_count > 0 else "unavailable",
         "detail": f"{transcript_count} transcript segment(s), {len(verbal_findings)} phrase finding(s)."},
        {"source": "Toxicity / abuse scoring", "status": "available" if toxicity_scores and toxicity_status == "completed" else "unavailable",
         "detail": (f"{toxicity_status}: actual scores recorded for {len(toxicity_scores)} transcript segment(s)."
                    if toxicity_scores else f"{toxicity_status}: no toxicity scores were recorded; this is not a negative result.")},
    ]
    reasons: list[str] = []
    visual_flag = bool(x3d_events)
    verbal_flag = bool(verbal_findings)
    toxicity_flag = any(float(score) >= toxicity_threshold for score in toxicity_scores)
    audio_flag = bool(audio_findings)
    title_risk_flag = bool(title_context.get("concerning_phrases"))
    title_conflict_flag = bool(title_context.get("conflict_phrases"))
    if visual_flag:
        reasons.append(f"X3D produced {len(x3d_events)} visual violence candidate event(s); verify the original clips.")
    if verbal_flag:
        reasons.append(f"Speech analysis found {len(verbal_findings)} configured/model verbal-abuse or threat candidate(s).")
    if audio_flag:
        reasons.append(f"Audio analysis produced {len(audio_findings)} sound-event candidate(s); a sound label alone does not prove violence.")
    if toxicity_flag:
        reasons.append(f"At least one actual multilingual toxicity score met the configured {toxicity_threshold:.2f} threshold.")
    if title_risk_flag:
        reasons.append(
            "The title contains a configured concerning phrase ("
            + ", ".join(title_context.get("concerning_phrases", []))
            + "); this is contextual evidence and still needs review."
        )
    if title_conflict_flag:
        reasons.append(
            "The title contains potential military/conflict context ("
            + ", ".join(title_context.get("conflict_phrases", []))
            + "); this is not proof of violence and requires video/audio review."
        )

    # Direct concerning evidence takes precedence over a positive title clue.
    if visual_flag or verbal_flag:
        category = "Potential violence or verbal abuse"
    elif toxicity_flag:
        category = "Potentially unsuitable content"
    elif audio_flag:
        category = "Potentially unsuitable content"
    elif title_conflict_flag:
        category = "Needs human review"
    elif title_risk_flag:
        category = "Potentially unsuitable content"
    else:
        incomplete = (
            visual_status != "completed" or scored_visual_windows <= 0 or
            run_status in ("incomplete", "failed", "interrupted") or
            audio_status != "completed" or audio_metadata.get("status") != "completed" or
            speech_status != "completed" or transcript_count <= 0 or
            toxicity_status != "completed" or not toxicity_scores
        )
        if incomplete:
            category = "Incomplete analysis"
            reasons.append("One or more required evidence sources are unavailable or incomplete; absence of findings is not a safety result.")
        elif title_context.get("strong_match"):
            category = "Likely child-friendly content"
            reasons.append("A known children’s rhyme title matched and the configured visual, audio, and speech checks reported no concerning candidates.")
        else:
            category = "Needs human review"
            reasons.append("No strong known children’s rhyme clue or concerning candidate determined suitability.")

    if title_context.get("strong_match") and (visual_flag or verbal_flag or audio_flag or toxicity_flag or title_conflict_flag):
        reasons.append("The nursery-rhyme title did not override contradictory model or transcript evidence.")
    if title_context.get("title"):
        reasons.append("Title matching is contextual only; it is not proof that the video is safe for children.")
    return {"category": category, "reasons": reasons, "sources": sources,
            "toxicity_threshold": toxicity_threshold, "rules_version": "content_suitability_rules_v1"}
