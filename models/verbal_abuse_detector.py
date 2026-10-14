"""Verbal Abuse, Bad Words, and Profanity Detection Engine.

Transcribes spoken audio from video streams and detects verbal aggression,
profanities, vulgarities, aggressive threats, hate speech, and derogatory insults.

Features:
- Categorized abuse taxonomy:
    * Verbal Aggression & Threats (kill you, destroy you, beat you up, break your neck)
    * Profanity & Vulgarity (expletives, curse words, vulgarities via better-profanity)
    * Insults, Harassment & Bullying (idiot, moron, loser, scum, trash, freak)
    * Hate Speech & Discriminatory Slurs
- Local audio extraction plus optional Faster-Whisper multilingual ASR with timestamps
- Optional multilingual toxicity classification (candidate signal, not a threat classifier)
- Subtitle ingestion retained as a transcript fallback
- Low-energy silence gating to skip quiet or non-dialogue segments
- Child-safe censorship masking (censor_text) and raw keyword tracking
- Standalone verbal abuse event generation and physical violence severity elevation
- Graceful offline execution (never crashes if audio or internet is missing)
"""
from __future__ import annotations

import logging
import gc
import os
import re
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
try:
    from better_profanity import profanity
except ImportError:  # optional fallback keeps threat/transcript analysis available in a minimal install
    profanity = None

from config.settings import ensure_ffmpeg, ensure_windows_runtime_compatibility

logger = logging.getLogger(__name__)


def _start_speech_heartbeat(task: str) -> tuple[threading.Event, threading.Thread]:
    """Keep long local model loads visible in the app terminal before progress can advance."""
    stopped = threading.Event()
    started_at = time.monotonic()

    def _heartbeat() -> None:
        while not stopped.wait(20.0):
            elapsed = time.monotonic() - started_at
            message = f"[speech-analysis] {task} is still active after {elapsed:.0f}s"
            print(message, flush=True)

    thread = threading.Thread(target=_heartbeat, name="speech-analysis-heartbeat", daemon=True)
    thread.start()
    return stopped, thread

# ---------------------------------------------------------------------------
# TAXONOMY CATEGORIES
# ---------------------------------------------------------------------------
CAT_THREAT = "Verbal Aggression & Threats"
CAT_PROFANITY = "Profanity & Vulgarity"
CAT_INSULT = "Insults & Bullying"
CAT_HATE = "Hate Speech & Slurs"
CAT_TOXICITY = "Toxic language candidate"
TOXICITY_MODEL_ID = "unitary/multilingual-toxic-xlm-roberta"
TOXICITY_MODEL_LANGUAGES = {"en", "fr", "es", "it", "pt", "tr", "ru"}
TAMIL_ABUSE_MODEL_ID = "Hate-speech-CNERG/tamil-codemixed-abusive-MuRIL"
TAMIL_ABUSE_MODEL_LANGUAGES = {"ta", "tam", "tamil"}

# Explicit threat patterns & keywords
THREAT_KEYWORDS = [
    "kill you", "i will kill you", "i'll kill you", "im gonna kill you", "i'm gonna kill you",
    "we will kill you", "we'll kill you", "kill them", "kill him", "kill her",
    "murder you", "i will murder you", "i'll murder you", "murder them",
    "beat you up", "beat your ass", "beat the crap out of", "beat the shit out of",
    "kick your ass", "kick your butt", "kick your face",
    "destroy you", "i will destroy you", "i'll destroy you",
    "break your neck", "break your bones", "break your face", "break your jaw",
    "smash your face", "smash your head", "smash your skull",
    "choke you", "strangle you", "slit your throat", "snap your neck",
    "drop dead", "burn in hell", "go to hell", "die you", "hope you die",
    "gonna hurt you", "i will hurt you", "i'll hurt you", "wipe you out",
    "blow your head off", "rip your head off", "punch your face"
]

# Insults, harassment and derogatory bullying keywords
INSULT_KEYWORDS = [
    "idiot", "idiots", "moron", "morons", "stupid", "dumb", "dumbass", "dumbasses",
    "loser", "losers", "pathetic", "ugly", "freak", "freaks", "scum", "scumbag",
    "scumbags", "trash", "piece of trash", "piece of shit", "worthless", "disgusting",
    "pig", "pigs", "clown", "clowns", "fool", "fools", "imbecile", "imbeciles",
    "creep", "creeps", "jerk", "jerks", "asshole", "assholes", "bastard", "bastards",
    "bitch", "bitches", "douchebag", "douchebags", "retard", "retarded",
    "coward", "cowards", "waste of space", "good for nothing"
]

# High-severity hate speech / discriminatory slurs
HATE_KEYWORDS = [
    "faggot", "fag", "nigger", "nigga", "chink", "kike", "spic", "wetback",
    "tranny", "retard"
]


class _FallbackProfanity:
    """Small English-only censor fallback used when better-profanity is not installed."""
    def __init__(self):
        self.words: set[str] = set()

    def load_censor_words(self):
        self.words.update({"fuck", "fucking", "shit", "bullshit", "damn", "hell"})

    def add_censor_words(self, words):
        self.words.update(str(word).lower() for word in words)

    def _pattern(self):
        choices = sorted((word for word in self.words if word), key=len, reverse=True)
        return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(word) for word in choices) + r")(?!\w)", re.IGNORECASE) if choices else None

    def contains_profanity(self, text: str) -> bool:
        pattern = self._pattern()
        return bool(pattern and pattern.search(text))

    def censor(self, text: str) -> str:
        pattern = self._pattern()
        return pattern.sub(lambda match: "*" * len(match.group()), text) if pattern else text


if profanity is None:
    profanity = _FallbackProfanity()
    logger.warning("better-profanity is not installed; using the built-in English fallback word list")

# Initialize and extend better-profanity censor dictionary
_INITIALIZED = False


def _init_profanity_lexicon() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    try:
        profanity.load_censor_words()
        # Add all custom threats, insults, and hate phrases to the censor set
        all_custom = THREAT_KEYWORDS + INSULT_KEYWORDS + HATE_KEYWORDS
        profanity.add_censor_words(all_custom)
        _INITIALIZED = True
    except Exception as e:
        logger.warning("Could not initialize better-profanity lexicon: %s", e)


_init_profanity_lexicon()


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------
@dataclass
class FlaggedWordMatch:
    """Individual flagged word or phrase detected in dialogue."""
    word: str
    category: str
    severity: str  # "Mild", "Moderate", "High"
    start_char: int
    end_char: int


@dataclass
class VerbalAbuseSpan:
    """A timestamped segment of audio containing flagged verbal abuse or profanity."""
    start_s: float
    end_s: float
    transcript: str
    censored_transcript: str
    flagged_words: list[str]
    categories: list[str]
    severity_level: str  # "Mild", "Moderate", "High"
    confidence: float
    detection_source: str  # "subtitles" or "faster_whisper"
    details: dict[str, Any] = field(default_factory=dict)
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str
    language: str | None
    confidence: float | None = None
    source: str = "faster_whisper"
    words: list[dict[str, Any]] = field(default_factory=list)
    toxicity_score: float | None = None
    toxicity_model: str | None = None
    toxicity_status: str = "not_scored"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# CORE DETECTOR
# ---------------------------------------------------------------------------
class VerbalAbuseDetector:
    """Acoustic and linguistic analyzer for detecting verbal violence and profanity."""

    def __init__(self, sample_rate: int = 16000, use_multilingual_classifier: bool = False,
                 toxicity_threshold: float = 0.85, whisper_model_size: str = "base"):
        self.sample_rate = sample_rate
        self.use_multilingual_classifier = use_multilingual_classifier
        self.toxicity_threshold = float(toxicity_threshold)
        self.whisper_model_size = whisper_model_size
        self._toxicity_model = None
        self._toxicity_model_key: str | None = None
        self._toxicity_attempted = False
        self._toxicity_error: str | None = None
        self._toxicity_model_errors: dict[str, str] = {}
        self._toxicity_models_used: set[str] = set()
        self._last_toxicity_model_id: str | None = None
        self._last_toxicity_status = "not_scored"
        _init_profanity_lexicon()

    def _score_multilingual_toxicity(self, text: str, language: str | None) -> float | None:
        """Score supported languages using their matching pretrained classifier."""
        lang = (language or "").split("-")[0].lower()
        self._last_toxicity_model_id = None
        if not self.use_multilingual_classifier:
            self._last_toxicity_status = "disabled"
            return None
        if lang in TAMIL_ABUSE_MODEL_LANGUAGES:
            model_key = TAMIL_ABUSE_MODEL_ID
        elif lang in TOXICITY_MODEL_LANGUAGES:
            model_key = TOXICITY_MODEL_ID
        else:
            self._last_toxicity_status = "unsupported_language"
            return None

        # Keep only one text classifier resident at a time. Some apps process
        # mixed-language audio, and both checkpoints are large enough to matter.
        if self._toxicity_model_key != model_key:
            self._toxicity_model = None
            gc.collect()
            self._toxicity_model_key = model_key
            self._toxicity_attempted = False
            self._toxicity_error = None
        try:
            if self._toxicity_model is None:
                if self._toxicity_attempted:
                    self._last_toxicity_status = "unavailable" if self._toxicity_error else "not_scored"
                    return None
                self._toxicity_attempted = True
                if model_key == TAMIL_ABUSE_MODEL_ID:
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer
                    import torch

                    stop_heartbeat, heartbeat_thread = _start_speech_heartbeat("Tamil abuse model loading")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(TAMIL_ABUSE_MODEL_ID)
                        model = AutoModelForSequenceClassification.from_pretrained(TAMIL_ABUSE_MODEL_ID)
                        model.eval()
                        labels = {int(index): str(label) for index, label in
                                  (getattr(model.config, "id2label", {}) or {}).items()}
                        abuse_index = next((index for index, label in labels.items()
                                            if any(term in label.casefold() for term in
                                                   ("abusive", "offensive", "toxic", "profanity"))), None)
                        if abuse_index is None:
                            generic_labels = all(label.casefold() in (f"label_{index}".casefold(), str(index))
                                                 for index, label in labels.items())
                            # The model card documents LABEL_0=Normal and LABEL_1=Abusive.
                            if model.config.num_labels == 2 and generic_labels:
                                abuse_index = 1
                            else:
                                raise ValueError(f"Cannot verify abusive label mapping for {TAMIL_ABUSE_MODEL_ID}: {labels}")
                        if model.config.num_labels != 2 or abuse_index not in labels:
                            raise ValueError(f"Unexpected Tamil abuse model labels: {labels}")

                        class _TamilAbuse:
                            def __init__(self, mdl, tok, positive_index):
                                self.model = mdl
                                self.tokenizer = tok
                                self.positive_index = positive_index

                            def predict(self, text: str):
                                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                                with torch.inference_mode():
                                    logits = self.model(**inputs).logits
                                    probability = torch.softmax(logits, dim=-1)[0, self.positive_index].item()
                                return {"toxicity": probability}

                        self._toxicity_model = _TamilAbuse(model, tokenizer, abuse_index)
                    finally:
                        stop_heartbeat.set()
                        heartbeat_thread.join(timeout=1.0)
                else:
                    try:
                        from detoxify import Detoxify
                        stop_heartbeat, heartbeat_thread = _start_speech_heartbeat("Multilingual toxicity model loading")
                        try:
                            self._toxicity_model = Detoxify("multilingual", device="cpu")
                        finally:
                            stop_heartbeat.set()
                            heartbeat_thread.join(timeout=1.0)
                    except ImportError:
                        logger.info("detoxify package not found; loading %s via transformers", TOXICITY_MODEL_ID)
                        from transformers import AutoModelForSequenceClassification, AutoTokenizer
                        import torch
                        stop_heartbeat, heartbeat_thread = _start_speech_heartbeat("Transformers toxicity model loading")
                        try:
                            tok = AutoTokenizer.from_pretrained(TOXICITY_MODEL_ID)
                            mdl = AutoModelForSequenceClassification.from_pretrained(TOXICITY_MODEL_ID)
                            mdl.eval()
                            class _HFToxicity:
                                def __init__(self, m, t):
                                    self.m = m
                                    self.t = t
                                def predict(self, txt: str):
                                    inputs = self.t(txt, return_tensors="pt", truncation=True, max_length=512)
                                    with torch.inference_mode():
                                        logits = self.m(**inputs).logits
                                        probs = torch.sigmoid(logits)[0].tolist()
                                    return {"toxicity": probs[0] if probs else 0.0}
                            self._toxicity_model = _HFToxicity(mdl, tok)
                        finally:
                            stop_heartbeat.set()
                            heartbeat_thread.join(timeout=1.0)
                logger.info("Multilingual toxicity model loaded")
            result = self._toxicity_model.predict(text)
            value = result.get("toxicity") if isinstance(result, dict) else None
            score = float(np.asarray(value).reshape(-1)[0]) if value is not None else None
            if score is None or not np.isfinite(score):
                self._last_toxicity_status = "unavailable"
                return None
            self._last_toxicity_model_id = model_key
            self._last_toxicity_status = "scored"
            self._toxicity_models_used.add(model_key)
            return score
        except Exception as exc:
            self._toxicity_error = f"{type(exc).__name__}: {exc}"
            self._toxicity_model_errors[model_key] = self._toxicity_error
            self._last_toxicity_status = "unavailable"
            logger.warning("Multilingual toxicity classifier unavailable: %s", exc)
            return None

    def release_models(self) -> None:
        """Release the optional toxicity model before the AudioSet model is loaded."""
        self._toxicity_model = None
        self._toxicity_model_key = None
        gc.collect()

    # -----------------------------------------------------------------------
    # 1. Linguistic & Lexicon Analysis
    # -----------------------------------------------------------------------
    def analyze_text(self, text: str, start_s: float = 0.0, end_s: float = 0.0,
                     detection_source: str = "lexicon_match", language: str | None = None,
                     toxicity_score: float | None = None) -> VerbalAbuseSpan | None:
        """Analyzes text for profanities, threats, insults, and slurs."""
        if not text or not text.strip():
            return None

        clean_text = text.strip()
        lower_text = clean_text.lower()
        language_code = (language or "").split("-")[0].lower()
        # The hand-built lexicons and censor list are English-only. For other
        # languages, retain transcript and use only the supported broad toxicity model.
        english_lexicon = language_code in ("", "en")

        flagged: list[FlaggedWordMatch] = []
        categories_set: set[str] = set()

        # Check explicit threats
        for phrase in (THREAT_KEYWORDS if english_lexicon else ()):
            pattern = r"\b" + re.escape(phrase) + r"\b"
            for m in re.finditer(pattern, lower_text):
                # Threats of death or serious injury are High severity
                is_high = any(k in phrase for k in ("kill", "murder", "break", "smash", "slit", "choke", "strangle", "destroy", "die"))
                sev = "High" if is_high else "Moderate"
                flagged.append(FlaggedWordMatch(
                    word=m.group(),
                    category=CAT_THREAT,
                    severity=sev,
                    start_char=m.start(),
                    end_char=m.end()
                ))
                categories_set.add(CAT_THREAT)

        # Check hate speech
        for phrase in (HATE_KEYWORDS if english_lexicon else ()):
            pattern = r"\b" + re.escape(phrase) + r"\b"
            for m in re.finditer(pattern, lower_text):
                flagged.append(FlaggedWordMatch(
                    word=m.group(),
                    category=CAT_HATE,
                    severity="High",
                    start_char=m.start(),
                    end_char=m.end()
                ))
                categories_set.add(CAT_HATE)

        # Check insults and bullying
        for phrase in (INSULT_KEYWORDS if english_lexicon else ()):
            pattern = r"\b" + re.escape(phrase) + r"\b"
            for m in re.finditer(pattern, lower_text):
                # Avoid duplicate matches if already captured by threat
                if any(m.start() >= f.start_char and m.end() <= f.end_char for f in flagged):
                    continue
                is_severe = phrase in ("asshole", "assholes", "bastard", "bastards", "bitch", "bitches", "piece of shit", "scumbag")
                sev = "Moderate" if is_severe else "Mild"
                flagged.append(FlaggedWordMatch(
                    word=m.group(),
                    category=CAT_INSULT,
                    severity=sev,
                    start_char=m.start(),
                    end_char=m.end()
                ))
                categories_set.add(CAT_INSULT)

        # Check general profanity via better-profanity
        try:
            if english_lexicon and profanity.contains_profanity(clean_text):
                # Identify words that were censored
                censored_check = profanity.censor(clean_text)
                words_orig = re.findall(r"\w+", clean_text)
                words_cens = re.findall(r"[\w\*]+", censored_check)
                for w_orig, w_cens in zip(words_orig, words_cens):
                    if "*" in w_cens:
                        w_lower = w_orig.lower()
                        if not any(w_lower in f.word.lower() for f in flagged):
                            flagged.append(FlaggedWordMatch(
                                word=w_orig,
                                category=CAT_PROFANITY,
                                severity="Moderate",
                                start_char=0,
                                end_char=0
                            ))
                            categories_set.add(CAT_PROFANITY)
        except Exception:
            pass

        toxicity_score = (self._score_multilingual_toxicity(clean_text, language)
                          if toxicity_score is None else float(toxicity_score))
        if toxicity_score is not None and toxicity_score >= self.toxicity_threshold:
            categories_set.add(CAT_TOXICITY)

        if not flagged and not categories_set:
            return None

        # Determine overall severity level
        if any(f.severity == "High" for f in flagged):
            overall_sev = "High"
        elif any(f.severity == "Moderate" for f in flagged) or len(flagged) >= 2:
            overall_sev = "Moderate"
        elif categories_set == {CAT_TOXICITY}:
            overall_sev = "Not assessed"
        else:
            overall_sev = "Mild"

        # Unique flagged words preserving order
        unique_words: list[str] = []
        for f in flagged:
            w_norm = f.word.strip().lower()
            if w_norm not in [u.lower() for u in unique_words]:
                unique_words.append(f.word.strip())

        # Child-safe censored transcript
        censored_transcript = self.censor_text(clean_text) if english_lexicon else clean_text

        # Confidence based on severity and match count
        conf = (toxicity_score if toxicity_score is not None and CAT_TOXICITY in categories_set else
                (0.95 if overall_sev == "High" else (0.85 if overall_sev == "Moderate" else 0.75)))

        return VerbalAbuseSpan(
            start_s=round(start_s, 2),
            end_s=round(end_s if end_s > start_s else start_s + 3.0, 2),
            transcript=clean_text,
            censored_transcript=censored_transcript,
            flagged_words=unique_words,
            categories=sorted(categories_set),
            severity_level=overall_sev,
            confidence=conf,
            detection_source=detection_source,
            details={"matches_count": len(flagged), "toxicity_score": toxicity_score,
                     "toxicity_model": self._last_toxicity_model_id if toxicity_score is not None else None},
            language=language,
        )

    def censor_text(self, text: str) -> str:
        """Masks profanity, insults, and aggressive threats with child-safe asterisks."""
        if not text:
            return ""
        try:
            return profanity.censor(text)
        except Exception:
            return text

    # -----------------------------------------------------------------------
    # 2. Subtitle Ingestion (Zero-latency offline)
    # -----------------------------------------------------------------------
    def extract_subtitles(self, video_path: str | Path) -> list[tuple[float, float, str]]:
        """Finds and parses .vtt or .srt subtitle files matching video name."""
        path = Path(video_path)
        candidates = [
            path.with_suffix(".vtt"),
            path.with_suffix(".srt"),
            path.with_suffix(".en.vtt"),
            path.with_suffix(".en.srt"),
            path.parent / f"{path.stem}.en.vtt",
            path.parent / f"{path.stem}.en.srt",
        ]
        # Also check for any .vtt or .srt in the same directory sharing the stem
        if path.parent.is_dir():
            for f in path.parent.glob(f"{path.stem}*.vtt"):
                if f not in candidates:
                    candidates.append(f)
            for f in path.parent.glob(f"{path.stem}*.srt"):
                if f not in candidates:
                    candidates.append(f)

        for sub_file in candidates:
            if sub_file.is_file():
                try:
                    cues = self._parse_subtitle_file(sub_file)
                    if cues:
                        logger.info("Loaded %d subtitle cues from %s", len(cues), sub_file.name)
                        return cues
                except Exception as e:
                    logger.warning("Error parsing subtitle file %s: %s", sub_file, e)

        return []

    def _parse_subtitle_file(self, sub_path: Path) -> list[tuple[float, float, str]]:
        """Parses WebVTT or SRT timestamps and caption blocks."""
        text = sub_path.read_text(encoding="utf-8", errors="replace")
        time_pattern = re.compile(
            r"(?:(\d{1,2}):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d{1,2}):)?(\d{2}):(\d{2})[.,](\d{3})"
        )

        def to_seconds(h, m, s, ms) -> float:
            hours = float(h) if h else 0.0
            return hours * 3600.0 + float(m) * 60.0 + float(s) + float(ms) / 1000.0

        cues: list[tuple[float, float, str]] = []
        blocks = re.split(r"\n\s*\n", text)
        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if not lines:
                continue
            time_match = None
            for idx, line in enumerate(lines):
                m = time_pattern.search(line)
                if m:
                    time_match = m
                    content_lines = lines[idx + 1:]
                    break
            if time_match:
                start_s = to_seconds(time_match.group(1), time_match.group(2), time_match.group(3), time_match.group(4))
                end_s = to_seconds(time_match.group(5), time_match.group(6), time_match.group(7), time_match.group(8))
                cue_text = " ".join(content_lines).strip()
                # Remove HTML tags (e.g., <c>, <v Speaker>, <b>, <i>)
                cue_text = re.sub(r"<[^>]+>", "", cue_text).strip()
                if cue_text:
                    cues.append((start_s, end_s, cue_text))
        return cues

    # -----------------------------------------------------------------------
    # 3. Audio Extraction & Speech-to-Text
    # -----------------------------------------------------------------------
    def extract_audio_to_wav(self, video_path: str | Path, target_wav: str | None = None) -> str | None:
        """Extracts audio to a 16kHz mono 16-bit PCM WAV using ffmpeg."""
        video_path = str(video_path)
        if not os.path.isfile(video_path):
            return None

        ffmpeg_bin = ensure_ffmpeg() or "ffmpeg"
        if not target_wav:
            temp_dir = tempfile.gettempdir()
            target_wav = os.path.join(temp_dir, f"verbal_{os.getpid()}_{hash(video_path) & 0xffffffff}.wav")

        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-loglevel", "error",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(self.sample_rate),
                "-ac", "1",
                target_wav
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
            if res.returncode == 0 and os.path.isfile(target_wav) and os.path.getsize(target_wav) > 1024:
                return target_wav
        except Exception as e:
            logger.debug("ffmpeg audio extraction failed: %s", e)

        if target_wav and os.path.exists(target_wav):
            try:
                os.remove(target_wav)
            except OSError:
                pass
        return None

    def _fallback_speech_recognition(
        self,
        wav_path: str,
        duration_s: float | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple[list[TranscriptSegment], dict[str, Any]]:
        """Fallback speech transcription using SpeechRecognition when Faster-Whisper is unavailable."""
        progress_callback = progress_callback or (lambda fraction, message: None)
        progress_callback(0.20, "Faster-Whisper unavailable; falling back to secondary speech recognizer")
        transcript: list[TranscriptSegment] = []
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            if text and text.strip():
                transcript.append(TranscriptSegment(
                    start_s=0.0,
                    end_s=float(duration_s or 1.0),
                    text=text.strip(),
                    language="en",
                    source="speech_recognition",
                ))
            metadata = {
                "model": "speech_recognition/google_fallback",
                "language": "en",
                "language_probability": 0.8,
                "supported_languages": "English speech recognition fallback",
            }
            progress_callback(0.80, f"Fallback transcription complete; {len(transcript)} segment(s) identified")
            return transcript, metadata
        except Exception as err:
            logger.info("Speech recognition fallback completed without speech detected: %s", err)
            metadata = {
                "model": "speech_recognition/none",
                "language": None,
                "language_probability": None,
                "supported_languages": "none",
                "error": str(err),
            }
            return [], metadata

    def transcribe_audio(
        self,
        wav_path: str,
        duration_s: float | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple[list[TranscriptSegment], dict[str, Any]]:
        """Transcribe normalized audio locally with multilingual Faster-Whisper."""
        progress_callback = progress_callback or (lambda fraction, message: None)
        ensure_windows_runtime_compatibility()

        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            logger.warning("Could not import faster_whisper: %s; using secondary speech recognizer", e)
            return self._fallback_speech_recognition(wav_path, duration_s, progress_callback)

        logger.info("Loading Faster-Whisper %s on CPU", self.whisper_model_size)
        progress_callback(0.08, f"Loading Faster-Whisper {self.whisper_model_size} on CPU; first use may download model files")
        transcript: list[TranscriptSegment] = []
        info = None
        stop_heartbeat, heartbeat_thread = _start_speech_heartbeat("Faster-Whisper load/transcription")
        try:
            model = None
            try:
                model = WhisperModel(
                    self.whisper_model_size, device="cpu", compute_type="int8",
                    cpu_threads=max(1, min(8, os.cpu_count() or 4)), num_workers=1
                )
            except Exception as e_int8:
                logger.warning("Faster-Whisper int8 failed (%s); trying float32 fallback", e_int8)
                model = WhisperModel(
                    self.whisper_model_size, device="cpu", compute_type="float32",
                    cpu_threads=max(1, min(8, os.cpu_count() or 4)), num_workers=1
                )

            logger.info("Faster-Whisper model loaded; decoding audio")
            progress_callback(0.15, "Whisper model loaded; transcribing audio")
            segments, info = model.transcribe(
                wav_path, beam_size=3, word_timestamps=True, vad_filter=True,
                multilingual=True, condition_on_previous_text=False,
            )
            last_log_at = time.monotonic()
            audio_duration = max(float(duration_s or 1.0), 1.0)
            for segment_index, segment in enumerate(segments, start=1):
                words = [{"start_s": float(word.start), "end_s": float(word.end),
                          "text": str(word.word), "confidence": float(word.probability)}
                         for word in (getattr(segment, "words", None) or [])]
                transcript.append(TranscriptSegment(
                    start_s=float(segment.start), end_s=float(segment.end), text=str(segment.text).strip(),
                    language=getattr(info, "language", None), source="faster_whisper", words=words,
                ))
                now = time.monotonic()
                if segment_index == 1 or now - last_log_at >= 8.0:
                    audio_progress = min(1.0, float(segment.end) / audio_duration)
                    logger.info("Whisper transcription reached %.1f/%.1f seconds; collected %d segment(s)",
                                float(segment.end), audio_duration, segment_index)
                    progress_callback(
                        0.15 + 0.63 * audio_progress,
                        f"Transcribed {audio_progress:.0%} of audio ({float(segment.end):.0f}/{audio_duration:.0f}s); "
                        f"{segment_index} speech segment(s) found",
                    )
                    last_log_at = now
        except Exception as exc:
            logger.warning("Faster-Whisper transcription failed (%s); trying fallback recognizer", exc)
            return self._fallback_speech_recognition(wav_path, duration_s, progress_callback)
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=1.0)

        metadata = {
            "model": f"faster-whisper/{self.whisper_model_size}",
            "language": getattr(info, "language", None) if info else None,
            "language_probability": getattr(info, "language_probability", None) if info else None,
            "supported_languages": "multilingual Whisper model; language detected for the audio run",
        }
        logger.info("Whisper transcription complete: %d speech segment(s), language=%s",
                    len(transcript), metadata["language"] or "unknown")
        progress_callback(0.80, f"Transcription complete; scoring {len(transcript)} speech segment(s)")
        return transcript, metadata

    def _is_chunk_audible(self, wav_path: str, offset_s: float, duration_s: float, min_rms: float = 0.005) -> bool:
        """Quickly checks if a WAV slice contains audible sound (avoids unnecessary API calls for silence)."""
        try:
            with wave.open(wav_path, "rb") as wf:
                framerate = wf.getframerate()
                sampwidth = wf.getsampwidth()
                start_frame = int(offset_s * framerate)
                n_frames = int(duration_s * framerate)
                wf.setpos(min(start_frame, max(0, wf.getnframes() - 1)))
                raw_bytes = wf.readframes(n_frames)
                if not raw_bytes:
                    return False
                if sampwidth == 2:
                    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = np.sqrt(np.mean(samples ** 2)) if len(samples) > 0 else 0.0
                    return bool(rms >= min_rms)
        except Exception:
            pass
        return True

    # -----------------------------------------------------------------------
    # 4. End-to-End Analysis
    # -----------------------------------------------------------------------
    def analyze_video_full(
        self,
        video_path: str | Path,
        duration_s: float,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple[
            list[VerbalAbuseSpan], list[TranscriptSegment], dict[str, Any], list[str]]:
        """Transcribe the full audio track, retain speech text, and flag review candidates."""
        progress_callback = progress_callback or (lambda fraction, message: None)
        spans: list[VerbalAbuseSpan] = []
        transcripts: list[TranscriptSegment] = []
        warnings: list[str] = []
        metadata: dict[str, Any] = {"model": None, "language": None, "language_probability": None}

        progress_callback(0.01, "Checking for sidecar subtitles")
        subtitle_cues = self.extract_subtitles(video_path)
        for start_s, end_s, text in subtitle_cues:
            transcripts.append(TranscriptSegment(start_s, end_s, text, None, source="subtitle"))
            span = self.analyze_text(text, start_s, end_s, "subtitles")
            if span:
                spans.append(span)

        progress_callback(0.02, "Extracting the audio track")
        logger.info("Extracting 16 kHz mono audio for multilingual speech analysis")
        wav_path = self.extract_audio_to_wav(video_path)
        if wav_path:
            try:
                progress_callback(0.06, "Audio extracted; preparing multilingual transcription")
                asr_segments, metadata = self.transcribe_audio(
                    wav_path, duration_s=duration_s, progress_callback=progress_callback,
                )
                transcripts.extend(asr_segments)
                for segment_index, segment in enumerate(asr_segments, start=1):
                    progress_callback(
                        0.80 + 0.19 * (segment_index / max(len(asr_segments), 1)),
                        f"Scoring speech segment {segment_index}/{len(asr_segments)} ({segment.language or 'language unknown'})",
                    )
                    toxicity = self._score_multilingual_toxicity(segment.text, segment.language)
                    segment.toxicity_score = toxicity
                    segment.toxicity_model = self._last_toxicity_model_id
                    segment.toxicity_status = self._last_toxicity_status
                    span = self.analyze_text(
                        segment.text, segment.start_s, segment.end_s, "faster_whisper",
                        language=segment.language, toxicity_score=toxicity,
                    )
                    if span:
                        spans.append(span)
            except Exception as exc:
                warnings.append(f"Multilingual ASR unavailable or failed: {type(exc).__name__}: {exc}")
                logger.warning("Faster-Whisper transcription failed: %s", exc)
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
        elif not transcripts:
            warnings.append("No decodable audio track or subtitle transcript was available.")
            progress_callback(0.95, "No audio track or subtitle transcript was available")
            logger.warning("No audio track could be extracted and no subtitle transcript was found")
        else:
            warnings.append("Audio extraction failed; only the available subtitle transcript was analyzed.")
            progress_callback(0.95, "Audio extraction failed; subtitle transcript retained")
            logger.warning("Audio extraction failed; continuing with %d subtitle transcript segment(s)",
                           len(transcripts))

        detected_language = str(metadata.get("language") or "").split("-")[0].lower()
        transcript_languages = sorted({str(item.language or "").split("-")[0].lower()
                                       for item in transcripts if item.language})
        scored_models = sorted(self._toxicity_models_used)
        if not self.use_multilingual_classifier:
            metadata["toxicity_classifier"] = {"status": "disabled"}
        elif scored_models:
            models = [TOXICITY_MODEL_ID if key == TOXICITY_MODEL_ID else key for key in scored_models]
            metadata["toxicity_classifier"] = {
                "status": "completed", "model": models[0] if len(models) == 1 else None,
                "models": models,
                "supported_languages": sorted(TOXICITY_MODEL_LANGUAGES | {"ta"}),
                "scored_segments": sum(item.toxicity_score is not None for item in transcripts),
                "language_note": (
                    "Tamil scorer is trained for Tamil/code-mixed abusive-language detection; "
                    "it is not validated for every dialect or ASR transcript."
                    if TAMIL_ABUSE_MODEL_ID in self._toxicity_models_used else None
                ),
                "errors": dict(self._toxicity_model_errors),
            }
            if self._toxicity_model_errors:
                warnings.append("Some language-specific toxicity/abuse scores could not be produced: "
                                + "; ".join(f"{model}: {error}" for model, error in self._toxicity_model_errors.items()))
        elif self._toxicity_model_errors:
            errors = dict(self._toxicity_model_errors)
            metadata["toxicity_classifier"] = {"status": "unavailable", "errors": errors}
            warnings.append("Optional toxicity/abuse scoring failed: "
                            + "; ".join(f"{model}: {error}" for model, error in errors.items()))
        elif not any(language in TOXICITY_MODEL_LANGUAGES or language in TAMIL_ABUSE_MODEL_LANGUAGES
                     for language in transcript_languages or [detected_language]):
            metadata["toxicity_classifier"] = {
                "status": "unsupported_language", "languages": transcript_languages or [detected_language or None],
                "supported_languages": sorted(TOXICITY_MODEL_LANGUAGES | {"ta"}),
            }
        else:
            metadata["toxicity_classifier"] = {"status": "no_scored_speech"}

        progress_callback(1.0, f"Speech analysis finished; {len(transcripts)} transcript segment(s), {len(spans)} candidate(s)")
        return self._merge_spans(spans), transcripts, metadata, warnings

    def analyze_video(self, video_path: str | Path, duration_s: float,
                      chunk_s: float = 6.0, step_s: float = 4.0) -> list[VerbalAbuseSpan]:
        """Backward-compatible span-only API; use analyze_video_full for transcript records."""
        spans, _, _, _ = self.analyze_video_full(video_path, duration_s)
        return spans

    def _merge_spans(self, spans: list[VerbalAbuseSpan], gap_tol_s: float = 2.0) -> list[VerbalAbuseSpan]:
        """Merges contiguous or overlapping verbal abuse detections."""
        if not spans:
            return []

        sorted_spans = sorted(spans, key=lambda s: s.start_s)
        merged: list[VerbalAbuseSpan] = []
        cur = sorted_spans[0]

        for nxt in sorted_spans[1:]:
            if nxt.start_s <= cur.end_s + gap_tol_s:
                # Merge into cur
                new_end = max(cur.end_s, nxt.end_s)
                combined_transcript = f"{cur.transcript} {nxt.transcript}".strip()
                combined_language = cur.language if cur.language == nxt.language else "mixed/unknown"
                combined_censored = (self.censor_text(combined_transcript)
                                     if combined_language in (None, "en") else combined_transcript)
                combined_flagged = list(dict.fromkeys(cur.flagged_words + nxt.flagged_words))
                combined_cats = sorted(set(cur.categories + nxt.categories))

                # Keep toxicity-only candidates distinct from assessed verbal severity.
                rank = {"Not assessed": 0, "Mild": 1, "Moderate": 2, "High": 3}
                sev = max((cur.severity_level, nxt.severity_level), key=lambda x: rank.get(x, 0))

                cur = VerbalAbuseSpan(
                    start_s=cur.start_s,
                    end_s=new_end,
                    transcript=combined_transcript,
                    censored_transcript=combined_censored,
                    flagged_words=combined_flagged,
                    categories=combined_cats,
                    severity_level=sev,
                    confidence=max(cur.confidence, nxt.confidence),
                    detection_source=cur.detection_source,
                    details={"merged_count": cur.details.get("merged_count", 1) + 1},
                    language=combined_language,
                )
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
        return merged

    # -----------------------------------------------------------------------
    # 5. Correlation with Physical Events
    # -----------------------------------------------------------------------
    def correlate_with_physical_events(
        self,
        events: list[Any],
        abuse_spans: list[VerbalAbuseSpan],
        padding_s: float = 1.0
    ) -> tuple[dict[int, list[VerbalAbuseSpan]], list[VerbalAbuseSpan]]:
        """Maps verbal abuse spans to overlapping physical events and identifies standalone verbal events.

        Returns:
            (event_id_to_spans, standalone_abuse_spans)
        """
        event_matches: dict[int, list[VerbalAbuseSpan]] = {ev.event_no: [] for ev in events}
        standalone_spans: list[VerbalAbuseSpan] = []

        for span in abuse_spans:
            matched = False
            for ev in events:
                # Overlap check with tolerance padding
                if not (span.end_s < ev.start_s - padding_s or span.start_s > ev.end_s + padding_s):
                    event_matches[ev.event_no].append(span)
                    matched = True

            if not matched:
                standalone_spans.append(span)

        return event_matches, standalone_spans
