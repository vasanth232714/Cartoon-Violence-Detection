SCHEMA_VERSION = 4

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT NOT NULL UNIQUE,
    original_name TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    file_available INTEGER NOT NULL DEFAULT 1,
    file_size_bytes INTEGER NOT NULL,
    duration_s REAL NOT NULL,
    fps REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    frame_count INTEGER NOT NULL,
    container TEXT,
    fourcc TEXT,
    video_title TEXT,
    title_source TEXT NOT NULL DEFAULT 'filename',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    status TEXT NOT NULL,            -- running|completed|completed_with_warnings|incomplete|failed|interrupted
    model_key TEXT NOT NULL,
    model_info_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    app_version TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    elapsed_s REAL,
    timing_json TEXT,
    windows_total INTEGER,
    windows_done INTEGER,
    warnings_json TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_video ON analysis_runs(video_id);

CREATE TABLE IF NOT EXISTS window_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    window_index INTEGER NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    p_violent REAL,
    status TEXT NOT NULL,
    frame_indices_json TEXT,
    frame_times_json TEXT,
    inference_ms REAL,
    note TEXT,
    UNIQUE(run_id, window_index)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    event_no INTEGER NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    duration_s REAL NOT NULL,
    label TEXT NOT NULL,
    peak_prob REAL NOT NULL,
    mean_prob REAL NOT NULL,
    n_windows INTEGER NOT NULL,
    n_positive_runs INTEGER NOT NULL,
    peak_time_s REAL NOT NULL,
    window_indices_json TEXT NOT NULL,
    severity_score INTEGER,
    severity_label TEXT NOT NULL,
    severity_method TEXT NOT NULL,
    severity_json TEXT,
    motion_json TEXT,
    explanation TEXT,
    description_method TEXT,
    qwen_explanation TEXT,
    qwen_status TEXT,
    UNIQUE(run_id, event_no)
);

-- Transcript chunks and audio-event model outputs remain independent from visual event labels.
CREATE TABLE IF NOT EXISTS audio_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,             -- transcript|verbal_abuse|abuse_keyword|audio_event|processing_note
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    language TEXT,
    text TEXT,
    category TEXT,
    confidence REAL,
    source TEXT,
    details_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_audio_segments_run_time ON audio_segments(run_id, start_s);

CREATE TABLE IF NOT EXISTS evidence_frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    timestamp_s REAL NOT NULL,
    kind TEXT NOT NULL
);

-- Append-only: human decisions never modify the model prediction stored in `events`.
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    decision TEXT NOT NULL CHECK (decision IN ('reviewed','confirmed','rejected')),
    human_severity INTEGER CHECK (human_severity IS NULL OR human_severity BETWEEN 0 AND 3),
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""
