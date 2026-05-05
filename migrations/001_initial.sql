CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    capture_date TEXT,
    filesize_bytes INTEGER NOT NULL,
    source_ext TEXT NOT NULL,
    stored_original_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processing_state (
    video_id TEXT PRIMARY KEY,
    pipeline_version TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    bucket TEXT,
    top_confidence REAL,
    top_category TEXT,
    top_frame INTEGER,
    status TEXT NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifacts (
    video_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    exists_flag INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (video_id, artifact_type),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_processing_state_pipeline_version
ON processing_state(pipeline_version);

CREATE INDEX IF NOT EXISTS idx_processing_state_status
ON processing_state(status);

CREATE INDEX IF NOT EXISTS idx_artifacts_type
ON artifacts(artifact_type);
