# Concise Plan

Note: this is not meant to be absolutely rigid, but simply to steer the upcoming work so we don't miss major issues.

# Permanent Output Migration

## Rough Plan

- Use `sha256` of full video contents as the canonical `video_id` for deduplication and storage.
- Store each video and its derived artifacts under one canonical per-video folder keyed by `video_id`.
- Do not store `source_relative_path`; imports are flat dumps from trail-camera cards, so source path is not a meaningful long-term identifier.
- Use canonical in-place replacement: one active record and one active artifact set per `video_id`.
- Add `pipeline_version` to metadata and processing outputs.
- Add two processing modes via command-line switch:
	- Process new videos only (skip known `video_id`s)
	- Reprocess existing videos from canonical stored originals
- Save original source video into canonical output storage so reprocessing does not require re-importing camera-card files.
- Use SQLite as the authoritative metadata store for incremental processing and reprocessing.
- Treat JSON files as export/report artifacts, not source of truth.
- Add core metadata tables for `videos`, `processing_state`, and `artifacts` keyed by `video_id`.
- Always generate report videos, preview frames, and species crops during processing (independent of report UI generation).
- Do not add CLI toggles for report videos, preview frames, or species crop generation.
- Replace static HTML summary with a local dynamic reporting app backed by SQLite.
- Implement reporting app first slice with list/filter/sort views and artifact preview playback.
- Include a "needs reprocess" view keyed by `pipeline_version` mismatch.

## Phased Implementation Checklist

### Phase 1: Core schema, versioning, and storage layout

- [ ] Add global pipeline version constant and config defaults.
	- Target files: `pipeline_models.py`, `pipeline_config.py`, `process_videos.config.yaml`
- [ ] Define canonical output layout for per-video storage keyed by `video_id`.
	- Target files: `pipeline_config.py`, `file_ops.py`
- [ ] Add SQLite database bootstrap and migrations for core tables.
	- New files: `metadata_store.py`, `migrations/001_initial.sql`
	- Tables: `videos`, `processing_state`, `artifacts`
- [ ] Add database path/config wiring.
	- Target files: `pipeline_models.py`, `pipeline_config.py`, `process_videos.config.yaml`

### Phase 2: Ingestion and identity pipeline

- [ ] Compute `sha256` for each discovered input video and use as `video_id`.
	- Target files: `file_ops.py`, `process_videos.py`
- [ ] Persist canonical original video file into output storage under `video_id`.
	- Target files: `classification.py` or new `storage_layout.py`, `process_videos.py`
- [ ] Upsert `videos` table rows on import/first-seen and refresh immutable metadata checks.
	- Target files: `metadata_store.py`, `process_videos.py`
- [ ] Add tests for hash identity and canonical original placement.
	- New/target files: `tests/test_file_ops.py`, `tests/test_storage_layout.py`

### Phase 3: Processing modes and reprocessing behavior

- [ ] Add CLI switch for processing mode: `new-only` and `reprocess-existing`.
	- Target files: `process_videos.py`, `README.md`
- [ ] Implement `new-only` query path against SQLite (`video_id` not present in `videos`).
	- Target files: `metadata_store.py`, `process_videos.py`
- [ ] Implement `reprocess-existing` path that reads canonical stored originals.
	- Target files: `metadata_store.py`, `process_videos.py`
- [ ] Record `pipeline_version`, `processed_at`, and mode into `processing_state`.
	- Target files: `metadata_store.py`, `process_videos.py`
- [ ] Add stale artifact cleanup policy for reprocessing.
	- Target files: `metadata_store.py`, `process_videos.py`, `file_ops.py`
- [ ] Add tests for mode behavior and in-place replacement rules.
	- New/target files: `tests/test_processing_modes.py`, `tests/test_artifact_cleanup.py`

### Phase 4: Artifact generation refactor (always-on)

- [ ] Ensure report videos are always generated independently of report UI generation.
	- Target files: `reporting.py`, `process_videos.py`
- [ ] Ensure preview frames and species crops are always generated.
	- Target files: `preview_frames.py`, `process_videos.py`
- [ ] Register all produced artifacts in `artifacts` table with upsert semantics.
	- Target files: `metadata_store.py`, `process_videos.py`, `reporting.py`, `preview_frames.py`
- [ ] Remove obsolete artifact toggles from config and docs.
	- Target files: `pipeline_models.py`, `pipeline_config.py`, `process_videos.config.yaml`, `README.md`
- [ ] Keep compatibility JSON exports as optional snapshots from SQLite.
	- Target files: `reporting.py`, `process_videos.py`

### Phase 5: Dynamic reporting app (first slice)

- [ ] Create local reporting app entrypoint backed by SQLite.
	- New files: `report_app.py` (or `report_app/__init__.py` + routes/views modules)
- [ ] Implement list page with filters: date range, bucket, species, confidence.
	- Target files: `report_app.py` (+ templates/static if split)
- [ ] Implement sorting and pagination for large datasets.
	- Target files: `report_app.py`
- [ ] Implement per-video detail view with report video, clip, preview, and species crop.
	- Target files: `report_app.py`
- [ ] Add `needs reprocess` filter using `pipeline_version` mismatch.
	- Target files: `report_app.py`, `metadata_store.py`
- [ ] Document run/start instructions for the reporting app.
	- Target files: `README.md`

### Phase 6: Hardening and migration

- [ ] Write one-time migration utility from legacy run folders into canonical SQLite-backed layout.
	- New file: `scripts/migrate_legacy_runs.py`
- [ ] Validate backward compatibility for current pipeline outputs during transition.
	- Target files: `process_videos.py`, `reporting.py`
- [ ] Add integration tests for end-to-end new-only and reprocess flows.
	- New/target files: `tests/test_pipeline_e2e.py`
- [ ] Final cleanup: remove deprecated run-folder assumptions and static-report-only paths.
	- Target files: `pipeline_config.py`, `process_videos.py`, `README.md`

### Suggested implementation order

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6

## Topics To Discuss

1. Unique video identity and canonical storage layout
	- Decision: use `sha256` of the full file contents as the unique video identifier.
	- Decision: store each identified video and its related artifacts together under one canonical folder keyed by `video_id`.
	- Decision: do not track `source_relative_path` in long-term metadata.

2. Reprocessing behavior and stale-artifact cleanup
	- Decision: use canonical in-place replacement (no per-run/per-version artifact copies by default).
	- Decision: introduce two processing modes (`new-only`, `reprocess-existing`) as a CLI switch.
	- Decision: preserve a canonical original video file under output storage for each `video_id` and reprocess from that source.
	- Decision: add `pipeline_version` to enable explicit reprocessing after logic upgrades (for example `0.1.0` -> `0.1.1`).
	- Rule: when reprocessing a `video_id`, replace owned derived artifacts and remove stale ones that no longer apply.

3. Incremental metadata storage and source of truth
	- Decision: use SQLite as the authoritative metadata store.
	- Decision: keep JSON outputs as export/report artifacts only.
	- Decision: store canonical state in three core tables:
		- `videos` (`video_id`, original filename/date/size/extension, stored original path, created_at)
		- `processing_state` (`video_id`, `pipeline_version`, processed_at, mode, bucket, top detection fields, status)
		- `artifacts` (`video_id`, artifact_type, path, updated_at, exists flag)
	- Rule: processing performs upserts for current state and removes stale artifacts that are no longer valid.

4. Artifact generation policy
	- Decision: always generate report videos, preview frames, and species crops so viewing tools are immediately responsive.
	- Decision: artifact generation is independent from report UI generation; reporting does not control whether these artifacts are produced.
	- Decision: no CLI toggles for report videos, preview frames, or species crops.
	- Rule: reprocessing replaces current artifacts in place and removes stale ones that no longer apply.

5. Reporting application
	- Decision: replace generated static HTML with a local dynamic reporting application backed by SQLite.
	- Decision: first slice includes:
		- Video list view with filtering by date range, bucket, species label, and confidence range.
		- Sorting by date, top species, and confidence.
		- Per-video detail panel with report video, canonical clip, preview image, and species crop.
		- Pagination and fast search for large libraries.
	- Decision: include a "needs reprocess" filter based on `pipeline_version` mismatch against current pipeline target version.

## Notes

- We are moving away from isolated run folders toward a permanent output target.
- Decisions from the discussion should be promoted into the Rough Plan section above.

