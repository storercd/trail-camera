# Trail Camera Media Sorting

This project runs MegaDetector on camera trap media (videos and images) to detect objects of interest and classifies them using SpeciesNet.
The pipeline maintains a permanent SQLite catalog of all processed media with their canonical originals,
classification results, and derivative artifacts (clips, previews, crops).

## Configuration

All runtime settings are loaded from [process_videos.config.yaml](process_videos.config.yaml).

### Config keys

- input_dir: Folder containing camera trap media to process (videos and images)
- output_dir: Root directory for catalog, canonical media storage, and artifacts
- metadata_db_path: SQLite metadata catalog location (relative paths resolve under output_dir)
- pipeline_version: Logical pipeline version stored with processing metadata (default: 0.1.0)
- model: MegaDetector model identifier or .pt path (default: MDV5A)
- frame_sample: Process every Nth frame (default: 5)
- interesting_threshold: Detection confidence threshold for classifying a video as interesting (default: 0.7)
- interesting_categories: Category IDs considered interesting. MD default labels are 1=animal, 2=person, 3=vehicle.
- excluded_megadetector_categories: Category IDs to force as uninteresting even if included in interesting_categories
- uninteresting_species_labels: SpeciesNet top labels to force as uninteresting (case-insensitive)
- generic_species_labels_to_skip: Generic SpeciesNet labels to ignore so the next most likely specific candidate becomes the primary species
- move_files: Move source files to canonical storage instead of copying (only effective in new-only mode)
- save_uninteresting_files: Save media classified as uninteresting to output/uninteresting
- clip_interesting_videos: Clip interesting videos to the detected frame window with frame_sample buffering (images are copied)
- recursive: No longer has any effect — the pipeline always scans the input directory recursively, including all subdirectories
- detector_verbose: Enable verbose MegaDetector output while processing
- generate_html_report: Generate summary.html report output
- auto_open_html_report: Open summary.html in the default local app after it is generated
- write_json_exports: Write compatibility JSON exports (summary snapshot) from SQLite
- preview_output_dir: Output folder for top-frame preview images (relative to run output unless absolute)
- preview_include_uninteresting: Include uninteresting videos when extracting previews
- speciesnet_model: SpeciesNet model identifier (empty uses SpeciesNet default)
- speciesnet_geofence: Enable SpeciesNet geofence filtering
- speciesnet_label_in_filename: Add species label and score to preview image filenames
- species_crop_output_dir: Output folder for saved species crops (relative to run output unless absolute)
- species_crop_padding: Normalized bbox padding used when creating species crops

These descriptions are preserved from the original command-line parameter help text, adapted to config key names.

## Usage

Install the project using your local machine Python environment
(without re-resolving heavy ML dependencies):

```bash
python3 -m pip install --user -e . --no-deps
```

Ensure your user scripts directory is on PATH:

```bash
export PATH="$HOME/Library/Python/3.12/bin:$PATH"
```

To persist this for future terminals, add the same line to `~/.zshrc`.

Run with the default config file:

```bash
trail-camera-process
```

Ingest media from a mounted memory card into configured input_dir:

```bash
trail-camera-ingest-card
```

By default, successful ingest ejects the memory card.

Ingest from a custom mount root and keep source files on the card:

```bash
trail-camera-ingest-card --mount-root /Volumes --no-delete
```

Skip eject at the end when needed:

```bash
trail-camera-ingest-card --no-eject
```

Override destination directory and enable SHA-256 verification:

```bash
trail-camera-ingest-card --destination-dir input --verify-sha256
```

Run with a custom config file:

```bash
trail-camera-process --config my_config.yaml
```

Run only newly discovered media (default mode):

```bash
trail-camera-process --mode new-only
```

Reset matching catalog rows for files currently in input and re-ingest as new:

```bash
trail-camera-process --mode reprocess-input
```

Reprocess already cataloged entries from canonical stored originals:

```bash
trail-camera-process --mode reprocess-existing
```

Generate reports only from SQLite + artifact files, without reprocessing videos:

```bash
trail-camera-process --mode report-only
```

Top-frame preview extraction, SpeciesNet classification, and species-crop
generation run automatically as part of each processing pass.

In `new-only` mode, the pipeline hashes discovered inputs, stores canonical
originals, and processes only videos whose `video_id` is not already in the
metadata catalog.

In `reprocess-existing` mode, the pipeline reads stored canonical originals
from the metadata catalog and processes those videos without requiring camera
card input files.

Browser-playable report_videos are generated on every processing pass.
The HTML summary file itself is still controlled by generate_html_report.

## Design Notes

### Preview frame selection

Preview extraction is based on MegaDetector detections for each video:

1. The pipeline filters detections to only `interesting_categories` and only those at or above `interesting_threshold`.
2. It picks the single highest-confidence detection from that filtered set.
3. The detection's `frame_number` becomes `top_frame`.
4. Preview extraction reads exactly that frame from the video and saves it as the preview image.

If no detection passes filters, the video is classified as `uninteresting`, `top_frame` is unset, and preview extraction is skipped by default.
If `preview_include_uninteresting` is enabled, uninteresting videos are considered for preview extraction, but they still need a valid `top_frame`.

After SpeciesNet classification, the pipeline can also demote videos to `uninteresting`
when their top species label matches `uninteresting_species_labels`.
Matching species rows are then purged from SQLite after sync, except for videos marked as favorites.
When a SpeciesNet result is too generic (for example `bird`), the pipeline can skip that label and promote the next most likely specific candidate using `generic_species_labels_to_skip`.

## Local Reporting App

Run the local SQLite-backed reporting app:

```bash
trail-camera-report --config process_videos.config.yaml
```

The app starts a local HTTP server, defaulting to `http://127.0.0.1:8000`.

Optional flags:

- `--host`: bind address for the local server
- `--port`: TCP port (default `8000`)
- `--debug`: enable Flask debug mode

The reporting app reads only from:

- `output/metadata/catalog.sqlite3`
- per-video artifact files under `output/videos/<shard>/<shard>/<video_id>/...`

It does not run MegaDetector, SpeciesNet, or any artifact generation.

The catalog list view prefers species crop images for video card thumbnails, with
a fallback to full-frame preview images.

## Cleanup Existing Output

To reclaim disk space for already-processed videos that match uninteresting filters,
run:

```bash
/usr/local/bin/python3 scripts/cleanup_uninteresting.py --config process_videos.config.yaml
```

Use dry-run mode first to review matches:

```bash
/usr/local/bin/python3 scripts/cleanup_uninteresting.py --config process_videos.config.yaml --dry-run
```

Current first-slice capabilities:

- list page with filters for date range, bucket, species label, confidence, and needs reprocess
- sorting by capture date, processed time, species, confidence, and bucket
- pagination controls for larger catalogs
- per-video detail page with browser video, native clip, preview image, and species crop

When write_json_exports is true, summary.json is written as a snapshot exported
from the SQLite metadata catalog. Set write_json_exports to false to skip JSON
exports while keeping SQLite as the source of truth.

When auto_open_html_report is true, the generated summary.html is opened in the
default local browser/app after the file is written.

When clip_interesting_videos is true, each interesting output video is trimmed to
the first and last interesting detection frame, expanded by frame_sample on both
sides (bounded by video start/end).

All videos are stored under a permanent canonical directory keyed by video hash
(SHA-256), with the structure `output/videos/<shard>/<shard>/<video_id>/`.
This enables efficient reprocessing without requiring the original input files.

When multiple source files would produce the same output filename (same basename from
different dates), the pipeline appends a `__<video_id[:8]>` suffix to avoid collisions
while maintaining a deterministic, reproducible mapping.

Preview image filenames include the top predicted species label and confidence score.
SpeciesNet classifies cropped animal regions derived from MegaDetector bounding
boxes, and the crop images are saved for review.

## Dependencies

This repository is typically run against a pre-existing local Python
environment that already has `megadetector`/`speciesnet` and their
transitive dependencies installed.

Install only the project code (recommended path):

```bash
python3 -m pip install --user -e . --no-deps
```

If you need a full local dependency install into an isolated environment,
use:

```bash
python3 -m pip install -r requirements.txt
```

Or use the bootstrap script to create/recreate a clean virtual environment and
install all pinned dependencies in one command:

```bash
scripts/setup_env.sh --recreate
```

## Outputs

- Metadata catalog (SQLite): `output/metadata/catalog.sqlite3` (persistent, expandable with each processing run)
- Canonical video storage: `output/videos/<shard>/<shard>/<video_id>/` with source, interesting clips, reports, previews, crops
- Metadata exports: `output/metadata/megadetector_results.json`, `output/metadata/summary.json`, `output/metadata/summary.html`, `output/metadata/species_classifications.json`
- Preview images: `output/preview_frames/*.jpg` (or configured preview_output_dir)
- Species crops: `output/preview_species_crops/*.jpg` (or configured species_crop_output_dir)
