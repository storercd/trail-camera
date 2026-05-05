# Trail Camera Video Sorting

This project runs MegaDetector on videos to detect objects of interest and classifies them using SpeciesNet.
The pipeline maintains a permanent SQLite catalog of all processed videos with their canonical originals,
classification results, and derivative artifacts (clips, previews, crops).

## Configuration

All runtime settings are loaded from [process_videos.config.yaml](process_videos.config.yaml).

### Config keys

- input_dir: Folder containing videos to process
- output_dir: Root directory for catalog, canonical video storage, and artifacts
- metadata_db_path: SQLite metadata catalog location (relative paths resolve under output_dir)
- pipeline_version: Logical pipeline version stored with processing metadata (default: 0.1.0)
- model: MegaDetector model identifier or .pt path (default: MDV5A)
- frame_sample: Process every Nth frame (default: 5)
- interesting_threshold: Detection confidence threshold for classifying a video as interesting (default: 0.7)
- interesting_categories: Category IDs considered interesting. MD default labels are 1=animal, 2=person, 3=vehicle.
- move_files: Move source files to canonical storage instead of copying (only effective in new-only mode)
- save_uninteresting_files: Save videos classified as uninteresting to output/uninteresting
- clip_interesting_videos: Clip interesting videos to the detected frame window with frame_sample buffering
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

Run with the default config file:

```bash
/usr/local/bin/python3 process_videos.py
```

Run with a custom config file:

```bash
/usr/local/bin/python3 process_videos.py --config my_config.yaml
```

Run only newly discovered videos (default mode):

```bash
/usr/local/bin/python3 process_videos.py --mode new-only
```

Reprocess already cataloged videos from canonical stored originals:

```bash
/usr/local/bin/python3 process_videos.py --mode reprocess-existing
```

Generate reports only from SQLite + artifact files, without reprocessing videos:

```bash
/usr/local/bin/python3 process_videos.py --mode report-only
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

## Local Reporting App

Run the local SQLite-backed reporting app:

```bash
/usr/local/bin/python3 report_app.py --config process_videos.config.yaml
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

Install project dependencies with:

```bash
/usr/local/bin/python3 -m pip install -r requirements.txt
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
