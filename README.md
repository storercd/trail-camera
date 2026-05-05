# Trail Camera Video Sorting

This project runs MegaDetector on videos and sorts them into output buckets based on whether detections are interesting.

## Configuration

All runtime settings are loaded from [process_videos.config.yaml](process_videos.config.yaml).

### Config keys

- input_dir: Folder containing videos
- output_dir: Folder to receive sorted videos
- metadata_db_path: SQLite metadata catalog location (relative paths resolve under output_dir)
- pipeline_version: Logical pipeline version stored with processing metadata (default: 0.1.0)
- model: MegaDetector model identifier or .pt path (default: MDV5A)
- frame_sample: Process every Nth frame (default: 5)
- interesting_threshold: Detection confidence threshold for classifying a video as interesting (default: 0.7)
- interesting_categories: Category IDs considered interesting. MD default labels are 1=animal, 2=person, 3=vehicle.
- move_files: Move files instead of copying them into output buckets
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

When write_json_exports is true, summary.json is written as a snapshot exported
from the SQLite metadata catalog. Set write_json_exports to false to skip JSON
exports while keeping SQLite as the source of truth.

When auto_open_html_report is true, the generated summary.html is opened in the
default local browser/app after the file is written.

When clip_interesting_videos is true, each interesting output video is trimmed to
the first and last interesting detection frame, expanded by frame_sample on both
sides (bounded by video start/end).

All output video filenames are date-prefixed using the source file's creation date
(format: YYYYMMDD-<original_filename>). Folder structure from the input tree is not
preserved in the output buckets — all files are written directly under the bucket
root (e.g. output/interesting/20260203-PICT0005.AVI). If two source files from
different input subfolders would produce the same output name, a numeric suffix is
appended to the later file (e.g. 20260203-PICT0005_1.AVI) to avoid overwriting.

Preview image filenames include top predicted species label and confidence score.
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

- Sorted videos: output/interesting, output/uninteresting, output/failed
- Metadata: output/metadata/megadetector_results.json, output/metadata/summary.json, output/metadata/summary.html, output/metadata/species_classifications.json
- Metadata catalog (SQLite): output/metadata/catalog.sqlite3 (default, configurable via metadata_db_path)
- Canonical video storage root: output/videos
- Preview images: output/preview_frames/*.jpg (or preview_output_dir)
- Species crops: output/preview_species_crops/*.jpg (or species_crop_output_dir)
