# Trail Camera Video Sorting

This project runs MegaDetector on videos and sorts them into output buckets based on whether detections are interesting.

## Configuration

All runtime settings are loaded from [process_videos.config.yaml](process_videos.config.yaml).

### Config keys

- input_dir: Folder containing videos
- output_dir: Folder to receive sorted videos
- model: MegaDetector model identifier or .pt path (default: MDV5A)
- frame_sample: Process every Nth frame (default: 5)
- interesting_threshold: Detection confidence threshold for classifying a video as interesting (default: 0.7)
- interesting_categories: Category IDs considered interesting. MD default labels are 1=animal, 2=person, 3=vehicle.
- move_files: Move files instead of copying them into output buckets
- save_uninteresting_files: Save videos classified as uninteresting to output/uninteresting
- clip_interesting_videos: Clip interesting videos to the detected frame window with frame_sample buffering
- run_folder_mode: Output mode; use none for persistent output or timestamped for output/runs/<run_id>
- recursive: Recursively scan input directory for videos
- detector_verbose: Enable verbose MegaDetector output while processing
- generate_html_report: Generate summary.html and browser-playable report_videos sidecars
- generate_top_frame_previews: Extract top-frame preview images in the same run
- preview_output_dir: Output folder for top-frame preview images (relative to run output unless absolute)
- preview_include_uninteresting: Include uninteresting videos when extracting previews
- classify_previews_with_speciesnet: Run SpeciesNet classification on preview images
- speciesnet_model: SpeciesNet model identifier (empty uses SpeciesNet default)
- speciesnet_geofence: Enable SpeciesNet geofence filtering
- speciesnet_label_in_filename: Add species label and score to preview image filenames
- speciesnet_use_crops: Use MegaDetector bbox crops for species classification
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

Top-frame preview extraction runs automatically as part of this command when
generate_top_frame_previews is true in the config.

The HTML summary report and its browser-playable report_videos sidecars are only
generated when generate_html_report is true in the config.

When clip_interesting_videos is true, each interesting output video is trimmed to
the first and last interesting detection frame, expanded by frame_sample on both
sides (bounded by video start/end).

When SpeciesNet preview classification is enabled, preview image filenames include
the top predicted species label and confidence score.

When speciesnet_use_crops is true, SpeciesNet classifies cropped animal regions
derived from MegaDetector bounding boxes, and the crop images are saved for review.

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

- If run_folder_mode is none:
  - Sorted videos: output/interesting, output/uninteresting, output/failed
  - Metadata: output/metadata/megadetector_results.json, output/metadata/summary.json, output/metadata/summary.html, output/metadata/species_classifications.json
  - Preview images: output/preview_frames/*.jpg (or preview_output_dir)
  - Species crops: output/preview_species_crops/*.jpg (or species_crop_output_dir)
- If run_folder_mode is timestamped:
  - Each run is written to output/runs/<run_id>/
  - Sorted videos: output/runs/<run_id>/interesting, uninteresting, failed
  - Metadata: output/runs/<run_id>/metadata/megadetector_results.json, summary.json, summary.html, and species_classifications.json
  - Preview images: output/runs/<run_id>/preview_frames/*.jpg (or preview_output_dir)
  - Species crops: output/runs/<run_id>/preview_species_crops/*.jpg (or species_crop_output_dir)
