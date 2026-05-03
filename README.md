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
- run_folder_mode: Output mode; use none for persistent output or timestamped for output/runs/<run_id>
- recursive: Recursively scan input directory for videos
- detector_verbose: Enable verbose MegaDetector output while processing
- generate_top_frame_previews: Extract top-frame preview images in the same run
- preview_output_dir: Output folder for top-frame preview images (relative to run output unless absolute)
- preview_include_uninteresting: Include uninteresting videos when extracting previews

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

## Outputs

- If run_folder_mode is none:
  - Sorted videos: output/interesting, output/uninteresting, output/failed
  - Metadata: output/metadata/megadetector_results.json, output/metadata/summary.json
  - Preview images: output/preview_frames/*.jpg (or preview_output_dir)
- If run_folder_mode is timestamped:
  - Each run is written to output/runs/<run_id>/
  - Sorted videos: output/runs/<run_id>/interesting, uninteresting, failed
  - Metadata: output/runs/<run_id>/metadata/megadetector_results.json and summary.json
  - Preview images: output/runs/<run_id>/preview_frames/*.jpg (or preview_output_dir)
