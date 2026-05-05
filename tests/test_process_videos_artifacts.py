"""Tests for species artifact mapping helpers."""

import json
from pathlib import Path

from processing_modes import collect_species_artifact_maps


def test_collect_species_artifact_maps_should_extract_preview_and_crop_paths(
    tmp_path: Path,
) -> None:
    """Build preview/crop maps keyed by source_relative_path from species report."""
    report_path = tmp_path / "species_classifications.json"
    payload = {
        "entries": [
            {
                "source_relative_path": "interesting/video1.AVI",
                "preview_image_final": str(tmp_path / "preview1.jpg"),
                "classification_input_image": str(tmp_path / "crop1.jpg"),
                "used_species_crop": True,
            },
            {
                "source_relative_path": "interesting/video2.AVI",
                "preview_image_final": str(tmp_path / "preview2.jpg"),
                "classification_input_image": str(tmp_path / "preview2.jpg"),
                "used_species_crop": False,
            },
        ]
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    preview_map, crop_map = collect_species_artifact_maps(report_path)

    assert preview_map == {
        "interesting/video1.AVI": tmp_path / "preview1.jpg",
        "interesting/video2.AVI": tmp_path / "preview2.jpg",
    }
    assert crop_map == {
        "interesting/video1.AVI": tmp_path / "crop1.jpg",
    }


def test_collect_species_artifact_maps_should_handle_missing_report(tmp_path: Path) -> None:
    """Return empty maps when species classification report is not present."""
    preview_map, crop_map = collect_species_artifact_maps(tmp_path / "missing.json")

    assert preview_map == {}
    assert crop_map == {}
