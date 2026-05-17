"""Tests for configurable classification filters."""

from __future__ import annotations

from classification import analyze_video_result


def test_analyze_video_result_should_ignore_excluded_megadetector_categories() -> None:
    """Videos with only excluded categories should be marked uninteresting."""
    decision = analyze_video_result(
        image_entry={
            "file": "camera.avi",
            "detections": [
                {"category": "3", "conf": 0.98, "frame_number": 10, "bbox": [0.1, 0.1, 0.2, 0.2]}
            ],
        },
        interesting_categories={"1", "2", "3"},
        threshold=0.7,
        excluded_megadetector_categories={"3"},
    )

    assert decision.bucket == "uninteresting"
    assert decision.top_frame is None
    assert decision.top_confidence is None
