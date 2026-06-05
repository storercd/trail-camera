"""Tests for SpeciesNet candidate selection helpers."""

from preview_frames import select_speciesnet_top_class


def test_select_speciesnet_top_class_should_skip_generic_bird_label() -> None:
    """Promote the next most likely specific candidate when bird is configured as generic."""
    candidates = [
        {
            "raw_class": "id;aves;;;;;bird",
            "label": "bird",
            "score": 0.6051611304283142,
        },
        {
            "raw_class": "id;aves;passeriformes;corvidae;cyanocitta;stelleri;steller's jay",
            "label": "steller's jay",
            "score": 0.32681676745414734,
        },
    ]

    selected = select_speciesnet_top_class(candidates, generic_labels_to_skip=["bird"])

    assert selected is not None
    assert selected.label == "steller's jay"
    assert selected.raw_class.endswith("steller's jay")


def test_select_speciesnet_top_class_should_fallback_when_all_candidates_generic() -> None:
    """Keep the best generic label when no more specific candidate exists."""
    candidates = [
        {
            "raw_class": "id;aves;;;;;bird",
            "label": "bird",
            "score": 0.7,
        },
        {
            "raw_class": "id;animal;;;;;animal",
            "label": "animal",
            "score": 0.2,
        },
    ]

    selected = select_speciesnet_top_class(candidates, generic_labels_to_skip=["bird", "animal"])

    assert selected is not None
    assert selected.label == "bird"
