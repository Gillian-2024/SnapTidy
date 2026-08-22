"""Smoke test: synthetic images → full pipeline → valid JSON report.

All testing uses synthetic images only — never real photo libraries.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


TEST_IMAGES = Path(__file__).parent / "validation" / "hdbscan" / "test_hdbscan_images"


def _skip_if_no_images() -> None:
    if not TEST_IMAGES.is_dir():
        raise RuntimeError("SKIP: Test images not available")


def test_pipeline_returns_valid_json() -> None:
    """End-to-end smoke: process_directory produces PipelineResult with groups."""
    from snap_tidy.pipeline.pipeline import process_directory, PipelineResult

    _skip_if_no_images()
    result = process_directory(TEST_IMAGES, window="week")
    assert isinstance(result, PipelineResult)
    assert result.total_photos > 0

    # Verify JSON serialization round-trip
    json_str = result.to_json()
    data = json.loads(json_str)
    assert "total_photos" in data
    assert "after_dedup" in data
    assert "groups" in data
    assert "dedup_removed" in data
    assert "summary" in data


def test_group_structure() -> None:
    """Each group has required fields and sorted photos."""
    from snap_tidy.pipeline.pipeline import process_directory

    _skip_if_no_images()
    result = process_directory(TEST_IMAGES, window="week")

    for group in result.groups:
        assert group.key  # combined key
        assert group.date_group
        assert group.photos
        # Photos sorted by quality desc
        scores = [p.quality_score for p in group.photos]
        assert scores == sorted(scores, reverse=True)


def test_quality_scores_exist() -> None:
    """Every non-duplicate photo has a quality score."""
    from snap_tidy.pipeline.pipeline import process_directory

    _skip_if_no_images()
    result = process_directory(TEST_IMAGES, window="week")

    for group in result.groups:
        for photo in group.photos:
            assert 0 <= photo.quality_score <= 100
            assert isinstance(photo.quality_reasons, list)


def test_output_file_written() -> None:
    """output_path writes JSON to disk."""
    from snap_tidy.pipeline.pipeline import process_directory

    _skip_if_no_images()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_path = Path(tmp.name)

    try:
        process_directory(
            TEST_IMAGES, window="month", output_path=output_path,
        )
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["total_photos"] > 0
    finally:
        output_path.unlink(missing_ok=True)


def test_run_pipeline_api() -> None:
    """Procedural API returns (state, report_dict)."""
    from snap_tidy.pipeline.pipeline import run_pipeline, PipelineState

    _skip_if_no_images()

    state, report = run_pipeline(
        str(TEST_IMAGES),
        window="week",
        min_cluster_size=3,
        quality_threshold=60.0,
        top_k=3,
    )

    assert isinstance(state, PipelineState)
    assert state.elapsed_ms > 0
    assert report["total_photos"] > 0
    assert report["after_dedup"] > 0
    assert "summary" in report
    assert "n_groups" in report
