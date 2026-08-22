"""Main pipeline orchestrator.

Scans a directory, runs dedup → quality scoring → clustering → grouping,
and outputs a PipelineResult ready for JSON serialization.

Pre-compute strategy: all intermediate data computed at once after upload,
then assembled into the final structured output.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from snap_tidy.pipeline.clustering import compute_embeddings, date_group, generate_labels, run_hdbscan
from snap_tidy.pipeline.dedup import find_duplicates
from snap_tidy.pipeline.models import GroupInfo, PhotoRecord, PipelineResult
from snap_tidy.pipeline.quality import score_photo
from snap_tidy.utils.exif import extract_basic_info, scan_directory
from snap_tidy.utils.hashes import hamming_distance


logger = logging.getLogger(__name__)


def process_directory(
    source_dir: str | Path,
    *,
    window: str = "week",
    min_cluster_size: int = 5,
    clip_device: str | None = None,
    clip_batch_size: int = 32,
    dhash_threshold: int = 5,
    simhash_threshold: float = 0.95,
    output_path: str | Path | None = None,
) -> PipelineResult:
    """Run the full photo cleanup pipeline on a directory.

    Args:
        source_dir: directory containing photos.
        window: date grouping granularity (day/week/month/year).
        min_cluster_size: HDBSCAN parameter.
        output_path: optional path to write JSON result.

    Returns:
        PipelineResult with groups, dedup info, and summary.
    """
    t_start = time.perf_counter()
    src = Path(source_dir).resolve()

    if not src.is_dir():
        raise ValueError(f"Not a directory: {src}")

    logger.info("Scan %s …", src)
    photo_paths = scan_directory(src)
    total_photos = len(photo_paths)
    logger.info("Found %d photos", total_photos)

    if total_photos == 0:
        return PipelineResult(
            source_dir=str(src),
            total_photos=0,
            after_dedup=0,
            groups=[],
            dedup_removed=[],
        )

    # ── Stage 1: load images + EXIF + basic info ───────────────────
    logger.info("Stage 1: load images …")
    records_map: dict[str, PhotoRecord] = {}
    images: list[PhotoRecord] = []  # parallel array of record refs (we mutate in place)

    for p in photo_paths:
        rec = PhotoRecord.from_path(p)
        info = extract_basic_info(p)
        rec.width = info.get("width", 0)
        rec.height = info.get("height", 0)
        rec.format = info.get("format", "")
        rec.capture_date = info.get("capture_date")
        records_map[rec.path] = rec
        images.append(rec)

    # ── Stage 2: dedup ─────────────────────────────────────────────
    logger.info("Stage 2: dedup …")
    pil_images = []
    paths_for_dedup = []

    for rec in images:
        try:
            from PIL import Image
            pil_images.append(Image.open(rec.path).convert("RGB"))
            paths_for_dedup.append(rec.path)
        except Exception as e:
            logger.warning("Cannot open %s: %s", rec.path, e)

    _, dup_groups = find_duplicates(
        pil_images, paths_for_dedup,
        dhash_threshold=dhash_threshold,
        simhash_threshold=simhash_threshold,
    )

    dedup_removed: list[PhotoRecord] = []

    for group_indices in dup_groups:
        group_records = [images[i] for i in group_indices]
        # Keep highest quality score — temporarily set to file size as proxy
        group_records.sort(key=lambda r: r.file_size, reverse=True)
        keeper = group_records[0]
        for dup_rec in group_records[1:]:
            dup_rec.is_duplicate = True
            dup_rec.duplicate_of = keeper.path
            dedup_removed.append(dup_rec)

    unique_images = [r for r in images if not r.is_duplicate]
    after_dedup = len(unique_images)

    # Clean up large image objects
    for img in pil_images:
        img.close()
    del pil_images

    logger.info("After dedup: %d unique (%d removed)", after_dedup, len(dedup_removed))

    if after_dedup == 0:
        return PipelineResult(
            source_dir=str(src),
            total_photos=total_photos,
            after_dedup=0,
            groups=[],
            dedup_removed=dedup_removed,
        )

    # ── Stage 3: quality scoring ───────────────────────────────────
    logger.info("Stage 3: quality scoring …")
    for rec in unique_images:
        try:
            with Image.open(rec.path) as img:
                qs = score_photo(img.convert("RGB"))
            rec.quality_score = qs["overall"]
            rec.quality_sharpness = qs["sharpness"]
            rec.quality_exposure = qs["exposure"]
            rec.quality_dimensions = qs["dimensions"]
            rec.quality_reasons = qs["reasons"]
        except Exception as e:
            logger.warning("Quality score failed for %s: %s", rec.path, e)
            rec.quality_score = 50.0  # neutral default

    # ── Stage 4: CLIP embeddings + HDBSCAN ─────────────────────────
    logger.info("Stage 4: CLIP + HDBSCAN clustering …")
    reloaded_pil = []
    for rec in unique_images:
        try:
            reloaded_pil.append(Image.open(rec.path).convert("RGB"))
        except Exception as e:
            logger.warning("Cannot reload %s: %s", rec.path, e)
            reloaded_pil.append(None)

    valid_idx = [i for i, img in enumerate(reloaded_pil) if img is not None]
    valid_imgs = [reloaded_pil[i] for i in valid_idx]

    if valid_imgs:
        device = clip_device if clip_device else "auto"
        embeddings = compute_embeddings(valid_imgs, device=device, batch_size=clip_batch_size)
    else:
        embeddings = np.zeros((after_dedup, 512))

    if embeddings.shape[0] > 0:
        cluster_labels = run_hdbscan(embeddings, min_cluster_size=min_cluster_size)
    else:
        cluster_labels = np.array([], dtype=int)

    # Assign cluster info back to records
    for i, idx in enumerate(valid_idx):
        unique_images[idx].cluster_id = int(cluster_labels[i])

    for img in reloaded_pil:
        if img:
            img.close()
    del reloaded_pil

    # ── Stage 5: date grouping ─────────────────────────────────────
    logger.info("Stage 5: date grouping …")
    dates_for_grouping = [(rec.path, rec.capture_date) for rec in unique_images]
    date_buckets = date_group(dates_for_grouping, window=window)

    # ── Stage 6: assign combined keys + labels ─────────────────────
    logger.info("Stage 6: assign groups …")

    # Generate visual group letters
    visual_letters = _assign_visual_groups(unique_images)

    for i, rec in enumerate(unique_images):
        rec.date_group = date_buckets.get(i, "unknown")
        rec.visual_group = visual_letters.get(i, "?")
        rec.combined_key = f"{rec.date_group}/{rec.visual_group}"

    # Generate cluster labels via CLIP prompts for clusters that exist
    n_clusters = max(cluster_labels) + 1 if len(cluster_labels) > 0 else 0
    cluster_centroids = {}
    if len(valid_idx) > 0 and n_clusters > 0:
        for cid in range(n_clusters):
            mask = cluster_labels == cid
            if mask.sum() > 0:
                centroid = embeddings[mask].mean(axis=0)
                cluster_centroids[cid] = centroid

    label_by_cid: dict[int, str] = {}
    if cluster_centroids:
        centroids_arr = np.stack([cluster_centroids[c] for c in sorted(cluster_centroids)])
        generated = generate_labels(centroids_arr)
        for j, cid in enumerate(sorted(cluster_centroids)):
            label_by_cid[cid] = generated[j]

    for rec in unique_images:
        cid = rec.cluster_id
        if cid >= 0 and cid in label_by_cid:
            rec.cluster_label = label_by_cid[cid]

    # ── Stage 7: build groups ──────────────────────────────────────
    logger.info("Stage 7: build groups …")
    group_map: dict[str, GroupInfo] = {}

    for rec in unique_images:
        key = rec.combined_key
        if key not in group_map:
            group_map[key] = GroupInfo(
                key=key,
                date_group=rec.date_group,
                visual_group=rec.visual_group,
                cluster_id=rec.cluster_id,
                label=rec.cluster_label or f"{rec.date_group} · {rec.visual_group}",
            )
        group_map[key].photos.append(rec)

    # Sort photos within each group by quality score descending
    for g in group_map.values():
        g.photos.sort(key=lambda p: -p.quality_score)
        # Assign position
        for pos, p in enumerate(g.photos, start=1):
            p.position_in_group = pos

    # Determine action for each photo (pre-audit, all pending initially)
    # Best photo in small groups (< threshold) can be auto-kept
    for g in group_map.values():
        for i, p in enumerate(g.photos):
            if i == 0 and g.size <= 1:
                p.action = "keep"
            elif i == 0 and g.size <= 3:
                p.action = "pending"  # user should review
            else:
                p.action = "pending"

    elapsed = time.perf_counter() - t_start
    logger.info("Pipeline complete in %.1fs", elapsed)

    result = PipelineResult(
        source_dir=str(src),
        total_photos=total_photos,
        after_dedup=after_dedup,
        groups=list(group_map.values()),
        dedup_removed=dedup_removed,
    )

    # Optionally write JSON
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.to_json())
        logger.info("Wrote %s", output_path)

    return result


def _assign_visual_groups(records: list[PhotoRecord]) -> dict[int, str]:
    """Assign letter-based visual group identifiers.

    Groups by cluster_id (HDBSCAN label). Noise (-1) gets "?" letter.
    Non-noise clusters get uppercase letters A, B, C, …
    """
    cluster_ids = [r.cluster_id for r in records]
    unique_clusters = sorted(set(cid for cid in cluster_ids if cid >= 0))

    letters = {cid: chr(ord("A") + i) for i, cid in enumerate(unique_clusters[:26])}
    # If more than 26 clusters, append numbers
    for i, cid in enumerate(unique_clusters[26:], start=1):
        letters[cid] = str(i)

    return {i: letters.get(cid, "?") for i, cid in enumerate(cluster_ids)}


# ── Adapter API for new CLI ─────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineState:
    """Mutable state mutated during run_pipeline()."""

    elapsed_ms: float = 0.0
    dedup_report: dict[str, list[str]] = field(default_factory=dict)


def run_pipeline(
    source_dir: str,
    *,
    window: str = "week",
    min_cluster_size: int = 5,
    quality_threshold: float = 60.0,
    top_k: int = 5,
    clip_device: str | None = None,
    clip_batch_size: int = 32,
    output_path: str | None = None,
    dhash_threshold: int = 5,
    simhash_threshold: float = 0.95,
) -> tuple[PipelineState, dict[str, Any]]:
    """Top-level entry point matching the CLI signature.

    Wraps process_directory() internally and returns (state, report_dict)
    to stay compatible with cli.py that expects this format.
    """
    t0 = time.perf_counter()
    state = PipelineState()

    result = process_directory(
        source_dir,
        window=window,
        min_cluster_size=min_cluster_size,
        clip_device=clip_device,
        clip_batch_size=clip_batch_size,
        dhash_threshold=dhash_threshold,
        simhash_threshold=simhash_threshold,
        output_path=output_path,
    )

    # Build dedup report: {path_kept: [dup1, dup2, ...]}
    for dup_rec in result.dedup_removed:
        keeper = dup_rec.duplicate_of or "?"
        if keeper not in state.dedup_report:
            state.dedup_report[keeper] = []
        state.dedup_report[keeper].append(dup_rec.path)

    # Apply quality threshold → auto-set keep/archive actions
    for g in result.groups:
        for p in g.photos:
            if p.quality_score >= quality_threshold:
                if p.position_in_group <= top_k:
                    p.action = "keep"
                else:
                    p.action = "pending"
            else:
                p.action = "archive"

    elapsed_s = time.perf_counter() - t0
    state.elapsed_ms = elapsed_s * 1000

    # Always include groups + dedup_removed — needed by frontend consumer
    report = {
        "total_photos": result.total_photos,
        "after_dedup": result.after_dedup,
        "n_groups": len(result.groups),
        "groups": [g.to_dict() for g in result.groups],
        "dedup_removed": [p.to_dict() for p in result.dedup_removed],
        "summary": result.summary,
        "elapsed_sec": round(elapsed_s, 2),
    }

    return state, report
