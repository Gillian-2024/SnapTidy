"""Deduplication via dHash + SimHash union strategy.

Two hashes are duplicates if EITHER:
- dHash Hamming distance ≤ dhash_threshold (default 5)
- SimHash cosine similarity ≥ simhash_threshold (default 0.95 → HD ≤ 3.2, use ≤ 3)

Union strategy catches 8/9 common image transforms.
Within each duplicate group, keep the highest-quality photo.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from PIL import Image

from snap_tidy.utils.hashes import compute_dhash, compute_simhash, hamming_distance


# Cosine ≥ 0.95 ⇔ HD/bits ≤ 0.05 ⇔ HD ≤ 3.2 → use HD ≤ 3
_SIMHASH_HD_THRESHOLD = 3


def _compute_hashes(image_path: str | Path, img: Image.Image) -> tuple[int, int]:
    """Compute both hashes for one image."""
    return (compute_dhash(img), compute_simhash(img))


def find_duplicates(
    images: list[Image.Image],
    paths: list[str] | None = None,
    dhash_threshold: int = 5,
    simhash_threshold: float | int = 0.95,
) -> tuple[list[dict], list[list[int]]]:
    """Find duplicate groups among a list of PIL Images.

    Args:
        images: PIL Image objects to check.
        paths: optional list of filenames matching images.
        dhash_threshold: max Hamming distance for dHash match.
        simhash_threshold: cosine similarity (0–1) or max HD (int) for SimHash.

    Returns:
        (records, groups) where:
        - records[i] = {"path": ..., "dhash": int, "simhash": int}
        - groups = list of index lists, each list = one dup group (≥2 members)
    """
    # Normalize: float → HD; int stays as-is
    if isinstance(simhash_threshold, float):
        max_hd = int(np.floor(64.0 * (1.0 - simhash_threshold)))
    else:
        max_hd = int(simhash_threshold)
    n = len(images)
    records = []

    for i, img in enumerate(images):
        dh, sh = _compute_hashes(paths[i] if paths else f"img_{i}", img)
        records.append({"path": paths[i] if paths and i < len(paths) else f"img_{i}", "dhash": dh, "simhash": sh})

    # Brute-force pairwise comparison — O(n²) but fine for typical uploads (<10k)
    groups: list[list[int]] = []
    seen: set[int] = set()

    for i in range(n):
        if i in seen:
            continue
        for j in range(i + 1, n):
            if j in seen:
                continue
            hd = hamming_distance(records[i]["dhash"], records[j]["dhash"])
            if hd <= dhash_threshold:
                groups.append([i, j])
                seen.add(i)
                seen.add(j)
                continue
            cs_hm = hamming_distance(records[i]["simhash"], records[j]["simhash"])
            if cs_hm <= max_hd:
                groups.append([i, j])
                seen.add(i)
                seen.add(j)

    return records, groups
