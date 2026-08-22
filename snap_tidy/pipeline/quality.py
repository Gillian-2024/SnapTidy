"""Rule-based quality scoring engine.

Weighted combination of 4 dimensions:
- Sharpness (Laplacian variance) — 35%
- Exposure (mean gray level) — 25%
- Dimensions (short edge) — 15%
- Burst redundancy (EXIF proximity) — 15% (computed externally)

Returns scores in 0-100 range with reason strings.
No ML models needed — pure PIL/numpy.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def score_photo(image: Image.Image, short_edge_penalty: bool = True) -> dict:
    """Score a single photo on quality dimensions.

    Args:
        image: PIL Image (any mode).
        short_edge_penalty: if True, penalize small images (default).

    Returns:
        {
            "overall": float (0-100),
            "sharpness": float (0-100),
            "exposure": float (0-100),
            "dimensions": float (0-100),
            "reasons": list[str],
        }
    """
    gray = _to_gray(image)

    sharpness = _score_sharpness(gray)
    exposure = _score_exposure(gray)
    dimensions = _score_dimensions(image) if short_edge_penalty else 100.0

    # Weighted composite
    overall = (
        0.35 * sharpness
        + 0.25 * exposure
        + 0.15 * dimensions
        + 0.15 * 100.0  # burst = 100 if not computed externally
    )

    reasons: list[str] = []
    if sharpness < 30:
        reasons.append(f"blurry (variance={sharpness:.0f})")
    elif sharpness < 60:
        reasons.append(f"slightly soft (variance={sharpness:.0f})")

    if exposure > 80:
        reasons.append("well-exposed")
    elif exposure < 30:
        reasons.append("poorly-exposed")

    dim_label = f"resolution={image.width}x{image.height}"
    if dimensions < 20:
        reasons.append(f"very-low-res ({dim_label})")
    elif dimensions < 50:
        reasons.append(f"low-res ({dim_label})")

    return {
        "overall": round(min(100.0, max(0.0, overall)), 1),
        "sharpness": round(sharpness, 1),
        "exposure": round(exposure, 1),
        "dimensions": round(dimensions, 1),
        "reasons": reasons,
    }


# ── Individual dimension scorers ────────────────────────────────────


def _to_gray(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy grayscale float array [0, 255]."""
    from PIL import Image as PILImage
    g = image.convert("L")
    return np.array(g, dtype=np.float64)


def _score_sharpness(gray: np.ndarray) -> float:
    """Sharpness via Laplacian variance.

    laplacian(gray.astype(float)) in validation used scipy.ndimage.laplacian.
    Here we approximate with cv2-like Sobel or pure numpy Laplacian kernel.
    """
    # 3×3 Laplacian kernel convolution via numpy slicing
    h, w = gray.shape
    lap = np.zeros((h - 2, w - 2), dtype=np.float64)

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                lap -= 4.0 * gray[1+h-2:1+h, 1+w-2:1+w]
            else:
                sign = 1.0 if (dy != 0 or dx != 0) else 0
                y_off = 1 + dy
                x_off = 1 + dx
                lap += sign * gray[y_off:y_off+h-2, x_off:x_off+w-2]

    # Clip boundary artifacts
    if lap.size == 0:
        return 0.0
    var = np.var(lap)

    # Scale to 0-100: typical sharp photo has var > 500
    if var >= 500:
        return min(100.0, 80.0 + (var - 500) / 500.0 * 20.0)
    elif var >= 50:
        return 10.0 + (var - 50) / 450.0 * 70.0
    else:
        return max(0.0, var / 50.0 * 10.0)


def _score_exposure(gray: np.ndarray) -> float:
    """Exposure quality based on mean gray level.

    Ideal: mean ≈ 128 (mid-tone). Penalize extremes.
    Overexposed: mean > 250 → very low.
    Underexposed: mean < 15 → very low.
    """
    mean = float(np.mean(gray))

    # Symmetric bell around mid-tone 128
    deviation = abs(mean - 128.0)
    max_deviation = 128.0  # range from 0 to 255 maps to 0–128 deviation

    ratio = 1.0 - (deviation / max_deviation)
    score = ratio * 100.0

    # Additional penalty for extreme over/under exposure
    if mean > 250:
        score *= max(0.1, (255 - mean) / 5.0)
    elif mean < 15:
        score *= max(0.1, mean / 15.0)

    return max(0.0, min(100.0, score))


def _score_dimensions(image: Image.Image) -> float:
    """Dimension quality score based on short edge.

    Hard limit: < 360px → 0 points.
    Linear decay: 360–720px → 0→50.
    Full score: ≥ 720px → 100.
    """
    short_edge = min(image.width, image.height)

    if short_edge < 360:
        return 0.0
    elif short_edge < 720:
        # Linear interpolation from 0 to 50
        return (short_edge - 360) / 360.0 * 50.0
    else:
        return 100.0
