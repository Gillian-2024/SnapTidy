"""EXIF metadata extraction for photo records.

Uses Pillow's built-in Image.getexif() (no extra install).
Falls back to file mtime when EXIF is missing.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


def extract_date(p: Path) -> str | None:
    """Extract capture date as ISO-8601 string (YYYY-MM-DD).

    Priority:
    1. EXIF DateTimeOriginal
    2. EXIF DateTimeDigitized
    3. File mtime (as fallback)
    4. None

    Returns "YYYY-MM-DD" or None if all sources fail.
    """
    # Try Pillow EXIF first
    try:
        img = _open_image(p)
        exif = img.getexif()
        if exif:
            # DateTimeOriginal tag = 0x9003
            dt = exif.get(0x9003) or exif.get(0x906C)  # DateTimeOriginal / DateTimeDigitized
            if dt and isinstance(dt, str) and len(dt) >= 10:
                parsed = datetime.fromisoformat(dt[:10])
                return parsed.strftime("%Y-%m-%d")
    except Exception:
        pass

    # Fallback to file mtime
    try:
        mtime = os.path.getmtime(p)
        parsed = datetime.fromtimestamp(mtime, tz=timezone.utc)
        return parsed.strftime("%Y-%m-%d")
    except OSError:
        return None


def extract_basic_info(p: Path) -> dict:
    """Extract basic image info without loading full pixel data."""
    info = {"path": str(p), "filename": p.name}

    try:
        img = _open_image(p)
        info["width"] = img.width
        info["height"] = img.height
        info["format"] = img.format or ""
        img.close()
    except Exception:
        pass

    try:
        info["file_size"] = p.stat().st_size
    except OSError:
        pass

    info["capture_date"] = extract_date(p)
    return info


def _open_image(p: Path) -> "PIL.Image.Image":
    """Lazy import to avoid hard dependency on Pillow at import time."""
    from PIL import Image
    return Image.open(p)


def scan_directory(directory: Path, extensions: set[str] | None = None) -> list[Path]:
    """Recursively find all image files in a directory.

    Default extensions: .jpg .jpeg .png .webp .bmp .tiff .tif .gif
    Returns sorted list of absolute paths.
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}

    ext_set = {e.lower() for e in extensions}
    photos = []

    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if Path(fname).suffix.lower() in ext_set:
                photos.append(Path(root) / fname)

    photos.sort()
    return photos
