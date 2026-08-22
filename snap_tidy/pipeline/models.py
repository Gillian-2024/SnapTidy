"""Data classes for the SnapTidy pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PhotoRecord:
    """Single photo after all pipeline stages."""

    # Identity
    path: str  # absolute file path
    filename: str
    width: int = 0
    height: int = 0
    file_size: int = 0  # bytes
    format: str = ""  # JPEG/PNG/WebP/...

    # Metadata
    capture_date: str | None = None  # ISO-8601 date (YYYY-MM-DD), from EXIF or mtime

    # Dedup
    dhash: int = 0
    simhash: int = 0
    is_duplicate: bool = False
    duplicate_of: str | None = None  # path of the kept photo in same dup group

    # Clustering
    cluster_id: int = -1  # HDBSCAN cluster id (-1 = noise)
    cluster_label: str = ""
    date_group: str = ""  # e.g. "W2026-33" (ISO week)
    visual_group: str = ""  # e.g. "A"
    combined_key: str = ""  # date_group + "/" + visual_group

    # Quality
    quality_score: float = 0.0  # 0-100
    quality_sharpness: float = 0.0
    quality_exposure: float = 0.0
    quality_dimensions: float = 0.0
    quality_reasons: list[str] = field(default_factory=list)

    # Audit
    action: str = "pending"  # keep | archive | pending

    # For dedup groups — set during audit output
    position_in_group: int = 0  # rank by quality score within its combined_key

    @classmethod
    def from_path(cls, p: Path) -> PhotoRecord:
        """Create a fresh record from a file path (before any processing)."""
        try:
            stat = p.stat()
            fsize = int(stat.st_size)
        except OSError:
            fsize = 0
        return cls(
            path=str(p.resolve()),
            filename=p.name,
            file_size=fsize,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "path": self.path,
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "format": self.format,
            "capture_date": self.capture_date,
            "dhash": self.dhash,
            "simhash": self.simhash,
            "is_duplicate": self.is_duplicate,
            "duplicate_of": self.duplicate_of,
            "cluster_id": self.cluster_id,
            "cluster_label": self.cluster_label,
            "date_group": self.date_group,
            "visual_group": self.visual_group,
            "combined_key": self.combined_key,
            "quality_score": round(self.quality_score, 1),
            "quality_sharpness": round(self.quality_sharpness, 1),
            "quality_exposure": round(self.quality_exposure, 1),
            "quality_dimensions": round(self.quality_dimensions, 1),
            "quality_reasons": self.quality_reasons,
            "action": self.action,
            "position_in_group": self.position_in_group,
        }


@dataclass
class GroupInfo:
    """A group of photos sharing (date_group, visual_group)."""

    key: str  # combined_key, e.g. "W2026-33/A"
    date_group: str
    visual_group: str
    cluster_id: int
    label: str = ""  # CLIP prompt-generated caption

    photos: list[PhotoRecord] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.photos)

    @property
    def best_photo(self) -> PhotoRecord | None:
        if not self.photos:
            return None
        return max(self.photos, key=lambda p: p.quality_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.key,
            "label": self.label,
            "dimensions": {
                "date": self.date_group,
                "visual": self.visual_group,
            },
            "cluster_id": self.cluster_id,
            "size": self.size,
            "photos": [p.to_dict() for p in sorted(self.photos, key=lambda p: -p.quality_score)],
        }


@dataclass
class PipelineResult:
    """Complete output of run_pipeline()."""

    source_dir: str
    total_photos: int  # including duplicates before removal
    after_dedup: int  # unique photos after dedup
    groups: list[GroupInfo]
    dedup_removed: list[PhotoRecord]

    @property
    def summary(self) -> dict[str, int]:
        counts = {"keep": 0, "archive": 0, "pending": 0}
        for g in self.groups:
            for p in g.photos:
                counts[p.action] = counts.get(p.action, 0) + 1
        return counts

    def to_json(self, indent: int = 2) -> str:
        payload = {
            "source_dir": self.source_dir,
            "total_photos": self.total_photos,
            "after_dedup": self.after_dedup,
            "groups": [g.to_dict() for g in self.groups],
            "dedup_removed": [p.to_dict() for p in self.dedup_removed],
            "summary": self.summary,
        }
        return json.dumps(payload, ensure_ascii=False, indent=indent)
