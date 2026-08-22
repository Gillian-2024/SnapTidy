"""Report formatting utilities — convert PipelineResult to display strings."""

from __future__ import annotations

from snap_tidy.pipeline.models import PipelineResult


def report_to_string(result: PipelineResult, indent: int = 2) -> str:
    """Return a human-readable string representation of the pipeline result.

    This is the primary export format used by the CLI audit view.
    """
    lines: list[str] = []
    lines.append(f"Source: {result.source_dir}")
    lines.append(f"Total photos scanned : {result.total_photos}")
    lines.append(f"After deduplication  : {result.after_dedup}")
    lines.append(f"Duplicates removed   : {len(result.dedup_removed)}")
    lines.append(f"Final groups         : {len(result.groups)}")

    summary = result.summary
    lines.append(f"Keep suggestions     : {summary['keep']}")
    lines.append(f"Archive suggestions  : {summary['archive']}")
    lines.append(f"Pending review       : {summary['pending']}")
    lines.append("")

    # Sort groups by size desc, then by key name
    sorted_groups = sorted(result.groups, key=lambda g: (-g.size, g.key))

    for g in sorted_groups:
        label = g.label if g.label else f"{g.date_group} · {g.visual_group}"
        lines.append(f"Group {g.key}: {label} ({g.size} items)")
        best = g.best_photo
        if best:
            lines.append(f"  Best: {best.filename} (score={best.quality_score:.0f}, action={best.action})")

        # Show remaining photos up to top_k
        remaining = g.photos[1:]
        for p in remaining[:9]:
            lines.append(
                f"    [{p.action:7s}] {p.filename} score={p.quality_score:.0f}"
            )
        if len(g.photos) > 10:
            lines.append(f"    ... (+{len(g.photos) - 10} more)")

    if result.dedup_removed:
        lines.append("")
        lines.append("--- Duplicates ---")
        for dup in result.dedup_removed:
            lines.append(f"  {dup.path} → {dup.duplicate_of}")

    return "\n".join(lines)
