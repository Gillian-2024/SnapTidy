"""dHash / SimHash duplicate detection benchmark for SnapTidy.

Generates synthetic 64x64 RGBA PNGs with known transform groups,
computes dHash (Hamming distance ≤5) and SimHash (cosine ≥0.95),
and reports per-group accuracy, false positive rate, and speed.

Security: ALL images are synthetically generated — never reads real photos.
"""

import hashlib
import os
import statistics
import time
from pathlib import Path

from typing import Union

import numpy as np
from PIL import Image

# ── Paths ───────────────────────────────────────────────────────────────
OUT = Path(__file__).parent / "generated"
OUT.mkdir(exist_ok=True)
# Clean old generated files before test
for _p in OUT.iterdir():
    if _p.suffix == '.png':
        _p.unlink()


# ── Synthetic image generator ──────────────────────────────────────────
def _next_gid():
    """Simple counter for unique group prefixes."""
    _next_gid.counter += 1
    return f"{_next_gid.counter:02d}"

_next_gid.counter = 0


def make_group(identical: int, brightness_range: tuple = None, crop_px: int = None):
    """Create *count* variations of one base pattern."""
    h, w = 64, 64
    gid = _next_gid()
    group = []
    for i in range(identical):
        img = Image.new("L", (w, h), 0)
        data = list(img.getdata())

        # Fill with structured gradient so hash isn't trivially flat
        for y in range(h):
            row_start = y * w
            base_val = int(128 + 80 * np.sin(2 * np.pi * y / h))
            for x in range(w):
                data[row_start + x] = min(255, max(0, base_val + int(x / w * 60)))

        if brightness_range and i > 0:
            shift = np.random.uniform(*brightness_range) * 100
            data = [max(0, min(255, v + shift)) for v in data]

        img.putdata(data)
        if crop_px and i > 0:
            img = img.crop((i * crop_px, i * crop_px, w - (i - 1) * crop_px, h - (i - 1) * crop_px))
            if img.size != (w, h):
                img = img.resize((w, h), Image.BICUBIC)

        path = OUT / f"{gid}_v{i}.png"
        img.save(path)
        group.append(path)
    return group


def gen_unique():
    """Generate 10 completely unique patterns."""
    paths = []
    for idx in range(10):
        arr = np.zeros((64, 64), dtype=np.uint8)
        angle = idx * 37  # arbitrary phase
        for y in range(64):
            for x in range(64):
                arr[y, x] = int(128 + 127 * np.sin(0.3 * x + angle) * np.cos(0.2 * y + angle))
        img = Image.fromarray(arr, mode="L")
        p = OUT / f"unique_{idx:02d}.png"
        img.save(p)
        paths.append(p)
    return paths


# ── Hash functions ─────────────────────────────────────────────────────
def dhash(image_path: Union[str, Path]) -> int:
    """Compute dHash from a grayscale image → 64-bit integer."""
    img = Image.open(image_path).convert("L").resize((9, 8), Image.LANCZOS)
    pixels = list(img.getdata())
    bits = []
    for y in range(8):
        for x in range(8):
            left = pixels[y * 9 + x]
            right = pixels[y * 9 + x + 1]
            bits.append(int(left > right))
    value = 0
    for b in bits:
        value = (value << 1) | b
    return value


def simhash(image_path: Union[str, Path], dim: int = 32) -> list:
    """Compute SimHash using pixel-gradient features + random projection.

    Returns a list of `dim` signed values (+1/-1).
    """
    rng = np.random.RandomState(42)
    img = Image.open(image_path).convert("L").resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype=float)

    # Gradient magnitude at each pixel
    gx = np.gradient(arr, axis=1)
    gy = np.gradient(arr, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_flat = grad_mag.flatten()  # 1024-d feature vector

    # Random projection → sign bit per dimension
    proj = rng.randn(dim, grad_flat.size) @ grad_flat
    return [1 if v >= 0 else -1 for v in proj]


def hamming_distance(h1: int, h2: int) -> int:
    """Bitwise Hamming distance between two integers."""
    xor = h1 ^ h2
    return bin(xor).count("1")


def cos_sim_simhash(a: list[int], b: list[int]) -> float:
    """Cosine similarity between two sign vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Main experiment ────────────────────────────────────────────────────
def run():
    np.random.seed(0)
    print("=" * 70)
    print("SnapTidy — dHash / SimHash Validation Test")
    print(f"Generated {len(list(OUT.iterdir()))} test images in {OUT}")
    print("=" * 70)

    # 1. Build groups
    identical_groups = [make_group(5) for _ in range(5)]
    color_groups = [make_group(5, (-0.1, 0.1)) for _ in range(5)]
    crop_groups = [make_group(5, crop_px=1) for _ in range(5)]
    unique_images = gen_unique()

    all_paths = [p for g in identical_groups + color_groups + crop_groups for p in g] + unique_images
    print(f"Total images: {len(all_paths)}")

    # 2. Compute dHash (corrected)
    print("\nComputing dHash...")
    t0 = time.perf_counter()
    dhashes = {str(p): dhash(p) for p in all_paths}
    t1 = time.perf_counter()
    init_ms = (t1 - t0) * 1000
    print(f"dHash init ({len(all_paths)} images): {init_ms:.0f}ms")

    print("Timing dHash (100 iterations)...")
    t0 = time.perf_counter()
    for _ in range(100):
        for p in all_paths:
            dhash(p)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000
    dhash_speed = len(all_paths) * 100 / (total_ms / 1000) if total_ms > 0 else float('inf')
    print(f"dHash speed: {dhash_speed:,.0f} images/sec")

    # Group-wise dHash stats
    print("\n--- dHash Results ---\n")
    print(f"{'Group':<25} {'N-pairs'} {'Avg HD':>7} {'Min':>5} {'Max':>5} {'≤5':>5}")
    print("-" * 60)
    for label, groups in [("Identical", identical_groups), ("Color-shift", color_groups), ("Crop", crop_groups)]:
        for gi, group in enumerate(groups):
            pairs = [(p, q) for i, p in enumerate(group) for j, q in enumerate(group) if i < j]
            dists = [hamming_distance(dhashes[str(p)], dhashes[str(q)]) for p, q in pairs]
            n_pass = sum(1 for d in dists if d <= 5)
            print(f"{label}-{gi+1:<19} {len(pairs):>5} {statistics.mean(dists):>7.1f} {min(dists):>5} {max(dists):>5} {n_pass:>5}/{len(pairs)}")

    # False positives among unique
    fp_pairs = 0
    total_unique_pairs = 0
    for i, p in enumerate(unique_images):
        for q in unique_images[i+1:]:
            total_unique_pairs += 1
            if hamming_distance(dhashes[str(p)], dhashes[str(q)]) <= 5:
                fp_pairs += 1
    fp_rate = fp_pairs / total_unique_pairs * 100 if total_unique_pairs > 0 else 0
    print(f"\nUnique-image false positive rate (HD ≤ 5): {fp_rate:.1f}% ({fp_pairs}/{total_unique_pairs})")

    # 3. Compute SimHash
    print("\nComputing SimHash (may take ~30s)...")
    simhashes = {str(p): simhash(p) for p in all_paths}

    print("\n--- SimHash Results ---\n")
    for threshold_name, thresh in [(">=0.90", 0.90), (">=0.95", 0.95), (">=0.97", 0.97)]:
        print(f"SimHash cosine similarity {threshold_name}:")
        for label, groups in [("Identical", identical_groups), ("Color-shift", color_groups), ("Crop", crop_groups)]:
            for gi, group in enumerate(groups):
                pairs = [(p, q) for i, p in enumerate(group) for j, q in enumerate(group) if i < j]
                sims = [cos_sim_simhash(simhashes[str(p)], simhashes[str(q)]) for p, q in pairs]
                n_pass = sum(1 for s in sims if s >= thresh)
                print(f"  {label}-{gi+1:<19} mean={statistics.mean(sims):.3f}  max={max(sims):.3f}  pass≥{thresh:.2f}={n_pass}/{len(pairs)}")

        # False positives
        fp = 0
        tp = 0
        for i, p in enumerate(unique_images):
            for q in unique_images[i+1:]:
                tp += 1
                if cos_sim_simhash(simhashes[str(p)], simhashes[str(q)]) >= thresh:
                    fp += 1
        fp_r = fp / tp * 100 if tp > 0 else 0
        print(f"  → Unique-image false positive rate: {fp_r:.1f}% ({fp}/{tp})")
        print()

    # Speed for SimHash
    t0 = time.perf_counter()
    for _ in range(20):
        for p in all_paths:
            simhash(p)
    t1 = time.perf_counter()
    simhash_speed = len(all_paths) * 20 / (t1 - t0) if (t1 - t0) > 0 else float('inf')

    # ── Final Summary Table ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (for README)")
    print("=" * 70)
    md = f"""| Metric | Value |
|--------|-------|
| Test images | {len(all_paths)} (identical ×25 + color-shift ×25 + crop ×25 + unique ×10) |
| dHash compute speed | {dhash_speed:,.0f} images/sec (100 iterations) |
| SimHash compute speed | {simhash_speed:,.0f} images/sec (20 iterations) |
| dHash HD identical (expected ≤5) | see detailed above |
| dHash FP rate on unique images | {fp_rate:.1f}% |
| SimHash ≥0.95 pass rate | see detailed above |
| SimHash FP rate on unique | see detailed above |
"""
    print(md)

    return md


if __name__ == "__main__":
    result = run()
    # Write raw output to file for parent process to capture
    with open(OUT / "output.txt", "w") as f:
        f.write(result)
    print("\nDone! Output written to:", OUT / "output.txt")
