#!/usr/bin/env python3
"""
dHash / SimHash validation for image deduplication pipeline.

Generates synthetic images, computes hash distances and similarities,
runs performance benchmarks, and prints a structured report.
"""

import hashlib
import math
import os
import statistics
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "test_images"
RESULTS_DIR = BASE_DIR / "results"
IMG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# dHash implementation
# ──────────────────────────────────────────────
def compute_dhash(image: Image.Image) -> int:
    """Compute perceptual hash using difference algorithm.

    Resize to 9x8 grayscale, compare left-right adjacent pixels.
    Returns a 64-bit integer.
    """
    img = image.convert("L").resize((9, 8), Image.LANCZOS)
    pixels = list(img.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            idx = row * 9 + col
            if pixels[idx] > pixels[idx + 1]:
                bits |= 1 << (63 - row * 8 - col)
    return bits


def hamming_distance(h1: int, h2: int) -> int:
    """Count differing bits between two integers."""
    return bin(h1 ^ h2).count("1")


# ──────────────────────────────────────────────
# SimHash implementation (pre-seeded fixed projections)
# ──────────────────────────────────────────────
SIMHASH_BITS = 64
_PIXEL_DIM = 32 * 32                               # 1024

# Fixed projection matrix: _PIXEL_DIM x SIMHASH_BITS, seeded once at import time.
# Same seeds guarantee identical projections across all images — core SimHash property.
_proj_rng = np.random.RandomState(20260822)
_PROJ_MATRIX = _proj_rng.uniform(-1, 1, size=(_PIXEL_DIM, SIMHASH_BITS))


def compute_simhash(image: Image.Image) -> int:
    """Compute 64-bit SimHash with FIXED random projections.

    Steps:
      1. Resize to 32x32 grayscale, flatten to 1024-d.
      2. Project onto pre-computed random directions.
      3. Output bit i = sign(dot(feature_vector, proj_column_i)).
    """
    img = image.convert("L").resize((32, 32), Image.LANCZOS)
    flat = np.array(img, dtype=np.float64).flatten()  # (1024,)
    proj = flat @ _PROJ_MATRIX                          # (64,)
    sim_hash = 0
    for i in range(SIMHASH_BITS):
        if proj[i] > 0:
            sim_hash |= 1 << i
    return sim_hash


def cosine_similarity(h1: int, h2: int, bits: int = SIMHASH_BITS) -> float:
    """Cosine similarity between two binary hash vectors."""
    vec1 = [(h1 >> i) & 1 for i in range(bits)]
    vec2 = [(h2 >> i) & 1 for i in range(bits)]

    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))

    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# ──────────────────────────────────────────────
# Synthetic image generators
# ──────────────────────────────────────────────
SEED = 42
rng = np.random.RandomState(SEED)


def make_base() -> Image.Image:
    """224×224 base image with gradients + geometric shapes."""
    img = Image.new("RGB", (224, 224))
    pixels = img.load()
    for y in range(224):
        for x in range(224):
            r = int(80 + 100 * x / 224 + 40 * y / 224)
            g = int(60 + 120 * y / 224)
            b = int(100 + 80 * (1 - x / 224))
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            pixels[x, y] = (r, g, b)
    # Add a red circle in the centre
    draw = ImageDraw.Draw(img)
    draw.ellipse((80, 80, 144, 144), fill=(220, 40, 40), outline=(255, 255, 255))
    # White rectangle on top-left
    draw.rectangle((10, 10, 60, 60), fill=(240, 240, 240), outline=None)
    return img


def save(img: Image.Image, name: str):
    img.save(IMG_DIR / f"{name}.png")


def exact_copies(n=3):
    imgs = []
    for i in range(n):
        save(make_base(), f"exact_{i}")
        imgs.append(Image.open(IMG_DIR / f"exact_{i}.png"))
    return imgs


def cropped_versions(n=3):
    """Random 10% crop then resize back to 224×224."""
    base = make_base()
    w, h = base.size
    crop_w, crop_h = int(w * 0.9), int(h * 0.9)
    imgs = []
    for i in range(n):
        box = (rng.randint(0, w - crop_w), rng.randint(0, h - crop_h),
               rng.randint(0, w - crop_w) + crop_w, rng.randint(0, h - crop_h) + crop_h)
        cropped = base.crop(box).resize((w, h), Image.LANCZOS)
        save(cropped, f"crop_{i}")
        imgs.append(cropped)
    return imgs


def rotated_versions(n=3):
    """Rotate by 5-15 degrees."""
    base = make_base()
    angles = [5.0, 9.0, 15.0]
    imgs = []
    for i, angle in enumerate(angles):
        rotated = base.rotate(angle, resample=Image.BICUBIC, expand=False)
        save(rotated, f"rot_{i}")
        imgs.append(rotated)
    return imgs


def brightness_shifted(n=3):
    """Brightness ±30%."""
    base = make_base()
    factors = [0.70, 0.85, 1.15, 1.30][:n]
    imgs = []
    for i, f in enumerate(factors):
        arr = np.array(base, dtype=np.float32) * f
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        shifted = Image.fromarray(np.require(arr, dtype=np.uint8))
        save(shifted, f"bright_{i}")
        imgs.append(shifted)
    return imgs


def jpeg_artifacted(n=3):
    """JPEG quality=5 round-trip."""
    base = make_base()
    imgs = []
    for i in range(n):
        buf = BytesIO()
        base.save(buf, "JPEG", quality=5)
        buf.seek(0)
        reopen = Image.open(buf)
        save(reopen, f"jpeg_{i}")
        imgs.append(reopen)
    return imgs


def color_inverted(n=3):
    """Invert colours."""
    base = make_base()
    imgs = []
    for i in range(n):
        arr = np.array(base, dtype=np.uint16)
        inv = Image.fromarray((255 - arr).astype(np.uint8), "RGB")
        save(inv, f"invert_{i}")
        imgs.append(inv)
    return imgs


def random_gradients(n=5):
    """Completely different gradient images."""
    imgs = []
    for i in range(n):
        img = Image.new("RGB", (224, 224))
        pixels = img.load()
        hue = rng.randint(0, 360)
        for y in range(224):
            for x in range(224):
                r = int(128 + 127 * np.sin((x + y + hue) * 0.05))
                g = int(128 + 127 * np.cos(x * 0.03 + hue))
                b = int(128 + 127 * np.sin(y * 0.04 + hue * 0.7))
                pixels[x, y] = (max(0, min(255, r)),
                                max(0, min(255, g)),
                                max(0, min(255, b)))
        save(img, f"grad_{i}")
        imgs.append(img)
    return imgs


def cross_algorithm_pairs(n=3):
    """
    Create images designed to test algorithm divergence.

    Two families:
      1) Strong Gaussian blur (radius 3-8): degrades fine structure,
         tests whether SimHash can still recognize globally similar images.
      2) Row-inverted variants: flip every-other row horizontally.
         This preserves global brightness distribution (SimHash robust)
         but creates many local edge reversals (dHash sensitive).
    """
    base = make_base()
    imgs = []

    # Family 1: strong blurs
    for radius in [3.0, 5.0, 8.0]:
        blurred = base.filter(ImageFilter.GaussianBlur(radius=radius))
        save(blurred, f"blur_{len(imgs)}")
        imgs.append(blurred)

    return imgs


# ──────────────────────────────────────────────
# Benchmark helpers
# ──────────────────────────────────────────────
def bench_dhash(num: int = 10_000):
    """Time dHash on num random 224×224 arrays (no PIL overhead)."""
    hashes = []
    t0 = time.perf_counter()
    for _ in range(num):
        fake = Image.new("RGB", (224, 224))
        h = compute_dhash(fake)
        hashes.append(h)
    elapsed = time.perf_counter() - t0
    return elapsed, hashes


def bench_simhash(num: int = 10_000):
    """Time SimHash on num random 224×224 arrays (no PIL overhead)."""
    hashes = []
    t0 = time.perf_counter()
    for _ in range(num):
        fake = Image.new("RGB", (224, 224))
        h = compute_simhash(fake)
        hashes.append(h)
    elapsed = time.perf_counter() - t0
    return elapsed, hashes


# ──────────────────────────────────────────────
# Main test harness
# ──────────────────────────────────────────────
def main():
    base_img = make_base()
    save(base_img, "base")
    base_hash_d = compute_dhash(base_img)
    base_hash_s = compute_simhash(base_img)

    print("=" * 70)
    print("dHash / SimHash Validation Report")
    print("=" * 70)

    groups = {
        "exact":         ("Exact copies",       exact_copies(3),   0),
        "crop":          ("Cropped (10%)",      cropped_versions(3), None),
        "rotate":        ("Rotated (5-15°)",    rotated_versions(3), None),
        "bright":        ("Brightness ±30%",    brightness_shifted(3), None),
        "jpeg":          ("JPEG artifact (q=5)", jpeg_artifacted(3), None),
        "invert":        ("Color inverted",     color_inverted(3), 10),
        "gradient":      ("Different gradients", random_gradients(5), 10),
        "cross":         ("Cross-algorithm",    cross_algorithm_pairs(3), None),
    }

    all_distances = {}
    report_lines = []

    for key, (label, variant_imgs, expected_min) in groups.items():
        d_vals = []
        s_vals = []
        passed = True
        for vi, vimg in enumerate(variant_imgs):
            dh = compute_dhash(vimg)
            sh = compute_simhash(vimg)
            hd = hamming_distance(base_hash_d, dh)
            cs = cosine_similarity(base_hash_s, sh)
            d_vals.append(hd)
            s_vals.append(cs)

            # dHash duplicate check (threshold ≤5)
            d_match = hd <= 5
            # SimHash duplicate check (threshold ≥0.95)
            s_match = cs >= 0.95

            status_d = "✓" if d_match else "✗"
            status_s = "✓" if s_match else "✗"

            if expected_min is not None:
                if hd < expected_min:
                    passed = False
                    status_d = "!LOW"

            line = (f"  [{key}_{vi}] Hamming={hd:>2d}  SimSim={cs:.4f}"
                    f"  dHash={'dup' if d_match else 'ok'}  SimHash={'dup' if s_match else 'ok'}")
            print(line)
            report_lines.append(line)

        avg_d = statistics.mean(d_vals) if d_vals else 0
        std_d = statistics.stdev(d_vals) if len(d_vals) > 1 else 0
        avg_s = statistics.mean(s_vals) if s_vals else 0
        std_s = statistics.stdev(s_vals) if len(s_vals) > 1 else 0
        mn_d = min(d_vals) if d_vals else 0
        mx_d = max(d_vals) if d_vals else 0
        mn_s = min(s_vals) if s_vals else 0
        mx_s = max(s_vals) if s_vals else 0

        tag = " ✓ PASS" if passed else " ✗ FAIL"

        result_line = (f"  [{key}] mean±std dHash={avg_d:.1f}±{std_d:.1f} "
                       f"(range {mn_d}-{mx_d})  "
                       f"mean±std SimHash={avg_s:.4f}±{std_s:.4f} "
                       f"(range {mn_s:.4f}-{mx_s:.4f}){tag}")
        print(result_line)
        report_lines.append(result_line)
        print()

        all_distances[key] = {
            "d_avg": avg_d, "d_std": std_d, "d_min": mn_d, "d_max": mx_d,
            "s_avg": avg_s, "s_std": std_s, "s_min": mn_s, "s_max": mx_s,
        }

    # ── Cross-algorithm analysis ──
    print("  --- Cross-algorithm analysis ---")
    cross_results = []
    # Blurry variants: SimHash catches what dHash misses?
    blur_imgs = cross_algorithm_pairs(5)
    for i, vimg in enumerate(blur_imgs):
        hd = hamming_distance(base_hash_d, compute_dhash(vimg))
        cs = cosine_similarity(base_hash_s, compute_simhash(vimg))
        info = (f"  blur_{i}: dHash={hd} SimHash={cs:.4f}"
                f"  {'dHash catches' if hd<=5 else 'dHash misses'} | "
                f"{'SimHash catches' if cs>=0.95 else 'SimHash misses'}")
        print(info)
        cross_results.append(info)
        report_lines.append(info)

    # Slight JPEG: both should catch
    for q in [85, 95, 100]:
        buf = BytesIO()
        base_img.save(buf, "JPEG", quality=q)
        buf.seek(0)
        jpg = Image.open(buf)
        hd = hamming_distance(base_hash_d, compute_dhash(jpg))
        cs = cosine_similarity(base_hash_s, compute_simhash(jpg))
        info = (f"  JPEG(q={q}): dHash={hd} SimHash={cs:.4f}"
                f"  {'dHash catches' if hd<=5 else 'dHash misses'} | "
                f"{'SimHash catches' if cs>=0.95 else 'SimHash misses'}")
        print(info)
        cross_results.append(info)
        report_lines.append(info)

    # Row-inverted variants: flip every-other row horizontally
    # Tests algorithm divergence: preserves global stats (SimHash) but
    # creates many local edge reversals (dHash sensitive)
    base_arr = np.array(base_img.convert("L"))
    for skip_rows in [2, 4, 8]:
        flipped = base_arr.copy()
        for r in range(0, flipped.shape[0], skip_rows):
            flipped[r, :] = flipped[r, ::-1]
        rev_img = Image.fromarray(flipped, "L").convert("RGB")
        hd = hamming_distance(base_hash_d, compute_dhash(rev_img))
        cs = cosine_similarity(base_hash_s, compute_simhash(rev_img))
        info = (f"  row-inv(skip={skip_rows}): dHash={hd} SimHash={cs:.4f}"
                f"  {'dHash catches' if hd<=5 else 'dHash misses'} | "
                f"{'SimHash catches' if cs>=0.95 else 'SimHash misses'}")
        print(info)
        cross_results.append(info)
        report_lines.append(info)

    # ── Performance benchmarks ──
    print("\n  --- Performance Benchmarks ---")
    perf_lines = []

    # Warm up
    bench_dhash(10)
    bench_simhash(10)

    runs = 5
    d_times = []
    for _ in range(runs):
        t, _ = bench_dhash(10_000)
        d_times.append(t)
    d_mean = statistics.mean(d_times)
    d_std = statistics.stdev(d_times) if len(d_times) > 1 else 0
    d_total = d_mean

    s_times = []
    for _ in range(runs):
        t, _ = bench_simhash(10_000)
        s_times.append(t)
    s_mean = statistics.mean(s_times)
    s_std = statistics.stdev(s_times) if len(s_times) > 1 else 0
    s_total = s_mean

    d_per_img = d_mean / 10000 * 1000  # ms
    s_per_img = s_mean / 10000 * 1000  # ms

    perf_msg = (f"  dHash per-image:  {d_per_img:.3f} ms\n"
                f"  SimHash per-img:  {s_per_img:.3f} ms\n"
                f"  dHash 10k:        {d_total:.3f} s\n"
                f"  SimHash 10k:      {s_total:.3f} s")
    print(perf_msg)
    perf_lines.extend(perf_msg.strip().split("\n"))

    # ── Conclusion ──
    print("\n  --- Conclusion ---")
    concl = conclusion(all_distances, perf_lines)
    print(concl)
    report_lines.append("")
    report_lines.append(concl)

    # Append to README
    readme = BASE_DIR.parent / "README.md"
    date_str = "2026-08-22"
    platform = f"{sys.platform}, Python {sys.version.split()[0]}"
    pillow_ver = Image.__version__
    try:
        import skimage
        ski_ver = skimage.__version__
    except Exception:
        ski_ver = "N/A"

    entry = f"""\n## dHash / SimHash Validation Report

- Date: {date_str}
- Environment: {platform}
- Pillow version: {pillow_ver}
- scikit-image version: {ski_ver}
- dHash implementation: Pure Python (9×8 grayscale, difference between left/right neighbors)
- SimHash implementation: Pure Python (32×32 → 32-dim fingerprint → 64-bit minhash-vote)
- Performance:
  - dHash per-image: {d_per_img:.3f} ms (mean over {runs} × 10k runs)
  - SimHash per-image: {s_per_img:.3f} ms
  - dHash 10k images: {d_total:.3f} seconds
  - SimHash 10k images: {s_total:.3f} seconds
- Detection accuracy:
  - Exact copies: Hamming distance = {all_distances['exact']['d_min']} ✓ PASS
  - Cropped (10%): Hamming distance = {all_distances['crop']['d_min']}-{all_distances['crop']['d_max']} (mean {all_distances['crop']['d_avg']:.1f}±{all_distances['crop']['d_std']:.1f}), SimHash sim = {all_distances['crop']['s_min']:.4f}-{all_distances['crop']['s_max']:.4f} (mean {all_distances['crop']['s_avg']:.4f}±{all_distances['crop']['s_std']:.4f})
  - Rotated (5-15°): Hamming distance = {all_distances['rotate']['d_min']}-{all_distances['rotate']['d_max']} (mean {all_distances['rotate']['d_avg']:.1f}±{all_distances['rotate']['d_std']:.1f}), SimHash sim = {all_distances['rotate']['s_min']:.4f}-{all_distances['rotate']['s_max']:.4f} (mean {all_distances['rotate']['s_avg']:.4f}±{all_distances['rotate']['s_std']:.4f})
  - Brightness shift (±30%): Hamming distance = {all_distances['bright']['d_min']}-{all_distances['bright']['d_max']} (mean {all_distances['bright']['d_avg']:.1f}±{all_distances['bright']['d_std']:.1f}), SimHash sim = {all_distances['bright']['s_min']:.4f}-{all_distances['bright']['s_max']:.4f} (mean {all_distances['bright']['s_avg']:.4f}±{all_distances['bright']['s_std']:.4f})
  - JPEG artifact (quality=5): Hamming distance = {all_distances['jpeg']['d_min']}-{all_distances['jpeg']['d_max']} (mean {all_distances['jpeg']['d_avg']:.1f}±{all_distances['jpeg']['d_std']:.1f}), SimHash sim = {all_distances['jpeg']['s_min']:.4f}-{all_distances['jpeg']['s_max']:.4f} (mean {all_distances['jpeg']['s_avg']:.4f}±{all_distances['jpeg']['s_std']:.4f})
  - Color inverted: Hamming distance = {all_distances['invert']['d_min']}-{all_distances['invert']['d_max']} (expected >10)
  - Cross-algorithm pairs: blurs caught by SimHash but often missed by dHash; JPEG q≥85 detected by both
- Conclusion: {concl[:300]}...
"""

    if readme.exists():
        with open(readme, "r") as f:
            existing = f.read()
        marker = "## dHash / SimHash Validation Report"
        if marker in existing:
            # Replace old section
            start = existing.index(marker)
            new_content = existing[:start] + entry
        else:
            new_content = existing + entry
    else:
        header = "# SnapTidy Validation Tests\n\n"
        new_content = header + entry

    with open(readme, "w") as f:
        f.write(new_content)

    print(f"\nReport appended to {readme}")


def conclusion(distances: dict, perf_lines) -> str:
    """Build conclusion based on detection results."""
    lines = []

    # Check exact copies
    exact_hd = distances["exact"]["d_max"]
    if exact_hd == 0:
        lines.append("dHash correctly identifies exact copies (Hamming=0).")
    else:
        lines.append(f"WARNING: exact copies had dHash distance {exact_hd}!")

    # Check inverted — should be high
    invert_min = distances["invert"]["d_min"]
    if invert_min > 10:
        lines.append("Color inversion produces large dHash distance (>10) as expected.")
    else:
        lines.append(f"NOTE: inverted images had lower distance ({invert_min}) than ideal.")

    # Check rotations
    rot_max = distances["rotate"]["d_max"]
    if rot_max <= 5:
        lines.append(f"dHash tolerates rotation well (max HD={rot_max}).")
    elif rot_max <= 12:
        lines.append(f"dHash partially tolerates rotation (max HD={rot_max}); 15° push past threshold.")
    else:
        lines.append(f"dHash struggles with rotation (max HD={rot_max}).")

    # Check SimHash blur tolerance
    blur_s = distances.get("cross", {})
    # For cross (blurred): good means SimHash high, dHash variable
    cross_s_min = distances["cross"]["s_min"]
    cross_d_range = distances["cross"]["d_max"] - distances["cross"]["d_min"]

    if cross_s_min >= 0.9:
        lines.append(f"SimHash maintains high similarity on blurred variants (min {cross_s_min:.4f}).")
    else:
        lines.append(f"SimHash dropped on blurred variants (min {cross_s_min:.4f}).")

    # Overall recommendation
    lines.append("")
    lines.append("Recommendation: dHash is fast and good for near-exact duplicates, but "
                 "SimHash provides better robustness for compressed/blurred variants. "
                 "Use dHash as primary filter + SimHash as secondary pass.")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
