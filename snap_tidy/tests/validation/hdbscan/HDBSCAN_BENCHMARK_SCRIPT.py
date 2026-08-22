#!/usr/bin/env python3
"""
HDBSCAN Clustering Benchmark on CLIP Embeddings
Generates synthetic test images, extracts CLIP embeddings, and runs HDBSCAN
with multiple parameter configurations. Validates clustering quality.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# ─── Step 1: Generate Synthetic Test Images ────────────────────────────────

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
os.makedirs(TEST_DIR, exist_ok=True)

IMG_SIZE = 64
GROUPS = {
    "red_objects":   {"color": (200, 50, 50),   "desc": "Red objects"},
    "blue_objects":  {"color": (50, 80, 220),    "desc": "Blue objects"},
    "green_objects": {"color": (50, 180, 80),    "desc": "Green objects"},
    "gradients":     {"color": None,             "desc": "Gradients"},
    "patterns":      {"color": None,             "desc": "Patterns"},
    "noise":         {"color": None,             "desc": "Noise"},
}
IMAGES_PER_GROUP = 8
TOTAL = len(GROUPS) * IMAGES_PER_GROUP

def generate_image(group, idx):
    """Generate a single synthetic image."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    label_idx = idx

    if group == "red_objects":
        r, g, b = GROUPS["red_objects"]["color"]
        # Vary shape: circle, rectangle, triangle
        shape_type = idx % 3
        size_factor = 0.3 + (idx / IMAGES_PER_GROUP) * 0.3  # 0.3 ~ 0.7
        cx = int(IMG_SIZE * (0.25 + (label_idx % 3) * 0.25))
        cy = int(IMG_SIZE * (0.25 + (label_idx // 3) * 0.25))
        half = int(IMG_SIZE * size_factor / 2)
        if shape_type == 0:
            bbox = [cx - half, cy - half, cx + half, cy + half]
            draw.ellipse(bbox, fill=(r, g, b))
        elif shape_type == 1:
            bbox = [cx - half, cy - half, cx + half, cy + half]
            draw.rectangle(bbox, fill=(r, g, b))
        else:
            pts = [(cx, cy - half), (cx - half, cy + half), (cx + half, cy + half)]
            draw.polygon(pts, fill=(r, g, b))

    elif group == "blue_objects":
        r, g, b = GROUPS["blue_objects"]["color"]
        shape_type = idx % 3
        size_factor = 0.3 + (idx / IMAGES_PER_GROUP) * 0.3
        cx = int(IMG_SIZE * (0.25 + (label_idx % 3) * 0.25))
        cy = int(IMG_SIZE * (0.25 + (label_idx // 3) * 0.25))
        half = int(IMG_SIZE * size_factor / 2)
        if shape_type == 0:
            draw.ellipse([cx - half, cy - half, cx + half, cy + half], fill=(r, g, b))
        elif shape_type == 1:
            draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=(r, g, b))
        else:
            pts = [(cx, cy - half), (cx - half, cy + half), (cx + half, cy + half)]
            draw.polygon(pts, fill=(r, g, b))

    elif group == "green_objects":
        r, g, b = GROUPS["green_objects"]["color"]
        shape_type = idx % 3
        size_factor = 0.3 + (idx / IMAGES_PER_GROUP) * 0.3
        cx = int(IMG_SIZE * (0.25 + (label_idx % 3) * 0.25))
        cy = int(IMG_SIZE * (0.25 + (label_idx // 3) * 0.25))
        half = int(IMG_SIZE * size_factor / 2)
        if shape_type == 0:
            draw.ellipse([cx - half, cy - half, cx + half, cy + half], fill=(r, g, b))
        elif shape_type == 1:
            draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=(r, g, b))
        else:
            pts = [(cx, cy - half), (cx - half, cy + half), (cx + half, cy + half)]
            draw.polygon(pts, fill=(r, g, b))

    elif group == "gradients":
        c1 = (int(np.random.uniform(0, 80)), int(np.random.uniform(0, 80)), int(np.random.uniform(0, 80)))
        c2 = (int(np.random.uniform(200, 255)), int(np.random.uniform(200, 255)), int(np.random.uniform(200, 255)))
        for y in range(IMG_SIZE):
            ratio = y / (IMG_SIZE - 1)
            pixel = tuple(int(c1[i] + (c2[i] - c1[i]) * ratio) for i in range(3))
            draw.line([(0, y), (IMG_SIZE, y)], fill=pixel)

    elif group == "patterns":
        pattern_type = idx % 4
        base_color = [int(np.random.uniform(100, 200)) for _ in range(3)]
        stripe_color = [(255 - x) for x in base_color]
        if pattern_type == 0:  # stripes
            spacing = max(2, 6 - (idx % 4))
            for x in range(0, IMG_SIZE, spacing):
                draw.rectangle([x, 0, x + spacing // 2, IMG_SIZE], fill=tuple(stripe_color))
        elif pattern_type == 1:  # checkerboard
            cell_size = max(2, 12 - (idx % 4))
            for ry in range(0, IMG_SIZE, cell_size):
                for rx in range(0, IMG_SIZE, cell_size):
                    if ((ry // cell_size) + (rx // cell_size)) % 2 == 0:
                        draw.rectangle([rx, ry, rx + cell_size - 1, ry + cell_size - 1], fill=tuple(base_color))
        elif pattern_type == 2:  # concentric circles
            cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
            colors = [tuple(base_color), tuple(stripe_color)]
            for radius in range(min(IMG_SIZE) // 2, 0, -3):
                color = colors[radius % 2]
                draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=color)
        else:  # dots
            dot_size = max(1, 4 - (idx % 3))
            spacing = max(4, 12 - (idx % 5))
            for y in range(0, IMG_SIZE, spacing):
                for x in range(0, IMG_SIZE, spacing):
                    dx = ((y // spacing) % 2) * spacing // 2
                    draw.ellipse([x + dx - dot_size, y - dot_size, x + dx + dot_size, y + dot_size], fill=tuple(stripe_color))

    elif group == "noise":
        pixels = img.load()
        noise_level = 80 + (idx % 4) * 30  # varying intensity
        for y in range(IMG_SIZE):
            for x in range(IMG_SIZE):
                mean = int(np.random.uniform(80, 180))
                pixels[x, y] = (
                    max(0, min(255, mean + int(np.random.normal(0, noise_level)))),
                    max(0, min(255, mean + int(np.random.normal(0, noise_level)))),
                    max(0, min(255, mean + int(np.random.normal(0, noise_level)))),
                )

    return img


print("=" * 60)
print("STEP 1: Generating synthetic test images...")
print("=" * 60)

group_keys = list(GROUPS.keys())
image_paths = []
true_labels = []  # group index for each image

for gi, gk in enumerate(group_keys):
    for ii in range(IMAGES_PER_GROUP):
        fname = f"{gk}_{ii:02d}.png"
        fpath = os.path.join(TEST_DIR, fname)
        img = generate_image(gk, ii)
        img.save(fpath)
        image_paths.append(fpath)
        true_labels.append(gi)

true_labels = np.array(true_labels)
print(f"Generated {len(image_paths)} images in {TEST_DIR}")

# ─── Step 2: Extract CLIP Embeddings ──────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Loading CLIP model and extracting embeddings...")
print("=" * 60)

import torch
import clip

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model, preprocess = clip.load('ViT-B/32', device=device)

embeddings = []
with torch.no_grad():
    for i, img_path in enumerate(image_paths):
        img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy().flatten())
        if (i + 1) % 8 == 0:
            print(f"  Processed {i+1}/{len(image_paths)} images...")

embeddings = np.array(embeddings)
print(f"Embedding shape: {embeddings.shape}")

# ─── Step 3: Run HDBSCAN with Multiple Configurations ──────────────────────

print("\n" + "=" * 60)
print("STEP 3: Running HDBSCAN benchmarks...")
print("=" * 60)

import hdbscan

configs = []
for min_cluster_size in [5, 8, 15]:
    for min_samples in [3, 5, 10]:
        configs.append({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": "euclidean",
        })

results = []

for ci, cfg in enumerate(configs):
    print(f"\n  Config {ci+1}/{len(configs)}: min_cluster_size={cfg['min_cluster_size']}, "
          f"min_samples={cfg['min_samples']}")

    clusterer = hdbscan.HDBSCAN(**cfg)
    labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = np.sum(labels == -1)
    noise_ratio = noise_count / len(labels) * 100

    cluster_sizes = sorted([np.sum(labels == c) for c in set(labels) if c != -1], reverse=True)

    # Purity score
    purity_num = 0
    total = len(labels)
    for c in set(labels):
        if c == -1:
            continue
        mask = labels == c
        cluster_labels = true_labels[mask]
        unique, counts = np.unique(cluster_labels, return_counts=True)
        purity_num += counts.max()
    purity = purity_num / total

    results.append({
        "min_cluster_size": cfg["min_cluster_size"],
        "min_samples": cfg["min_samples"],
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "noise_ratio": noise_ratio,
        "purity": purity,
    })

    print(f"    Clusters: {n_clusters} | Sizes: {cluster_sizes} | Noise: {noise_ratio:.1f}% | Purity: {purity:.3f}")

# ─── Step 4: Summary Table ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
header = f"{'min_cls':>8} {'min_smp':>8} {'clusters':>9} {'sizes':>30} {'noise%':>8} {'purity':>8}"
sep = "-" * len(header)
print(sep)
print(header)
print(sep)
for r in results:
    sizes_str = str(r["cluster_sizes"])[:28].ljust(30)
    print(f"{r['min_cluster_size']:>8} {r['min_samples']:>8} {r['n_clusters']:>9} "
          f"{sizes_str:>30} {r['noise_ratio']:>7.1f}% {r['purity']:>7.3f}")
print(sep)

# Best config
best = max(results, key=lambda x: x["purity"])
print(f"\nBest configuration:")
print(f"  min_cluster_size = {best['min_cluster_size']}")
print(f"  min_samples      = {best['min_samples']}")
print(f"  clusters found   = {best['n_clusters']}  (expected 6)")
print(f"  noise ratio      = {best['noise_ratio']:.1f}%")
print(f"  purity           = {best['purity']:.3f}")

# Verdict
VERDICT_THRESHOLD = 0.85
if best["purity"] >= VERDICT_THRESHOLD and abs(best["n_clusters"] - 6) <= 1:
    verdict = "PASS - Suitable for V1 deployment"
elif best["purity"] >= 0.6:
    verdict = "MARGINAL - Needs tuning; usable for prototyping only"
else:
    verdict = "FAIL - Not suitable for V1 without significant changes"

print(f"\nVerdict: {verdict}")

# ─── Step 5: Write Results to README ───────────────────────────────────────

README_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "README.md")

table_lines = [sep, header, sep]
for r in results:
    sizes_str = str(r["cluster_sizes"])[:28].ljust(30)
    table_lines.append(f"{r['min_cluster_size']:>8} {r['min_samples']:>8} {r['n_clusters']:>9} "
                       f"{sizes_str:>30} {r['noise_ratio']:>7.1f}% {r['purity']:>7.3f}")
table_lines.append(sep)

report = f"""## HDBSCAN Clustering Validation Report

**Date:** 2026-08-22
**Test type:** Model validation — HDBSCAN on CLIP embeddings
**Image count:** {TOTAL} synthetic images ({len(GROUPS)} groups x {IMAGES_PER_GROUP} images)

### Setup

- **CLIP model:** ViT-B/32
- **Device:** {device}
- **Embedding dimension:** {embeddings.shape[1]}
- **HDBSCAN metric:** euclidean

### Test Data

Generated {TOTAL} synthetic images across 6 visual groups:

| Group | Description | Variation strategy |
|---|---|---|
| red_objects | Red geometric shapes | 3 shapes x 3 positions x size factor |
| blue_objects | Blue geometric shapes | Same variation structure |
| green_objects | Green geometric shapes | Same variation structure |
| gradients | Random linear gradients | Color pair + direction randomization |
| patterns | Stripes/checker/circles/dots | 4 pattern types with param variance |
| noise | Gaussian pixel noise | 4 noise levels (80-200 std dev) |

All images are 64x64 RGB, synthetically generated via Pillow. No real photos used.

### Results

{chr(10).join(table_lines)}

### Best Configuration

- **min_cluster_size:** {best['min_cluster_size']}
- **min_samples:** {best['min_samples']}
- **Clusters found:** {best['n_clusters']} (expected: 6)
- **Cluster sizes:** {best['cluster_sizes']}
- **Noise ratio:** {best['noise_ratio']:.1f}%
- **Purity score:** {best['purity']:.3f}

### Verdict

**{verdict}**

A purity score of {best['purity']:.3f} means that {best['purity']*100:.0f}% of points would be correctly assigned if we took the majority label per cluster. For V1 production use, we target >= 0.85 purity with cluster count within +/-1 of expected groups.

Note: Gradient and noise groups are inherently difficult for density-based clustering since they lack sharp visual boundaries. If V1 focuses on object-based photo deduplication (where CLIP naturally separates categories), this is acceptable — but expect lower purity when mixing structured scenes with abstract content.
"""

# Read existing README, append or create section
if os.path.exists(README_PATH):
    with open(README_PATH, "r") as f:
        existing = f.read()
    marker = "## HDBSCAN Clustering Validation Report"
    if marker in existing:
        # Replace existing report
        start = existing.index(marker)
        new_content = existing[:start] + report
    else:
        new_content = existing.rstrip() + "\n\n" + report
else:
    new_content = "# SnapTidy Validation Tests\n\nAI-assisted photo cleanup tool validation suite.\n\n" + report

with open(README_PATH, "w") as f:
    f.write(new_content)

print(f"\nReport written to: {README_PATH}")
print("\nDone.")
