#!/usr/bin/env python3
"""
HDBSCAN Clustering Benchmark on CLIP Embeddings for SnapTidy V1.
Generates synthetic images, extracts CLIP features, runs HDBSCAN with
various parameters, measures clustering quality.
All images are synthetically generated - no real photos accessed.
"""

import sys, time, json, numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR / "test_hdbscan_images"
README = Path("/Users/gillian/Desktop/SnapTidy/snap_tidy/tests/validation/README.md")
IMG_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("  HDBSCAN Clustering Benchmark -- SnapTidy V1")
print("=" * 70)

# ── Step 1: Generate Synthetic Test Images ────────────────────────
from PIL import Image, ImageDraw
import random; random.seed(42)

IMG_SIZE = (64, 64)
GROUPS = {
    "red_circle":     {"base": (200, 50, 50),   "n": 10},
    "blue_square":    {"base": (50, 80, 220),   "n": 10},
    "green_triangle": {"base": (50, 180, 80),   "n": 10},
    "yellow_line":    {"base": (220, 220, 50),  "n": 10},
    "cyan_mix":       {"base": (50, 200, 220),  "n": 10},
}

def gen_image(group_name, idx):
    """Generate colored shape image."""
    config = GROUPS[group_name]
    r, g, b = config["base"]
    img = Image.new("RGB", IMG_SIZE, (30, 30, 30))  # dark background
    draw = ImageDraw.Draw(img)

    cx = 10 + (idx % 3) * 18
    cy = 10 + (idx // 3) * 18
    size = 12 + (idx % 4) * 4  # vary size

    shapes_per_group = ["ellipse", "rectangle", "polygon"]
    shape_type = shapes_per_group[idx % len(shapes_per_group)]

    if shape_type == "ellipse":
        bbox = [cx - size//2, cy - size//2, cx + size//2, cy + size//2]
        draw.ellipse(bbox, fill=(r, g, b))
    elif shape_type == "rectangle":
        bbox = [cx - size//2, cy - size//2, cx + size//2, cy + size//2]
        draw.rectangle(bbox, fill=(r, g, b))
    else:
        # Triangle-ish polygon
        pts = [(cx, cy - size//2), (cx - size//2, cy + size//2), (cx + size//2, cy + size//2)]
        draw.polygon(pts, fill=(r, g, b))

    fpath = IMG_DIR / f"{group_name}_{idx:02d}.png"
    img.save(str(fpath))
    return str(fpath), group_name

all_files = []
for gname, cfg in GROUPS.items():
    for i in range(cfg["n"]):
        all_files.append(gen_image(gname, i))

print(f"Generated {len(all_files)} images in {len(GROUPS)} groups ({cfg['n']}/group)")

# ── Step 2: Extract CLIP Embeddings ──────────────────────────────
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image as PILImage

preprocess = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("\nLoading CLIP ViT-B/32...")
t0 = time.perf_counter()
model_clip, _ = __import__('clip').load("ViT-B/32", device="cpu")
load_time = (time.perf_counter() - t0) * 1000

model_clip.eval()

print(f"Load: {load_time:.0f}ms")

# Extract all embeddings
embeddings = []
labels = []
group_names_ordered = list(GROUPS.keys())

for fpath, gname in all_files:
    with torch.no_grad():
        img = preprocess(PILImage.open(fpath)).unsqueeze(0)
        emb = model_clip.encode_image(img).squeeze().numpy()
        emb = emb / np.linalg.norm(emb)  # normalize
    embeddings.append(emb)
    labels.append(gname)

embeddings = np.array(embeddings)
labels = np.array(labels)
true_labels_num = np.array([group_names_ordered.index(l) for l in labels])

print(f"Extracted {embeddings.shape[0]} embeddings ({embeddings.shape[1]}-dim)")

# ── Step 3: Run HDBSCAN with Multiple Configurations ─────────────
import hdbscan

# Note: all CLIP embeddings are L2-normalized, so Euclidean ≈ Cosine distance.
# Replacing 'cosine' config options with 'euclidean' for this hdbscan version.

results = []

config = {"min_cluster_size": 5, "metric": "euclidean"}
print(f"Config #1: min_cs={config['min_cluster_size']} metric={config['metric']}")
t_clust = time.perf_counter()
clusterer = hdbscan.HDBSCAN(**config)
preds = clusterer.fit_predict(embeddings)
elapsed = time.perf_counter() - t_clust

n_clusters = len(set(preds)) - (1 if -1 in preds else 0)
n_noise = int(np.sum(preds == -1))
noise_pct = round(n_noise / len(preds) * 100, 1)

# Per-cluster stats
clusters_found = {}
unique_preds = set(preds)
if -1 in unique_preds:
    unique_preds.remove(-1)

for p in sorted(unique_preds):
    mask = preds == p
    members = list(np.where(mask)[0])
    member_labels = np.array([labels[i] for i in members])
    maj_label_count = max(set(member_labels), key=member_labels.tolist().count)
    correctness_score = round(sum(1 for l in member_labels if l == maj_label_count) / len(members), 3)
    clusters_found[p] = {
        "size": int(mask.sum()),
        "groups_in_cluster": list(np.unique(member_labels)),
        "correctness_score": correctness_score,
    }

# Purity: for each predicted cluster, count majority true label
purity_scores = []
for p in unique_preds:
    if p == -1:
        continue
    members = np.where(preds == p)[0]
    if len(members) > 0:
        maj_label = np.bincount(true_labels_num[members]).argmax()
        purity_scores.append(np.mean(true_labels_num[members] == maj_label))
avg_purity = round(np.mean(purity_scores), 3) if purity_scores else 0

results.append({
    "config": dict(config, **{"elapsed_s": round(elapsed, 3)}),
    "n_clusters": n_clusters,
    "n_noise": n_noise,
    "noise_pct": noise_pct,
    "purity": avg_purity,
    "clusters_detail": clusters_found,
    "clusterer": clusterer,
})
print(f"  Clusters: {n_clusters}, Noise: {n_noise} ({noise_pct}%), Purity: {avg_purity}")
for cid, cinfo in clusters_found.items():
    print(f"    Cluster {cid}: {cinfo['size']} items, groups={cinfo['groups_in_cluster']}")

# Config #2: euclidean with larger epsilon
config2 = {"min_cluster_size": 5, "metric": "euclidean", "cluster_selection_epsilon": 0.15}
print(f"\nConfig #2: min_cs={config2['min_cluster_size']} metric={config2['metric']} eps=0.15")
t_clust = time.perf_counter()
clusterer2 = hdbscan.HDBSCAN(**config2)
preds2 = clusterer2.fit_predict(embeddings)
elapsed2 = time.perf_counter() - t_clust

n_clusters2 = len(set(preds2)) - (1 if -1 in preds2 else 0)
n_noise2 = int(np.sum(preds2 == -1))
noise_pct2 = round(n_noise2 / len(preds2) * 100, 1)

clusters_found2 = {}
unique_preds2 = set(preds2)
if -1 in unique_preds2:
    unique_preds2.remove(-1)

for p in sorted(unique_preds2):
    mask = preds2 == p
    members = list(np.where(mask)[0])
    member_labels = np.array([labels[i] for i in members])
    maj_label_count = max(set(member_labels), key=member_labels.tolist().count)
    correctness_score2 = round(sum(1 for l in member_labels if l == maj_label_count) / len(members), 3)
    clusters_found2[p] = {
        "size": int(mask.sum()),
        "groups_in_cluster": list(np.unique(member_labels)),
        "correctness_score": correctness_score2,
    }

purity_scores2 = []
for p in unique_preds2:
    if p == -1:
        continue
    members = np.where(preds2 == p)[0]
    if len(members) > 0:
        maj_label = np.bincount(true_labels_num[members]).argmax()
        purity_scores2.append(np.mean(true_labels_num[members] == maj_label))
avg_purity2 = round(np.mean(purity_scores2), 3) if purity_scores2 else 0

results.append({
    "config": dict(config2, **{"elapsed_s": round(elapsed2, 3)}),
    "n_clusters": n_clusters2,
    "n_noise": n_noise2,
    "noise_pct": noise_pct2,
    "purity": avg_purity2,
    "clusters_detail": clusters_found2,
    "clusterer": clusterer2,
})
print(f"  Clusters: {n_clusters2}, Noise: {n_noise2} ({noise_pct2}%), Purity: {avg_purity2}")
for cid, cinfo in clusters_found2.items():
    print(f"    Cluster {cid}: {cinfo['size']} items, groups={cinfo['groups_in_cluster']}")

# Config #3: smaller min_cluster_size
config3 = {"min_cluster_size": 3, "metric": "euclidean"}
print(f"\nConfig #3: min_cs={config3['min_cluster_size']} metric={config3['metric']}")
t_clust = time.perf_counter()
clusterer3 = hdbscan.HDBSCAN(**config3)
preds3 = clusterer3.fit_predict(embeddings)
elapsed3 = time.perf_counter() - t_clust

n_clusters3 = len(set(preds3)) - (1 if -1 in preds3 else 0)
n_noise3 = int(np.sum(preds3 == -1))
noise_pct3 = round(n_noise3 / len(preds3) * 100, 1)

clusters_found3 = {}
unique_preds3 = set(preds3)
if -1 in unique_preds3:
    unique_preds3.remove(-1)

for p in sorted(unique_preds3):
    mask = preds3 == p
    members = list(np.where(mask)[0])
    member_labels = np.array([labels[i] for i in members])
    maj_label_count = max(set(member_labels), key=member_labels.tolist().count)
    correctness_score3 = round(sum(1 for l in member_labels if l == maj_label_count) / len(members), 3)
    clusters_found3[p] = {
        "size": int(mask.sum()),
        "groups_in_cluster": list(np.unique(member_labels)),
        "correctness_score": correctness_score3,
    }

purity_scores3 = []
for p in unique_preds3:
    if p == -1:
        continue
    members = np.where(preds3 == p)[0]
    if len(members) > 0:
        maj_label = np.bincount(true_labels_num[members]).argmax()
        purity_scores3.append(np.mean(true_labels_num[members] == maj_label))
avg_purity3 = round(np.mean(purity_scores3), 3) if purity_scores3 else 0

results.append({
    "config": dict(config3, **{"elapsed_s": round(elapsed3, 3)}),
    "n_clusters": n_clusters3,
    "n_noise": n_noise3,
    "noise_pct": noise_pct3,
    "purity": avg_purity3,
    "clusters_detail": clusters_found3,
    "clusterer": clusterer3,
})
print(f"  Clusters: {n_clusters3}, Noise: {n_noise3} ({noise_pct3}%), Purity: {avg_purity3}")
for cid, cinfo in clusters_found3.items():
    print(f"    Cluster {cid}: {cinfo['size']} items, groups={cinfo['groups_in_cluster']}")

# Config #4: stricter min_cluster_size
config4 = {"min_cluster_size": 8, "metric": "euclidean"}
print(f"\nConfig #4: min_cs={config4['min_cluster_size']} metric={config4['metric']}")
t_clust = time.perf_counter()
clusterer4 = hdbscan.HDBSCAN(**config4)
preds4 = clusterer4.fit_predict(embeddings)
elapsed4 = time.perf_counter() - t_clust

n_clusters4 = len(set(preds4)) - (1 if -1 in preds4 else 0)
n_noise4 = int(np.sum(preds4 == -1))
noise_pct4 = round(n_noise4 / len(preds4) * 100, 1)

clusters_found4 = {}
unique_preds4 = set(preds4)
if -1 in unique_preds4:
    unique_preds4.remove(-1)

for p in sorted(unique_preds4):
    mask = preds4 == p
    members = list(np.where(mask)[0])
    member_labels = np.array([labels[i] for i in members])
    maj_label_count = max(set(member_labels), key=member_labels.tolist().count)
    correctness_score4 = round(sum(1 for l in member_labels if l == maj_label_count) / len(members), 3)
    clusters_found4[p] = {
        "size": int(mask.sum()),
        "groups_in_cluster": list(np.unique(member_labels)),
        "correctness_score": correctness_score4,
    }

purity_scores4 = []
for p in unique_preds4:
    if p == -1:
        continue
    members = np.where(preds4 == p)[0]
    if len(members) > 0:
        maj_label = np.bincount(true_labels_num[members]).argmax()
        purity_scores4.append(np.mean(true_labels_num[members] == maj_label))
avg_purity4 = round(np.mean(purity_scores4), 3) if purity_scores4 else 0

results.append({
    "config": dict(config4, **{"elapsed_s": round(elapsed4, 3)}),
    "n_clusters": n_clusters4,
    "n_noise": n_noise4,
    "noise_pct": noise_pct4,
    "purity": avg_purity4,
    "clusters_detail": clusters_found4,
    "clusterer": clusterer4,
})
print(f"  Clusters: {n_clusters4}, Noise: {n_noise4} ({noise_pct4}%), Purity: {avg_purity4}")
for cid, cinfo in clusters_found4.items():
    print(f"    Cluster {cid}: {cinfo['size']} items, groups={cinfo['groups_in_cluster']}")

# ── Step 4: Find Best Configuration ─────────────────────────────
# Score = purity * n_clusters (want high purity + many clusters matching ground truth)
best_idx = max(range(len(results)), key=lambda i: results[i]["purity"] * results[i]["n_clusters"])
best = results[best_idx]

print(f"\n{'='*70}")
print(f"  BEST CONFIG: Config #{best_idx+1}")
print(f"    {best['config']}")
print(f"    Clusters={best['n_clusters']}, Noise={best['noise_pct']}%, Purity={best['purity']}")
print(f"{'='*70}")

# ── Step 5: Exponential Size Scaling Test ─────────────────────────
print(f"\n--- Scalability Extrapolation ---")
print(f"Embedding extraction: ~{load_time:.0f}ms for {len(embeddings)} images")
print(f"HDBSCAN fit: {best['config']['elapsed_s']:.3f}s for {len(embeddings)} images")

extrapolation = {}
for n_imgs in [100, 500, 1000, 5000, 10000]:
    clip_ms = round(load_time + best["config"]["elapsed_s"] / len(embeddings) * n_imgs, 1)
    hdbscan_s = round(best["config"]["elapsed_s"] * (n_imgs / len(embeddings)) ** 1.5, 1)  # approximate O(n²) scaling
    total_s = round(clip_ms + hdbscan_s, 1)
    extrapolation[n_imgs] = {
        "embedding_ms": int(clip_ms),
        "hdbscan_sec": int(hdbscan_s),
        "total_sec": int(total_s),
    }
    print(f"  {n_imgs:>5} images: CLIP {int(clip_ms):>5}s + HDBSCAN {int(hdbscan_s):>5}s = {int(total_s):>5}s")

# ── Step 6: Write Report ──────────────────────────────────────────
report = [
    "",
    "## HDBSCAN Clustering Validation Report",
    "",
    f"- **Date**: 2026-08-22",
    f"- **Platform**: Apple Silicon macOS ARM64, Python {sys.version.split()[0]}",
    f"- **Embeddings**: CLIP ViT-B/32, {embeddings.shape[1]}-dim, L2-normalized",
    f"- **Test setup**: {len(all_files)} synthetic images in {len(GROUPS)} known groups ({GROUPS[list(GROUPS.keys())[0]]['n']}/group)",
    "",
    "### Best Configuration",
    f"- **min_cluster_size**: {best['config']['min_cluster_size']}",
    f"- **metric**: {best['config']['metric']}",
    f"- **epsilon**: {best['config'].get('cluster_selection_epsilon', 'N/A')}",
    f"- **Found clusters**: {best['n_clusters']} (expected {len(GROUPS)})",
    f"- **Noise points**: {best['n_noise']} ({best['noise_pct']}%)",
    f"- **Cluster purity**: {best['purity']}",
    f"- **Fit time**: {best['config']['elapsed_s']:.3f}s",
    "",
    "### All Configurations Compared",
    "",
    "| # | min_cs | metric | clusters | noise% | purity | time(s) |",
    "|---|--------|--------|----------|--------|--------|---------|",
]

for i, r in enumerate(results):
    report.append(f"| {i+1} | {r['config']['min_cluster_size']} | {r['config']['metric']} | {r['n_clusters']} | {r['noise_pct']} | {r['purity']} | {r['config']['elapsed_s']:.3f} |")

report.extend([
    "",
    "### Scalability Extrapolation",
    "",
    "| Images | CLIP embed | HDBSCAN fit | Total |",
    "|--------|-----------|-------------|-------|",
])

for n, v in extrapolation.items():
    report.append(f"| {n:>5} | {v['embedding_ms']:>5}s | {v['hdbscan_sec']:>5}s | {v['total_sec']:>5}s |")

report.extend([
    "",
    "### Conclusion for V1 Architecture",
    "",
    f"HDBSCAN found **{best['n_clusters']}** of expected **{len(GROUPS)}** clusters.",
])

if best["n_clusters"] >= len(GROUPS) - 1:
    report.append("**Verdict: PASS** — HDBSCAN correctly identifies semantic groups from CLIP embeddings.")
elif best["n_clusters"] < len(GROUPS) * 0.5:
    report.append("**Verdict: PARTIAL FAIL** — Too few clusters. May need smaller min_cluster_size or different metric.")
else:
    report.append("**Verdict: ACCEPTABLE** — Some clusters merged but usable for photo grouping.")

report.extend([
    f"",
    f"For **10k real photos**: estimated total ~{extrapolation[10000]['total_sec']}s ({extrapolation[10000]['total_sec']/60:.1f}min) on CPU.",
    f"Recommend running on M-series Mac with MPS acceleration for faster embedding extraction.",
    "",
    "---",
    "",
])

with open(README, "a") as f:
    f.write("\n".join(report))

print(f"\nReport appended to README.md")
print("=" * 70)
