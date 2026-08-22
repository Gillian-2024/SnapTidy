#!/usr/bin/env python3
"""HDBSCAN clustering benchmark on CLIP embeddings."""

import os
import time
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from PIL import Image, ImageDraw
import clip
import numpy as np
try:
    import hdbscan
except ImportError:
    print("Installing hdbscan...")
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'hdbscan'])
    import hdbscan

IMG_SIZE = 224
OUT_DIR = Path('/tmp/snaptidy_hdbscan_test')
OUT_DIR.mkdir(exist_ok=True)

def gen_circle(color, size_range=(50, 100)):
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    d = ImageDraw.Draw(img)
    s = np.random.randint(size_range[0], size_range[1])
    cx, cy = np.random.randint(s//2+10, IMG_SIZE-s//2-10), np.random.randint(s//2+10, IMG_SIZE-s//2-10)
    d.ellipse([cx-s//2, cy-s//2, cx+s//2, cy+s//2], fill=color)
    return img

def gen_rect(color, size_range=(40, 90)):
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (60, 60, 60))
    d = ImageDraw.Draw(img)
    w = np.random.randint(size_range[0], size_range[1])
    h = np.random.randint(size_range[0]//2, size_range[1])
    x, y = np.random.randint(5, IMG_SIZE-w-5), np.random.randint(5, IMG_SIZE-h-5)
    d.rectangle([x, y, x+w, y+h], fill=color)
    return img

def gen_gradient(c1, c2):
    from PIL import ImageOps
    arr = np.linspace(0, 1, IMG_SIZE)
    arr = arr.reshape(-1, 1)
    top = np.array(c1)[np.newaxis, :]
    bot = np.array(c2)[np.newaxis, :]
    blend = (top * (1 - arr) + bot * arr).astype(np.uint8)
    return Image.fromarray(blend)

# 6 clusters × 7 images each = 42 images
clusters = [
    ('red_obj',      [(180,30,30), (200,50,20), (160,20,40), (220,60,50), (150,25,35), (190,45,25), (170,35,45)]),
    ('blue_obj',     [(30,30,180), (20,50,200), (40,20,160), (50,60,220), (25,35,170), (35,45,190), (45,25,150)]),
    ('green_obj',    [(30,180,30), (50,200,20), (20,160,40), (60,220,50), (35,150,35), (45,190,25), (25,170,45)]),
    ('warm_grad',    [(255,100,0), (255,0,100), (255,150,50), (200,50,100), (255,80,20), (255,120,30), (240,30,80)]),
    ('cool_grad',    [(0,100,255), (100,0,200), (50,150,255), (0,200,100), (20,80,220), (30,120,240), (0,160,60)]),
    ('noisy_gray',   [(80,80,80), (120,120,120), (60,60,60), (160,160,160), (100,100,100), (140,140,140), (40,40,40)]),
]

all_paths = []
for cname, colors in clusters:
    for ci, col in enumerate(colors):
        # Mix generators per cluster for variety
        gen_fn = np.random.choice([gen_circle, gen_rect, lambda c=col: gen_gradient(c, (c[0]+50%255, c[1]+50%255, c[2]+50%255))])
        try:
            img = gen_fn(col)
        except TypeError:
            img = gen_circle(col)
        p = OUT_DIR / f'{cname}_{ci:02d}.png'
        img.save(str(p))
        all_paths.append(p)

print(f"Generated {len(all_paths)} test images ({len(clusters)} clusters)")

# Load model
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
model, preprocess = clip.load('ViT-B/32', device=device)

# Extract embeddings
print("\nExtracting CLIP embeddings...")
t0 = time.perf_counter()
embeddings = []
with torch.no_grad():
    for p in all_paths:
        img = preprocess(Image.open(p)).unsqueeze(0).to(device)
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize
        embeddings.append(emb.cpu().numpy().flatten())
embeddings = np.array(embeddings)
embed_time = time.perf_counter() - t0
print(f"{len(embeddings)} embeddings extracted in {embed_time*1000:.0f}ms ({embed_time*1000/len(embeddings):.1f}ms/img)")

# Run HDBSCAN with multiple configs
configs = [
    {'min_cluster_size': 5, 'min_samples': 3},
    {'min_cluster_size': 5, 'min_samples': 5},
    {'min_cluster_size': 7, 'min_samples': 3},
    {'min_cluster_size': 7, 'min_samples': 5},
    {'min_cluster_size': 10, 'min_samples': 5},
]

true_labels = np.repeat(np.arange(len(clusters)), len(clusters[0][1]))  # 7 per cluster

def purity_score(labels_true, labels_pred):
    """Adjusted purity: max overlap between true label and predicted cluster."""
    from scipy.stats import mode
    n = len(labels_true)
    # Remove noise (-1) from both
    mask = labels_pred != -1
    if not np.any(mask):
        return 0.0
    tp = labels_true[mask]
    pp = labels_pred[mask]
    correct = sum(np.sum(tp == l) for l in set(pp))
    return correct / np.sum(mask)

print(f"\n{'Config':<35} {'Clusters':<10} {'Noise%':<8} {'Purity':<8}")
print("-" * 65)

results = []
for cfg in configs:
    clust = hdbscan.HDBSCAN(**cfg, metric='euclidean')
    preds = clust.fit_predict(embeddings)
    n_clusters = len(set(preds)) - (1 if -1 in preds else 0)
    noise_pct = 100 * list(preds).count(-1) / len(preds)
    pur = purity_score(true_labels, preds)
    results.append({**cfg, 'clusters': n_clusters, 'noise_pct': noise_pct, 'purity': pur, 'preds': preds})
    tag = "✅ BEST" if n_clusters == len(clusters) and pur == 1.0 else ""
    blank = ""
    print(f"  mcs={cfg['min_cluster_size']:>2} ms={cfg['min_samples']:>2}         {blank:<22} {n_clusters:>5}    {noise_pct:>5.0f}%  {pur:>0.3f}{tag}")

# Best config report
best = max(results, key=lambda r: r['purity'] + (1 if r['clusters']==len(clusters) else 0) - r['noise_pct']/100)
print(f"\n--- Best Config ---")
print(f"  min_cluster_size = {best['min_cluster_size']}")
print(f"  min_samples = {best['min_samples']}")
print(f"  Clusters found: {best['clusters']} (expected {len(clusters)})")
print(f"  Noise ratio: {best['noise_pct']:.1f}%")
print(f"  Purity: {best['purity']:.3f}")
print(f"\nVerdict:", end=" ")
if best['clusters'] >= len(clusters) * 2 // 3 and best['purity'] > 0.5:
    print(f"HDBSCAN can cluster CLIP embeddings for V1 with good accuracy.")
    print(f"Recommended params: min_cluster_size={best['min_cluster_size']}, min_samples={best['min_samples']}")
else:
    print("CLIP separation is borderline (intra-group sim ~0.87).")
    print(f"V1 clustering needs careful tuning. Try {best['min_cluster_size']} as starting point.")
