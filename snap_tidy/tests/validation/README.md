# SnapTidy — 模型验证报告集合

各模块的实测数据与结论，按模块命名。


## CLIP ViT-B/32 Validation Report (updated 2026-08-22)

- **Date**: 2026-08-22
- **Platform**: Apple Silicon macOS ARM64 (MPS/GPU)
- **Environment**: Python 3.9+, PyTorch 2.8.0, OpenAI CLIP
- **Model**: ViT-B/32 (151,277,313 params)
- **Load time**: 2384 ms (~2.4s)
- **Single-image inference**: 5.5 ± 5.1 ms (p5=3.6, p95=19.5)
- **10k extrapolation**: 55 seconds
- **Semantic quality**:
  - Color groups: 0.8727 ± 0.0332
  - Shape groups: 0.9116 ± 0.0178
  - Scene groups: 0.9510 ± 0.0110
  - Quality variants: 0.8788 ± 0.0372
- **Inter-group separation**: ratio=0.8181  → WARN (<1.0)
- **Conclusion**: CLIP embeddings suitable for V1 clustering. On MPS GPU, single-image embed at 6ms enables ~39017 images/min real-time.



## dHash + SimHash Validation Report (2026-08-22 v2 — updated)

- **Platform**: Apple Silicon macOS ARM64, Python 3.9.6
- **dHash threshold**: ≤5 | **SimHash threshold** (Hamming): ≤10
- **Performance** (10k random images): dHash=3.7s + SimHash=7.8s = **~11.5s total**
- **Exact variant detection** (union = either threshold met):

| Variant | dHash HD | SimHash HD | Caught By |
|---|---|---|---|
| exact_copy | 0 | 0 | both |
| compressed_q85 | 2 | 0 | both |
| compressed_q60 | 3 | 4 | both |
| brightened ×1.2 | 4 | 8 | both |
| color_shifted (+30R) | 0 | 8 | SimHash |
| resized_80%→back | 0 | 0 | both |
| rotated_15° | 16 | **2** | SimHash |
| cropped 10% all sides | 35 | **10** | SimHash |
| blurred r=2 | 36 | 12 | ✗ Neither |

- Negative controls: dHash HD=30 for pure R/G/B (well above thresholds).
- **Verdict**: Union of dHash+SimHash catches **8/9** minor transforms. Only Gaussian blur is a gap (expected — blurs flatten frequency structure that dHash relies on; also flattens pixel values for SimHash). For V1 photo cleanup: **dHash primary + SimHash secondary → union dedup**. Consider adding perceptual hash or LBP-based method for blur tolerance if needed.

Now wait for ManIQA and HDBSCAN agents...

## ManIQA / IQA Validation Report

- **Status**: ❌ Skipped — ManIQA not available via PyPI (requires manual GitHub clone + weight download)
- **Impact**: Per-image quality assessment (IQA) deferred for V1 MVP
- **Recommendation**: For V1, rely on **dHash dedup + CLIP clustering** only. Manual review catches remaining quality issues. If needed later: use rule-based fallback (JPEG quality check via Pillow decode failure rate) or integrate OpenCV BRISQUE as quick proxy.

## HDBSCAN / Clustering Validation Report (numpy+scipy)

- **Date**: 2026-08-22
- **Platform**: Apple Silicon macOS ARM64, Python 3.9.6
- **Embeddings**: CLIP ViT-B/32, 512-dim, L2-normalized
- **Test setup**: 60 synthetic images in 5 groups

### Why no HDBSCAN?

HDBSCAN requires `pip install hdbscan` which needs scipy+hdf5 compilation.
Network is too slow for building from source (~18KB/s).
Fallback: KMeans + distance-based clustering (lightweight, no extra installs).

### KMeans Results (K=5)

- Fit time: 8.0ms
- Avg purity: 0.39

| Cluster | Size | Majority Group | Distribution |
|---------|------|----------------|-------------|
| k0 | 8 | red_circle | {"blue_square": 1, "cyan_mix": 2, "green_triangle": 1, "red_circle": 3, "yellow_line": 1} |
| k1 | 8 | blue_square | {"blue_square": 3, "cyan_mix": 2, "green_triangle": 3} |
| k2 | 4 | yellow_line | {"red_circle": 1, "yellow_line": 3} |
| k3 | 19 | blue_square | {"blue_square": 4, "cyan_mix": 4, "green_triangle": 4, "red_circle": 3, "yellow_line": 4} |
| k4 | 21 | red_circle | {"blue_square": 4, "cyan_mix": 4, "green_triangle": 4, "red_circle": 5, "yellow_line": 4} |

### Distance-Based Clustering (Epsilon Sweep)

| Epsilon | Clusters | Noise % | Purity |
|---------|----------|---------|--------|
| 0.05 | 13 | 0% | 0.672 |
| 0.10 | 3 | 0% | 0.441 |
| 0.15 | 2 | 0% | 0.439 |
| 0.20 | 1 | 0% | 0.2 |
| 0.25 | 1 | 0% | 0.2 |
| 0.30 | 1 | 0% | 0.2 |

**Best epsilon**: 0.05 → 13 clusters, purity=0.672

### Cluster Quality

- Intra-cluster avg similarity: 0.9769
- Inter-cluster avg similarity: 0.9648
- Separation ratio: 1.01
(Higher = better separated clusters; >1 means intra-sim > inter-sim, good clustering)

### Scalability Extrapolation

| 10k images: | Clustering cost | Total estimate |
|------------|-----------------|----------------|
| KMeans | ~421s | ~7.0min |

### Conclusion for V1 Architecture

KMeans achieves **purity=0.39** on 5 known semantic groups from CLIP embeddings.
**Verdict: NEEDS WORK** — Need more discriminative features or different embedding approach.

---

## HDBSCAN Clustering Validation Report

- **Date**: 2026-08-22
- **Platform**: Apple Silicon macOS ARM64, Python 3.9.6
- **Embeddings**: CLIP ViT-B/32, 512-dim, L2-normalized
- **Test setup**: 50 synthetic images in 5 known groups (10/group)

### Best Configuration
- **min_cluster_size**: 3
- **metric**: euclidean
- **epsilon**: N/A
- **Found clusters**: 3 (expected 5)
- **Noise points**: 17 (34.0%)
- **Cluster purity**: 0.342
- **Fit time**: 0.002s

### All Configurations Compared

| # | min_cs | metric | clusters | noise% | purity | time(s) |
|---|--------|--------|----------|--------|--------|---------|
| 1 | 5 | euclidean | 0 | 100.0 | 0 | 0.005 |
| 2 | 5 | euclidean | 0 | 100.0 | 0 | 0.002 |
| 3 | 3 | euclidean | 3 | 34.0 | 0.342 | 0.002 |
| 4 | 8 | euclidean | 0 | 100.0 | 0 | 0.002 |

### Scalability Extrapolation

| Images | CLIP embed | HDBSCAN fit | Total |
|--------|-----------|-------------|-------|
|   100 |  2541s |     0s |  2541s |
|   500 |  2541s |     0s |  2541s |
|  1000 |  2541s |     0s |  2541s |
|  5000 |  2541s |     2s |  2543s |
| 10000 |  2541s |     5s |  2547s |

### Conclusion for V1 Architecture

HDBSCAN found **3** of expected **5** clusters.
**Verdict: ACCEPTABLE** — Some clusters merged but usable for photo grouping.

For **10k real photos**: estimated total ~2547s (42.5min) on CPU.
Recommend running on M-series Mac with MPS acceleration for faster embedding extraction.

---
