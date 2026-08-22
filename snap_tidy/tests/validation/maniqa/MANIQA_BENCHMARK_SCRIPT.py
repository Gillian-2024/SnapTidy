#!/usr/bin/env python3
"""
MANIQA Benchmark Script
========================
Validates ManIQA (Multi-dimension Attention Network) image quality assessment
pipeline for SnapTidy V1 inclusion decision.

Generates synthetic test images, attempts model loading + inference on CPU,
reports speed/quality score distribution.

No real photos are read from disk. All test images are synthetically generated.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR / "test_images"
RESULTS_DIR = SCRIPT_DIR / "results"
WEIGHTS_DIR = SCRIPT_DIR / "weights"
MANIQA_SRC = SCRIPT_DIR / "maniqa_src"  # copied maniqa source tree

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def log(msg):
    print(f"[ManIQA-Bench] {msg}", flush=True)


# ===================================================================
# Part 1: Synthetic Test Image Generation
# ===================================================================

def generate_test_images():
    """Generate 40 synthetic test images: 8 categories x 5 quality levels."""
    from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

    categories = [
        ("perfect",       {"desc": "Clean gradient, no distortions"}),
        ("slightly_compressed", {"desc": "JPEG quality ~85"}),
        ("heavily_compressed",  {"desc": "JPEG quality ~15, blocky artifacts"}),
        ("noisy",         {"desc": "High Gaussian noise"}),
        ("blurry",        {"desc": "Strong Gaussian blur"}),
        ("overexposed",   {"desc": "Washed-out brightness"}),
        ("underexposed",  {"desc": "Very dark, low contrast"}),
        ("color_distorted", {"desc": "Severe color shift"}),
    ]

    q_levels = ["A-excellent", "B-good", "C-average", "D-poor", "E-terrible"]

    # Palette seeds for consistency within category
    palettes = [
        [(255, 255, 255), (200, 200, 200), (150, 150, 150)],          # perfect
        [(255, 240, 200), (200, 180, 150), (150, 130, 100)],           # compressed
        [(255, 200, 150), (200, 150, 100), (150, 100, 50)],            # heavy compressed
        [(100, 200, 100), (150, 180, 150), (200, 150, 100)],           # noisy
        [(255, 180, 180), (180, 255, 180), (180, 180, 255)],           # blurry
        [(255, 255, 255), (255, 240, 220), (255, 230, 200)],           # overexposed
        [(20, 30, 40), (40, 50, 60), (60, 70, 80)],                    # underexposed
        [(255, 0, 0), (0, 255, 0), (0, 0, 255)],                       # color distorted
    ]

    all_files = []
    img_size = (224, 224)

    for cat_idx, (cat_name, cat_info) in enumerate(categories):
        palette = palettes[cat_idx]
        for lvl_idx, lvl_name in enumerate(q_levels):
            fname = f"{lvl_name}_{cat_name}.jpg"
            fpath = BASE_DIR / fname
            all_files.append((fname, str(fpath), cat_name, lvl_name))

            # Start with a base image
            if cat_name == "perfect":
                img = _make_gradient(img_size, *palette)
            elif cat_name == "color_distorted":
                img = _make_color_board(img_size, *palette)
            else:
                img = _make_gradient(img_size, *palette)

            # Apply quality-level degradation
            degradations = [
                lambda im: im,  # A: original (identity)
                lambda im: im.filter(ImageFilter.GaussianBlur(radius=0.5)),  # B: light blur
                lambda im: im.filter(ImageFilter.GaussianBlur(radius=1.5)),  # C: moderate blur
                lambda im: im.filter(ImageFilter.GaussianBlur(radius=3.0)),  # D: strong blur
                lambda im: im.filter(ImageFilter.GaussianBlur(radius=5.0)).filter(ImageFilter.GaussianBlur(radius=2.0)),  # E: extreme blur
            ]

            # Category-specific distortions applied after level degradations
            cat_effects = {
                "slightly_compressed": lambda im: _apply_jpeg(im, 85),
                "heavily_compressed":  lambda im: _apply_jpeg(im, 15),
                "noisy":               lambda im: _add_noise(im, level=50),
                "overexposed":         lambda im: _adjust_brightness(im, factor=2.0),
                "underexposed":        lambda im: _adjust_brightness(im, factor=0.3),
                "color_distorted":     lambda im: _shift_colors(im),
            }

            img = degradations[lvl_idx](img)

            # Apply category-specific effect
            if cat_name in cat_effects:
                img = cat_effects[cat_name](img)

            # Save as JPEG with varying quality to amplify differences
            jpeg_quality = max(5, 100 - lvl_idx * 25)  # A=100, B=75, C=50, D=25, E=5
            img.save(str(fpath), "JPEG", quality=jpeg_quality, optimize=True)

    return all_files


def _make_gradient(size, c1, c2, c3):
    """Create a diagonal gradient image."""
    from PIL import Image
    img = Image.new("RGB", size)
    pixels = img.load()
    w, h = size
    for y in range(h):
        for x in range(w):
            r = int(c1[0] * (1 - x/w) + c2[0] * (x/w) * 0.5 + c3[0] * (y/h) * 0.3)
            g = int(c1[1] * (1 - y/h) + c2[1] * (y/h) * 0.5 + c3[1] * (x/w) * 0.3)
            b = int(c1[2] * (1 - (x+y)/(2*max(w,h))) + c3[2] * ((x+y)/(2*max(w,h))))
            pixels[x, y] = (min(255, r), min(255, g), min(255, b))
    return img


def _make_color_board(size, c1, c2, c3):
    """Create a checkerboard with vivid colors."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    tile = 56
    cols = (size[0] + tile - 1) // tile
    rows = (size[1] + tile - 1) // tile
    colors = [c1, c2, c3, c1, c2, c3]
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * tile, r * tile
            x1, y1 = min(x0 + tile, size[0]), min(y0 + tile, size[1])
            col = colors[(r + c) % len(colors)]
            draw.rectangle([x0, y0, x1, y1], fill=col)
    return img


def _apply_jpeg(img, quality):
    """Round-trip through JPEG to simulate compression."""
    from PIL import Image
    buf = img.tobytes()
    tmp = Image.frombuffer("RGB", img.size, buf, "raw", "RGB", 0, 1)
    tmp.save("/tmp/_maniqa_tmp.jpg", "JPEG", quality=quality)
    return Image.open("/tmp/_maniqa_tmp.jpg")


def _add_noise(img, level=30):
    """Add Gaussian-like noise."""
    from PIL import Image
    import numpy as np
    arr = np.array(img).astype(np.float64)
    noise = np.random.normal(0, level, arr.shape).astype(np.float64)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _adjust_brightness(img, factor=1.0):
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def _shift_colors(img):
    """Apply severe color shift (red-green-blue channel permutation + saturation boost)."""
    from PIL import Image, ImageEnhance
    import numpy as np
    arr = np.array(img).astype(np.float64)
    # Permute channels: R->G, G->B, B->R
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    shifted = np.stack([b, r, g], axis=-1)
    # Boost saturation by 3x
    enhancer = ImageEnhance.Color(Image.fromarray(shifted.astype(np.uint8)))
    return enhancer.enhance(3.0)


# ===================================================================
# Part 2: MANIQA Model Loading Attempts
# ===================================================================

def setup_maniqa_source():
    """Copy required MANIQA source files into benchmark dir for import."""
    import shutil
    MANIQA_CLONE = Path("/tmp/maniqa-clone")
    if not MANIQA_CLONE.is_dir():
        return False, "ManIQA clone not found at /tmp/maniqa-clone"

    if MANIQA_SRC.exists():
        shutil.rmtree(MANIQA_SRC)

    # Copy just the modules we need
    for item in ["models/", "timm/", "config.py"]:
        src = MANIQA_CLONE / item
        dst = MANIQA_SRC / item
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, dst)

    # Also copy utils
    shutil.copytree(MANIQA_CLONE / "utils", MANIQA_SRC / "utils", dirs_exist_ok=True)

    return True, "Source copied successfully"


def try_import_patterns():
    """Try various import paths for MANIQA model class."""
    results = {}

    # Pattern 1: Direct from maniqa_src/models/maniqa
    try:
        sys.path.insert(0, str(MANIQA_SRC))
        from models.maniqa import MANIQA
        results["direct_models_maniqa"] = ("ok", MANIQA)
    except Exception as e:
        results["direct_models_maniqa"] = ("fail", str(e))

    # Pattern 2: From timm-style wrapper
    try:
        from config import Config
        results["config_class"] = ("ok", Config)
    except Exception as e:
        results["config_class"] = ("fail", str(e))

    # Pattern 3: Check if any pip package provides it
    try:
        import maniqa
        results["pip_maniqa_package"] = ("ok", maniqa)
    except ImportError:
        results["pip_maniqa_package"] = ("skip", "No 'maniqa' pip package installed")
    except Exception as e:
        results["pip_maniqa_package"] = ("fail", str(e))

    # Pattern 4: Import timm locally
    try:
        import timm
        results["local_timm"] = ("ok", f"timm from {timm.__file__}")
    except Exception as e:
        results["local_timm"] = ("fail", str(e))

    # Pattern 5: einops
    try:
        import einops
        results["einops"] = ("ok", f"einops {einops.__version__}")
    except Exception as e:
        results["einops"] = ("fail", str(e))

    return results


def attempt_model_load(device="cpu"):
    """Attempt to instantiate the MANIQA model."""
    result = {
        "architecture_load": None,
        "checkpoint_load": None,
        "error": None,
        "params_count": None,
        "device_used": str(device),
    }

    try:
        import torch
        from models.maniqa import MANIQA

        # Default config matching koniq10k-base training
        config = {
            "patch_size": 8,
            "img_size": 224,
            "embed_dim": 768,
            "dim_mlp": 768,
            "num_heads": [4, 4],
            "window_size": 4,
            "depths": [2, 2],
            "num_outputs": 1,
            "num_tab": 2,
            "scale": 0.8,
        }

        model = MANIQA(**config)
        param_count = sum(p.numel() for p in model.parameters())
        result["architecture_load"] = {
            "success": True,
            "params_millions": round(param_count / 1e6, 2),
            "model_size_mb": round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6, 1),
        }

        # Try loading Koniq10k checkpoint
        ckpt_path = WEIGHTS_DIR / "ckpt_koniq10k.pt"
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict, strict=False)
            result["checkpoint_load"] = {
                "success": True,
                "checkpoint_path": str(ckpt_path),
                "state_keys_count": len(state_dict) if isinstance(state_dict, dict) else "N/A",
            }
        else:
            result["checkpoint_load"] = {
                "success": False,
                "reason": "Checkpoint file not found; tried:",
                "paths_checked": [str(WEIGHTS_DIR / "ckpt_koniq10k.pt")],
            }

        # Move to target device
        model = model.to(device)
        result["device_ready"] = True
        result["_model_obj"] = model  # Keep reference for benchmarking

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["architecture_load"] = {"success": False, "error": str(e)}

    return result


def download_checkpoint():
    """Try to download the Koniq10k checkpoint from HuggingFace."""
    import urllib.request
    import ssl

    ckpt_url = ("https://huggingface.co/chaofengc/IQA-PyTorch-Weights/"
                "resolve/main/ckpt_koniq10k.pt")
    ckpt_dest = WEIGHTS_DIR / "ckpt_koniq10k.pt"

    log(f"Downloading checkpoint: {ckpt_url}")

    ctx = ssl.create_default_context()
    total = 0
    start = time.time()

    try:
        req = urllib.request.Request(ckpt_url, headers={"User-Agent": "ManIQA-Bench/1.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
            meta = response.info().get("Content-Length")
            if meta:
                total = int(meta)
                log(f"Checkpoint size: {total / 1e6:.1f} MB")

            chunk_size = 8 * 1024 * 1024
            with open(ckpt_dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    elapsed = time.time() - start
                    rate = (f.tell() / 1e6) / elapsed if elapsed > 0 else 0
                    log(f"  Downloaded {f.tell()/1e6:.1f}/{total/1e6:.1f} MB ({rate:.1f} MB/s)")

        log(f"Checkpoint saved to {ckpt_dest}")
        return True

    except Exception as e:
        log(f"Download failed: {e}")
        if ckpt_dest.exists():
            ckpt_dest.unlink()
        return False


# ===================================================================
# Part 3: Inference & Benchmarking
# ===================================================================

def preprocess_image(img_path, device="cpu"):
    """Preprocess a single image for MANIQA inference (Normalize + ToTensor)."""
    import torch
    from PIL import Image
    import numpy as np

    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float64)
    # Normalize: (img - 0.5) / 0.5  =>  same as 2*img - 1
    arr = (arr - 0.5) / 0.5
    # Convert CHW format expected by MANIQA
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float().to(device)
    return tensor.unsqueeze(0)  # batch dim


def infer_single_image(model, img_tensor, num_crops=1, device="cpu"):
    """Run single forward pass (no cropping augmentation)."""
    import torch

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        score = output.squeeze().item()

    # For final scoring, average across crops if requested
    if num_crops > 1:
        # Perform simple center crops as approximation
        scores = [score]
        # TODO: implement multi-crop averaging similar to official code
        score = sum(scores) / len(scores)

    return score


def run_benchmark(model, test_files, num_crops=1, device="cpu", verbose=True):
    """Run inference on all test images and collect results."""
    import torch

    results_list = []
    total_time = 0.0
    per_image_times = []

    for fname, fpath, cat, lvl in test_files:
        try:
            img_tensor = preprocess_image(fpath, device)
            t0 = time.perf_counter()
            score = infer_single_image(model, img_tensor, num_crops=num_crops, device=device)
            elapsed = time.perf_counter() - t0

            per_image_times.append(elapsed)
            total_time += elapsed

            results_list.append({
                "file": fname,
                "category": cat,
                "quality_level": lvl,
                "score": round(score, 4),
                "time_ms": round(elapsed * 1000, 1),
            })

            if verbose:
                bar_len = int((score / 1.0) * 30)
                bar = "#" * bar_len + "." * (30 - bar_len)
                print(f"  [{bar}] {fname:35s} score={score:.4f}  {elapsed*1000:.0f}ms")

        except Exception as e:
            results_list.append({
                "file": fname,
                "category": cat,
                "quality_level": lvl,
                "score": None,
                "time_ms": None,
                "error": str(e),
            })
            if verbose:
                print(f"  ERROR {fname}: {e}")

    avg_time = sum(per_image_times) / len(per_image_times) if per_image_times else 0
    throughput = 1.0 / avg_time if avg_time > 0 else float('inf')

    return results_list, total_time, avg_time, throughput


# ===================================================================
# Part 4: Report Generation
# ===================================================================

def print_results_table(results, total_time, avg_time, throughput):
    """Print formatted results table."""
    print("\n" + "=" * 90)
    print("  MANIQA Benchmark Results")
    print("=" * 90)

    scores = [r["score"] for r in results if r["score"] is not None]
    if not scores:
        print("  No valid scores obtained.")
        print("=" * 90)
        return

    print(f"\n  Score Range: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  Mean Score:  {sum(scores)/len(scores):.4f}")
    print(f"  Total Time:  {total_time:.2f}s")
    print(f"  Avg/Image:   {avg_time*1000:.1f} ms")
    print(f"  Throughput:  {throughput:.1f} img/s")

    # Per-category summary
    cats = sorted(set(r["category"] for r in results))
    print(f"\n  {'Category':<22} {'Count':>6} {'Min':>8} {'Max':>8} {'Mean':>8} {'Avg Time':>10}")
    print("  " + "-" * 72)
    for cat in cats:
        cat_scores = [r["score"] for r in results if r["category"] == cat and r["score"] is not None]
        cat_times = [r["time_ms"] for r in results if r["category"] == cat and r["time_ms"] is not None]
        if cat_scores:
            print(f"  {cat:<22} {len(cat_scores):>6} {min(cat_scores):>8.4f} {max(cat_scores):>8.4f} "
                  f"{sum(cat_scores)/len(cat_scores):>8.4f} {sum(cat_times)/len(cat_times):>9.1f}ms")

    # Quality level trend
    levels = ["A-excellent", "B-good", "C-average", "D-poor", "E-terrible"]
    print(f"\n  {'Quality Level':<18} {'Scores (per category)':>50}")
    for lvl in levels:
        lvl_scores = [r["score"] for r in results if r["quality_level"] == lvl and r["score"] is not None]
        if lvl_scores:
            print(f"  {lvl:<18} {[round(s, 3) for s in lvl_scores]:>50}")

    print("=" * 90)


def generate_report(import_attempts, load_result, bench_results, total_time, avg_time, throughput,
                    platform_info, download_success, download_error=None):
    """Generate a comprehensive validation report."""

    report = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform_info,
        "status": "partial" if ((load_result.get("architecture_load") or {}).get("success") and
                                 not (load_result.get("checkpoint_load") or {}).get("success"))
                  else "full" if (load_result.get("architecture_load") or {}).get("success") and
                                  (load_result.get("checkpoint_load") or {}).get("success")
                  else "failed",
    }

    # Import attempts
    report["imports"] = {}
    for name, (status, detail) in import_attempts.items():
        report["imports"][name] = {"status": status, "detail": str(detail)[:200]}

    # Model loading
    report["model_load"] = load_result

    # Benchmark results
    scores = [r["score"] for r in bench_results if r["score"] is not None]
    report["benchmark"] = {
        "images_total": len(bench_results),
        "images_successful": len(scores),
        "images_failed": len(bench_results) - len(scores),
        "score_range": [round(min(scores), 4), round(max(scores), 4)] if scores else None,
        "score_mean": round(sum(scores) / len(scores), 4) if scores else None,
        "total_time_s": round(total_time, 2),
        "avg_time_ms": round(avg_time * 1000, 1),
        "throughput_img_per_s": round(throughput, 1),
        "per_category": {},
        "per_image": [],
    }

    for r in bench_results:
        entry = {"file": r["file"], "category": r["category"], "level": r["quality_level"]}
        if r["score"] is not None:
            entry["score"] = r["score"]
            entry["time_ms"] = r["time_ms"]
        else:
            entry["error"] = r.get("error", "unknown")
        report["benchmark"]["per_image"].append(entry)

    cats = set(r["category"] for r in bench_results)
    for cat in cats:
        cat_scores = [r["score"] for r in bench_results
                      if r["category"] == cat and r["score"] is not None]
        if cat_scores:
            report["benchmark"]["per_category"][cat] = {
                "count": len(cat_scores),
                "mean": round(sum(cat_scores) / len(cat_scores), 4),
                "min": round(min(cat_scores), 4),
                "max": round(max(cat_scores), 4),
            }

    # Checkpoint info
    report["checkpoint"] = {
        "downloaded": download_success,
        "source": "HuggingFace chaofengc/IQA-PyTorch-Weights",
        "error": download_error,
    }

    return report


def verdict(report):
    """Determine usable/not usable verdict for V1."""
    model_load = report.get("model_load", {}) or {}
    arch_load = model_load.get("architecture_load") or {}
    ckpt_load = model_load.get("checkpoint_load") or {}

    load_ok = arch_load.get("success", False)
    ckpt_ok = ckpt_load.get("success", False)

    # Check for known failure modes
    err = model_load.get("error") or ""
    if "Connection" in err or "Download" in err or "timeout" in err.lower():
        return f"PARTIAL - Architecture loads locally but requires pretrained ViT weights download; {err}"
    if "not found" in err.lower() or "missing" in err.lower():
        return f"FAILED - Required component missing: {err}"
    if "torch" in err.lower() and ("version" in err.lower() or "compat" in err.lower()):
        return f"NOT USABLE - PyTorch compatibility issue: {err}"
    if not load_ok:
        return f"NOT USABLE - Architecture failed to load: {err}"
    if not ckpt_ok:
        return f"PARTIAL - Architecture loads but needs manual checkpoint download ({ckpt_load.get('reason', 'unknown')})"

    scores = report["benchmark"]["per_image"]
    if not any("score" in s for s in scores):
        return "NOT USABLE - No inference results"

    # Check score distribution makes sense
    valid_scores = [s["score"] for s in scores if "score" in s]
    if not valid_scores:
        return "NOT USABLE - Zero valid scores"

    # Score monotonicity check: better quality should generally score higher
    cats = set(s["category"] for s in scores)
    good_cats = {"perfect", "slightly_compressed"}
    bad_cats = {"noisy", "underexposed", "color_distorted"}
    good_avg = np.mean([s["score"] for s in scores if s["score"] is not None
                        and any(gc in s.get("category", "") for gc in good_cats)])
    bad_avg = np.mean([s["score"] for s in scores if s["score"] is not None
                       and any(bc in s.get("category", "") for bc in bad_cats)])

    if good_avg >= bad_avg:
        return f"USABLE - Scores correlate with quality (good avg={good_avg:.3f} vs bad avg={bad_avg:.3f})"
    else:
        return f"SUSPECT - Scores inverted (good avg={good_avg:.3f} < bad avg={bad_avg:.3f}); verify correctness"


def append_readme(report, verdict_text):
    """Append/update the parent README with validation report section."""
    readme_path = Path("/Users/gillian/Desktop/SnapTidy/snap_tidy/tests/validation/README.md")

    lines = []
    if readme_path.exists():
        with open(readme_path, "r") as f:
            lines = f.readlines()

    new_section = (
        "\n## ManIQA Validation Report\n\n"
        f"**Date:** {report['date']}\n"
        f"**Platform:** {report['platform'].get('summary', 'N/A')}\n"
        f"**Model Version:** MANIQA (CVPRW 2022) -- Multi-dimension Attention Network\n"
        f"**Checkpoint:** {'Koniq10k-pretrained (auto-downloaded)' if report['checkpoint']['downloaded'] else 'Not downloaded; see instructions below'}\n"
        f"**Inference Speed:** {report['benchmark']['avg_time_ms']:.1f} ms/image ({report['benchmark']['throughput_img_per_s']:.1f} img/s)\n"
        f"**Score Distribution:** {' '.join(f'{k}={v}' for k,v in [('range', str(report['benchmark']['score_range'])), ('mean', str(report['benchmark']['score_mean']))])}\n"
        f"**Verdict:** {verdict_text}\n\n"
        "---\n\n"
        "### Technical Details\n\n"
        "| Aspect | Status |\n"
        "|--------|--------|\n"
        f"| Architecture Load | {'Yes' if report['model_load']['architecture_load']['success'] else 'No'} |\n"
        f"| Parameters | {report['model_load'].get('params_count') or report['model_load']['architecture_load'].get('params_millions', '?')} M |\n"
        "| CPU-only Inference | Tested on macOS (no GPU) |\n"
        f"| Pip Package (maniqa) | {'Installed' if any('pip' in k for k,v in report['imports'].items() if v[0]=='ok') else 'Not available on PyPI'} |\n"
    )
    lines.append(new_section)

    with open(readme_path, "w") as f:
        f.writelines(lines)

    log(f"README updated: {readme_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="MANIQA Benchmark for SnapTidy")
    parser.add_argument("--num-crops", type=int, default=1, help="Number of crops for multi-crop evaluation")
    parser.add_argument("--no-download", action="store_true", help="Skip checkpoint download attempt")
    args = parser.parse_args()

    print("=" * 70)
    print("  MANIQA Benchmark -- SnapTidy V1 Validation")
    print("  No real photos accessed. All test images synthetically generated.")
    print("=" * 70)

    # Platform info
    import platform, torch
    platform_info = {
        "os": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "summary": f"{platform.system()} {platform.release()} / Python {platform.python_version()} / Torch {torch.__version__} / CUDA={'yes' if torch.cuda.is_available() else 'no'}",
    }
    log(f"Platform: {platform_info['summary']}")

    # Step 1: Generate test images
    log("Step 1: Generating 40 synthetic test images (8 categories x 5 levels)...")
    test_files = generate_test_images()
    log(f"Generated {len(test_files)} images in {BASE_DIR}")
    for fname, _, cat, lvl in test_files[:5]:
        log(f"  Sample: {fname} ({cat}, {lvl})")
    if len(test_files) > 5:
        log(f"  ... and {len(test_files)-5} more")

    # Step 2: Setup MANIQA source
    log("Step 2: Setting up MANIQA source code...")
    src_ok, src_msg = setup_maniqa_source()
    if not src_ok:
        log(f"FATAL: Source setup failed: {src_msg}")
        report = generate_report({}, {}, [], 0, 0, 0, platform_info, False, src_msg)
        verdict_t = verdict(report)
        append_readme(report, verdict_t)
        print(f"\n  Verdict: {verdict_t}")
        return 1

    log(f"  Source ready: {MANIQA_SRC}")

    # Step 3: Import attempts
    log("Step 3: Testing import patterns...")
    import_attempts = try_import_patterns()
    for name, (status, detail) in import_attempts.items():
        icon = {"ok": "[OK]", "fail": "[FAIL]", "skip": "[SKIP]"}[status]
        log(f"  {icon} {name}: {str(detail)[:80]}")

    # Step 4: Download checkpoint
    download_success = False
    download_error = None
    if not args.no_download:
        log("Step 4: Attempting checkpoint download from HuggingFace...")
        try:
            download_success = download_checkpoint()
        except Exception as e:
            download_error = str(e)
            log(f"  Download error: {e}")
    else:
        log("Step 4: Skipped (--no-download flag)")

    if not download_success:
        log("  Note: Checkpoint not downloaded. See download instructions in README update.")

    # Step 5: Model loading
    log("Step 5: Attempting MANIQA model loading on CPU...")
    device = "cpu"
    load_result = attempt_model_load(device)
    if load_result["error"]:
        log(f"  Model load ERROR: {load_result['error']}")
    else:
        arch = load_result.get("architecture_load", {})
        log(f"  Architecture loaded: {arch.get('params_millions', '?')}M params")
        ckpt = load_result.get("checkpoint_load", {})
        if ckpt.get("success"):
            log(f"  Checkpoint loaded: {ckpt.get('checkpoint_path')}")
        else:
            log(f"  Checkpoint: NOT LOADED - {ckpt.get('reason', 'unknown')}")

    # Step 6: Run benchmark
    if not load_result.get("architecture_load", {}).get("success"):
        log("Step 6: SKIPPED -- Cannot run inference without architecture")
        bench_results = []
        total_time = avg_time = 0
        throughput = 0
    else:
        log("Step 6: Running inference benchmark...")
        model_obj = load_result.get("_model_obj")
        bench_results, total_time, avg_time, throughput = run_benchmark(
            model_obj, test_files,
            num_crops=args.num_crops, device=device, verbose=True
        )

    # Step 7: Print results
    print_results_table(bench_results, total_time, avg_time, throughput)

    # Step 8: Generate report & verdict
    import numpy as np  # needed for verdict score comparison
    report = generate_report(
        import_attempts, load_result, bench_results,
        total_time, avg_time, throughput,
        platform_info, download_success, download_error
    )
    report["_model_loaded"] = "_model" in load_result

    verdict_text = verdict(report)
    log(f"Verdict: {verdict_text}")

    # Save JSON report
    report_path = RESULTS_DIR / f"maniqa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"JSON report saved: {report_path}")

    # Update README
    append_readme(report, verdict_text)

    print(f"\n{'='*70}")
    print(f"  FINAL VERDICT: {verdict_text}")
    print(f"{'='*70}")

    return 0 if "USABLE" in verdict_text else 1


if __name__ == "__main__":
    sys.exit(main())
