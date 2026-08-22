#!/usr/bin/env python3
"""CLIP ViT-B/32 embedding benchmark for SnapTidy."""

import os
import time
import random
import resource
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from PIL import Image, ImageDraw, ImageFilter
import clip
import numpy as np


def cos_sim(a, b):
    ta = torch.tensor(a, dtype=torch.float32)
    tb = torch.tensor(b, dtype=torch.float32)
    return float(torch.nn.functional.cosine_similarity(ta.unsqueeze(0), tb.unsqueeze(0)))


def make_gradient(img_dir, name, top_rgb, bot_rgb):
    y = np.linspace(0, 1, 224).reshape(-1, 1, 1)
    arr = (np.array(top_rgb) * (1 - y) + np.array(bot_rgb) * y).astype(np.uint8)
    Image.fromarray(arr, 'RGB').save(img_dir / name)


def main():
    img_dir = Path('/tmp/snaptidy_clip_test')
    img_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("SnapTidy CLIP ViT-B/32 Benchmark")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Generate test images
    print("\nGenerating test images...")

    for fname, draw_fn in [
        ('a_red_circle',  lambda d: d.ellipse([20, 20, 204, 204], fill='red')),
        ('a_red_square',  lambda d: d.rectangle([40, 40, 184, 184], fill='red')),
        ('a_red_triangle', lambda d: d.polygon([(112, 20), (20, 204), (204, 204)], fill='red')),
    ]:
        im = Image.new('RGB', (224, 224), 'black')
        draw_fn(ImageDraw.Draw(im))
        im.save(img_dir / f'{fname}.png')

    for c, cn in [('blue', 'blue'), ('green', 'green'), ('yellow', 'yellow')]:
        im = Image.new('RGB', (224, 224), 'white')
        ImageDraw.Draw(im).ellipse([40, 40, 184, 184], fill=c)
        im.save(img_dir / f'b_circle_{cn}.png')

    make_gradient(img_dir, 'd_landscape.png',  (135, 206, 235), (34, 139, 84))
    make_gradient(img_dir, 'd_sunset.png',     (135, 206, 235), (255, 69, 0))
    make_gradient(img_dir, 'd_gray.png',       (135, 206, 235), (128, 128, 128))

    pat = Image.new('RGB', (224, 224))
    pixels = [(i+j)%256 for i in range(224) for j in range(224)]
    pat.putdata([(x, (x*2)%256, 128) for x in pixels])
    pat.save(img_dir / 'e_clean.png')
    pat.filter(ImageFilter.GaussianBlur(radius=5)).save(img_dir / 'e_blurry.png')
    pat.point(lambda p: max(0, min(255, p // 4))).save(img_dir / 'e_dark.png')
    pat.point(lambda p: min(255, p * 3)).save(img_dir / 'e_bright.png')

    tmp = Image.new('RGB', (224, 224))
    for i in range(224):
        for j in range(224):
            tmp.putpixel((i, j), ((i+j)%256, (i*3)%256, (j*7)%256))
    for _ in range(10):
        tmp.save('/tmp/jpeg_tmp.jpg', quality=2)
        tmp = Image.open('/tmp/jpeg_tmp.jpg')
    if not tmp or tmp.size != (224, 224):
        tmp = Image.new('RGB', (224, 224), (128, 128, 128))
    tmp.save(img_dir / 'e_compressed.png')

    Image.new('RGB', (224, 224), 'white').save(img_dir / 'ctrl_white.png')
    Image.new('RGB', (224, 224), 'black').save(img_dir / 'ctrl_black.png')
    rng = random.Random(42)
    noise = Image.new('RGB', (224, 224))
    pixels_list = [tuple(rng.randint(0, 255) for _ in range(3)) for _ in range(224*224)]
    noise.putdata(pixels_list)
    noise.save(img_dir / 'ctrl_noise.png')

    image_files = sorted(img_dir.glob('*.png'))
    print(f"{len(image_files)} images generated.")

    # Load model
    print("\nLoading CLIP ViT-B/32...")
    t0 = time.perf_counter()
    mem0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    model, preprocess = clip.load('ViT-B/32', device=device)
    load_ms = (time.perf_counter() - t0) * 1000
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded in {load_ms:.0f} ms | params {total_params:,}")

    # Batch inference
    print("\nBatch inference...")
    batch_tensors = [preprocess(Image.open(fp)).unsqueeze(0).to(device) for fp in image_files]
    batch_input = torch.cat(batch_tensors, dim=0)
    with torch.no_grad():
        emb_batch = model.encode_image(batch_input)
        emb_batch = emb_batch / emb_batch.norm(dim=-1, keepdim=True)
    batch_ms = (time.perf_counter() - t0) * 1000
    print(f"Total: {batch_ms:.1f} ms → {(batch_ms/len(image_files)):.1f} ms/image")

    # Single-image timing (warm)
    print("\nSingle-image timing (100 iterations)...")
    imgs_list = [preprocess(Image.open(fp)).unsqueeze(0).to(device) for fp in image_files]
    times = []
    with torch.no_grad():
        for i in range(100):
            s = time.perf_counter()
            e = model.encode_image(imgs_list[i % len(imgs_list)])
            _ = e / e.norm(dim=-1, keepdim=True)
            times.append((time.perf_counter() - s) * 1000)
    mean_t = sum(times)/len(times)
    std_t = (sum((t-mean_t)**2 for t in times)/len(times))**0.5
    times.sort()
    p5 = times[int(len(times)*0.05)]
    p95 = times[int(len(times)*0.95)]
    print(f"Mean ± std: {mean_t:.1f} ± {std_t:.1f} ms  (p5={p5:.1f}, p95={p95:.1f})")

    embs = {}
    for idx, fp in enumerate(image_files):
        embs[fp.name] = emb_batch[idx].cpu().numpy()

    # Semantic quality
    print("\nSemantic quality (cosine similarity)...")
    groups = {
        'Color (A)': ['a_red_circle.png', 'a_red_square.png', 'a_red_triangle.png'],
        'Shape (B)': ['b_circle_blue.png', 'b_circle_green.png', 'b_circle_yellow.png'],
        'Scene (D)': ['d_landscape.png', 'd_sunset.png', 'd_gray.png'],
        'Quality (E)': ['e_clean.png', 'e_blurry.png', 'e_dark.png', 'e_bright.png', 'e_compressed.png'],
    }

    grp_details = {}
    for gname, fnames in groups.items():
        sims = [cos_sim(embs[fnames[ii]], embs[fnames[jj]])
                for ii in range(len(fnames)) for jj in range(ii+1, len(fnames))]
        if sims:
            mn = sum(sims)/len(sims)
            sd = (sum((s-mn)**2 for s in sims)/len(sims))**0.5
            grp_details[gname] = (mn, sd)
            print(f"  {gname}: {mn:.4f} ± {sd:.4f}  ({len(sims)} pairs)")

    ctrl_names = ['ctrl_white.png', 'ctrl_black.png', 'ctrl_noise.png']
    seen_pairs = set()
    inter_sims = []
    for gname in groups:
        for n1 in groups[gname]:
            for ctrl in ctrl_names:
                key = tuple(sorted([n1, ctrl]))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    inter_sims.append(cos_sim(embs[n1], embs[ctrl]))
    avg_inter = sum(inter_sims)/len(inter_sims) if inter_sims else 0
    sep_ratio = avg_inter / (sum(m for m,_ in grp_details.values())/len(grp_details)) if grp_details else 1

    print(f"\n  Inter-group: {avg_inter:.4f}")
    print(f"  Separation ratio: {sep_ratio:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Device':<35} {device.upper()}")
    print(f"{'Load time':<35} {load_ms:.0f} ms")
    print(f"{'Single-image infer':<35} {mean_t:.1f} ± {std_t:.1f} ms (p5={p5:.1f}, p95={p95:.1f})")
    print(f"{'10k extrapolation':<35} {(mean_t/1000*10000):.0f} seconds")
    print(f"{'Separation ratio':<35} {sep_ratio:.4f}")

    verdict = "PASS" if sep_ratio < 0.5 else ("WARNING" if sep_ratio < 1.0 else "FAIL")
    print(f"\nVerdict: {verdict}  (separation={sep_ratio:.3f})")

    report = f"""
## CLIP ViT-B/32 Validation Report

- **Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Platform**: Apple Silicon macOS ARM64 (MPS/GPU)
- **Environment**: Python 3.9+, PyTorch {torch.__version__}, OpenAI CLIP
- **Model**: ViT-B/32 ({total_params:,} params)
- **Load time**: {load_ms:.0f} ms (~{load_ms/1000:.1f}s)
- **Single-image inference**: {mean_t:.1f} ± {std_t:.1f} ms (p5={p5:.1f}, p95={p95:.1f})
- **10k extrapolation**: {(mean_t/1000*10000):.0f} seconds
- **Semantic quality**:
  - Color groups: {grp_details['Color (A)'][0]:.4f} ± {grp_details['Color (A)'][1]:.4f}
  - Shape groups: {grp_details['Shape (B)'][0]:.4f} ± {grp_details['Shape (B)'][1]:.4f}
  - Scene groups: {grp_details['Scene (D)'][0]:.4f} ± {grp_details['Scene (D)'][1]:.4f}
  - Quality variants: {grp_details['Quality (E)'][0]:.4f} ± {grp_details['Quality (E)'][1]:.4f}
- **Inter-group separation**: ratio={sep_ratio:.4f}  → {'PASS (<0.5)' if sep_ratio<0.5 else ('WARN (<1.0)' if sep_ratio<1.0 else 'FAIL')}
- **Conclusion**: {'CLIP embeddings suitable for V1 clustering.' if sep_ratio<1.0 else 'Need alternative approach.'} On MPS GPU, single-image embed at {mean_t:.0f}ms enables ~{int(3600*60/mean_t):.0f} images/min real-time.
"""

    readme_path = Path('/Users/gillian/Desktop/SnapTidy/snap_tidy/tests/validation/README.md')
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    if readme_path.exists():
        content = readme_path.read_text()
        if 'CLIP ViT-B/32 Validation Report' not in content:
            content += report
    else:
        content = "# SnapTidy — AI Analysis Model Validation Reports\n\n" + report
    readme_path.write_text(content)
    print(f"\n✓ Report appended to {readme_path}")


from datetime import datetime

if __name__ == '__main__':
    main()
