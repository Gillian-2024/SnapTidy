"""Clustering: CLIP embeddings + HDBSCAN + date buckets + prompt labels.

Combines visual similarity clustering (HDBSCAN on CLIP embeddings) with
date-based grouping to form combined keys for the final audit view.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


# ── Common prompts for cluster labeling ────────────────────────────

CLUSTER_PROMPTS = [
    "a photo of a beach scene",
    "a photo of a sunset or sunrise",
    "a photo of food or dining",
    "a photo of a city street",
    "a photo of a building or architecture",
    "a photo of nature or a garden",
    "a photo of a person or people",
    "a photo of an animal or pet",
    "a photo of a car or vehicle",
    "a photo of indoor home life",
    "a photo of a party or celebration",
    "a photo of sports or exercise",
    "a photo of artwork or painting",
    "a photo of children playing",
    "a photo of water or ocean",
    "a photo of mountains or hiking",
    "a photo of snow or winter",
    "a photo of flowers or plants",
    "a photo of a meal or dish",
    "a photo of drinks or beverages",
    "a photo of a document or text",
    "a photo of a screen or device",
    "a photo of night or dark scene",
    "a photo of fireworks or lights",
    "a photo of public transport",
    "a photo of a bridge",
    "a photo of a festival or parade",
]


def _preprocess_images(images: list[Image.Image], device: str) -> torch.Tensor:
    """Preprocess PIL images for CLIP ViT-B/32 via numpy → torch.

    Avoids a bug where torchvision.Compose + Resize produces tensors with an
    extra dimension ([N, 1, 3, 224, 224] instead of [N, 3, 224, 224]).

    Returns tensor of shape (N, 3, 224, 224).
    """
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    chunks: list[torch.Tensor] = []
    for img in images:
        rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        arr = torch.from_numpy(np.array(rgb)).to(device, dtype=torch.float32)
        # HWC [224,224,3] / 255 → CHW [3,224,224]
        arr = arr.permute(2, 0, 1) / 255.0
        arr.sub_(MEAN).div_(STD)
        chunks.append(arr)
    return torch.stack(chunks)


def compute_embeddings(
    images: list[Image.Image],
    device: str = "auto",
    batch_size: int = 32,
) -> np.ndarray:
    """Compute CLIP ViT-B/32 embeddings for a batch of PIL Images.

    Args:
        images: list of PIL Images (RGB preferred).
        device: "mps" | "cpu" | "auto" (auto picks mps if available).
        batch_size: number of images per inference batch.

    Returns:
        L2-normalized embeddings array of shape (N, 512).
    """
    if device == "auto":
        try:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # Lazy imports
    import torch

    model_clip, _ = __import__("clip").load("ViT-B/32", device=device)
    model_clip.eval()

    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = _preprocess_images(batch, device=device)
            feats = model_clip.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
) -> np.ndarray:
    """Run HDBSCAN on pre-computed L2-normalized embeddings.

    Uses euclidean metric (equivalent to cosine on normalized vectors).

    Args:
        embeddings: (N, 512) L2-normalized array.
        min_cluster_size: minimum cluster size for HDBSCAN.

    Returns:
        Label array same shape as embeddings[:, 0]. -1 = noise.
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_epsilon=0.0,
    )
    return clusterer.fit_predict(embeddings)


def date_group(photos_with_dates: list[tuple[str, str]], window: str = "week") -> dict:
    """Assign each photo to a date bucket.

    Args:
        photos_with_dates: list of (path_or_idx, date_string) where date is "YYYY-MM-DD".
        window: "day" | "week" | "month" | "year".

    Returns:
        {index_in_list: bucket_string} mapping.
    """
    groups: dict[int, str] = {}

    for idx, pair in enumerate(photos_with_dates):
        date_str = pair[1] if isinstance(pair, tuple) else pair
        if not date_str:
            groups[idx] = "unknown"
            continue

        parts = date_str.split("-")
        year = parts[0]
        month = parts[1]
        day = parts[2] if len(parts) > 2 else "01"

        if window == "year":
            groups[idx] = year
        elif window == "month":
            groups[idx] = f"{year}-{month}"
        elif window == "week":
            groups[idx] = _to_iso_week(date_str, year, month, day)
        else:  # day
            groups[idx] = date_str

    return groups


def _to_iso_week(date_str: str, year: str, month: str, day: str) -> str:
    """Convert YYYY-MM-DD to ISO week label like W2026-33."""
    from datetime import datetime
    dt = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
    iso_year, iso_week, _ = dt.isocalendar()
    return f"W{iso_year}-{iso_week:02d}"


def generate_labels(
    centroids: np.ndarray,
) -> list[str]:
    """Generate human-readable labels for clusters via CLIP prompt matching.

    For each centroid embedding, compare against preset prompts and pick top matches.

    Args:
        centroids: (K, 512) array of cluster center embeddings.

    Returns:
        List of K label strings.
    """
    import torch
    import clip as _clip  # tokenize is a module-level function

    # Load CLIP model for text encoding (needed for encode_text method)
    clip_model, _ = _clip.load("ViT-B/32", device="cpu")

    # Process prompts in batches
    batch_size = 32
    all_token_ids = []

    with torch.no_grad():
        for i in range(0, len(CLUSTER_PROMPTS), batch_size):
            batch_prompts = CLUSTER_PROMPTS[i:i + batch_size]
            tokens = _clip.tokenize(["a photo of " + p.replace("a photo of ", "") for p in batch_prompts])
            all_token_ids.append(tokens)

    full_tokens = torch.cat(all_token_ids, dim=0)

    with torch.no_grad():
        text_feats = clip_model.encode_text(full_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    labels = []
    for centroid in centroids:
        c = torch.from_numpy(centroid.astype(np.float32)).unsqueeze(0)
        sims = torch.matmul(c, text_feats.T).squeeze()
        top_indices = sims.topk(3).indices.tolist()
        caption_parts = [CLUSTER_PROMPTS[i].replace("a photo of ", "") for i in top_indices]
        labels.append(" · ".join(caption_parts))

    return labels
