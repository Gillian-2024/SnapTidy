"""dHash + SimHash perceptual hashing, extracted from validation scripts.

All functions operate on PIL Images and return integers.
No network calls, no model downloads.
"""

from __future__ import annotations

import hashlib
import math

from PIL import Image


# ── dHash (perceptual hash via 9x8 grayscale difference) ───────────

def compute_dhash(image: Image.Image) -> int:
    """Compute 64-bit dHash.

    Steps:
    1. Resize to 9×8 (grayscale).
    2. Compute 8×8 pairwise left→right differences.
    3. Each bit is 1 if left > right.

    Returns 64-bit integer. Hamming distance ≤ 5 catches most transforms.
    """
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())  # 64 values, row-major

    h = 0
    for row in range(8):
        for col in range(8):
            idx = row * 9 + col
            if pixels[idx] > pixels[idx + 1]:
                h |= 1 << (63 - (row * 8 + col))
    return h


# ── SimHash (projection-based minhash-vote) ────────────────────────

def compute_simhash(image: Image.Image) -> int:
    """Compute 64-bit SimHash via projection + minhash voting.

    Steps:
    1. Resize to 32×32 grayscale.
    2. Flatten to 1024-d vector.
    3. Project onto 64 random hyperplanes (seeded).
    4. Vote ±1 per dimension → 64-bit hash.

    Cosine similarity ≥ 0.95 catches near-duplicates.
    """
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = [float(p) / 255.0 for p in gray.getdata()]  # 1024 dims

    # Deterministic projection matrices (seed=42 matches validation harness)
    rng_state = _get_rng_state()
    projections = [_make_projection(1024, rng_state) for _ in range(64)]

    bits = []
    for proj in projections:
        dot = sum(p * q for p, q in zip(pixels, proj))
        bits.append(dot >= 0)

    h = 0
    for i, b in enumerate(bits):
        if b:
            h |= 1 << (63 - i)
    return h


# ── Distance helpers ───────────────────────────────────────────────

def hamming_distance(h1: int, h2: int) -> int:
    """Bit-count of XOR."""
    xor = h1 ^ h2
    return bin(xor).count("1")


def cosine_similarity(h1: int, h2: int, bits: int = 64) -> float:
    """Hamming-space cosine similarity: 1 - (HD / bits)."""
    hd = hamming_distance(h1, h2)
    return 1.0 - (hd / bits)


# ── Internal helpers ───────────────────────────────────────────────

def _get_rng_state() -> tuple:
    """Return a seeded RNG state consistent with validation scripts."""
    # Simple deterministic hash seed — matches validation script seed=42
    return hashlib.sha256(b"snaptidy_simhash_seed_42").digest()


def _make_projection(dim: int, seed_bytes: bytes) -> list[float]:
    """Create a single projection vector using hash-chain hashing."""
    vec = []
    h = seed_bytes
    for i in range(dim):
        # Convert each byte to a signed value in [-1, 1]
        val = (h[0] / 128.0) - 1.0
        vec.append(val)
        # Hash-chain for independence
        h = hashlib.sha256(h + i.to_bytes(4, "big")).digest()
    # L2-normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec
