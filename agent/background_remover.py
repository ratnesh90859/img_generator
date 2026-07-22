"""
background_remover.py
─────────────────────
Removes the pure white studio background from Gemini-generated car images
and outputs a transparent PNG (RGBA).

Strategy:
  1. Convert image to RGBA.
  2. BFS flood-fill from all 4 CORNERS (not edges) — corners are guaranteed
     to be background, not car. This avoids accidentally seeding into car body.
  3. Use a SAFE, LOW threshold (RGB distance ≤ 30) so we only touch
     near-white pixels. Dark wheels, glass, shadows are safe.
  4. Apply a 1-pixel alpha feather to smooth jagged edges.
  5. Output transparent PNG.

Why BFS from corners (not edges):
  Edge-seeding sometimes hits door reflections or hood highlights that
  touch the image border. Corner-seeding is much safer — corners of a
  studio photo are always pure background.
"""

import io
import logging
from collections import deque
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# Conservative threshold: only remove near-white pixels (≤ 30 distance from white).
# Dark wheels (≈ RGB 30,30,30), shadows (≈ RGB 180,180,180) are untouched.
BG_THRESHOLD = 30

# Pure white reference
_WHITE = (255, 255, 255)


def _color_distance(p: tuple) -> float:
    """Euclidean distance of a pixel from pure white (255,255,255)."""
    return ((p[0] - 255) ** 2 + (p[1] - 255) ** 2 + (p[2] - 255) ** 2) ** 0.5


def _bfs_fill_from_corners(img_rgba: Image.Image) -> Image.Image:
    """
    BFS flood-fill transparent from the 4 corners outward.
    Only pixels within BG_THRESHOLD of pure white are made transparent.
    Returns a new RGBA image.
    """
    width, height = img_rgba.size
    pixels = img_rgba.load()

    visited = [[False] * height for _ in range(width)]
    queue = deque()

    # Seed BFS from 4 corners only
    corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    for cx, cy in corners:
        r, g, b, a = pixels[cx, cy]
        if _color_distance((r, g, b)) <= BG_THRESHOLD:
            queue.append((cx, cy))
            visited[cx][cy] = True

    # 4-directional BFS
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        x, y = queue.popleft()
        r, g, b, a = pixels[x, y]
        # Make this pixel transparent
        pixels[x, y] = (r, g, b, 0)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not visited[nx][ny]:
                nr, ng, nb, na = pixels[nx, ny]
                if _color_distance((nr, ng, nb)) <= BG_THRESHOLD:
                    visited[nx][ny] = True
                    queue.append((nx, ny))

    return img_rgba


def remove_background(png_bytes: bytes, output_size: tuple = (1024, 1024)) -> bytes:
    """
    Remove the white studio background from a Gemini-generated car image.
    Outputs a transparent PNG with only the car visible.

    Strategy: BFS flood-fill from 4 corners (conservative threshold = 30).
    Car body, wheels, glass, and shadows are preserved.

    Args:
        png_bytes:   Raw PNG bytes from Gemini.
        output_size: Target (width, height) of output.

    Returns:
        Transparent PNG bytes (RGBA) — background is alpha=0.
    """
    logger.debug("Removing background from image (%d bytes input)...", len(png_bytes))

    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    # Resize to target size first
    if img.size != output_size:
        img = img.resize(output_size, Image.LANCZOS)

    # BFS transparent fill from corners
    img = _bfs_fill_from_corners(img)

    # Light alpha feather: slightly blur the alpha channel to smooth jagged edges
    r, g, b, alpha = img.split()
    alpha = alpha.filter(ImageFilter.SMOOTH)
    img = Image.merge("RGBA", (r, g, b, alpha))

    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=False)
    out_bytes = buffer.getvalue()

    logger.debug("  ✓ Background removed → %d bytes transparent PNG", len(out_bytes))
    return out_bytes


def quick_validate_alpha(png_bytes: bytes) -> dict:
    """
    Sanity check on the resulting PNG image.
    Returns {'valid': bool, 'issues': [], 'transparent_pct': float}
    """
    issues = []

    try:
        img = Image.open(io.BytesIO(png_bytes))
    except Exception as e:
        return {"valid": False, "issues": [f"Cannot open image: {e}"], "transparent_pct": 0.0}

    if img.width < 256 or img.height < 256:
        issues.append(f"Too small: {img.width}×{img.height}")

    if len(png_bytes) < 5_000:
        issues.append(f"File too small ({len(png_bytes)} bytes) — possibly corrupt")

    # Check transparency if RGBA
    transparent_pct = 0.0
    if img.mode == "RGBA":
        import numpy as np
        arr = np.array(img)
        total = arr.shape[0] * arr.shape[1]
        transparent = int((arr[:, :, 3] == 0).sum())
        transparent_pct = round(transparent / total * 100, 1)

        # If >95% of image is transparent, something went wrong
        if transparent_pct > 95:
            issues.append(f"Too much transparency ({transparent_pct}%) — BFS over-flooded")

    return {
        "valid":           len(issues) == 0,
        "issues":          issues,
        "transparent_pct": transparent_pct,
    }
