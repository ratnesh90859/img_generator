"""
background_remover.py
─────────────────────
Removes the studio background from Gemini-generated car images
using pure Pillow (BFS flood-fill from image edges).

Why NOT rembg: rembg calls sys.exit(1) when onnxruntime backend is
missing in this environment, which kills the background thread.

This uses a two-pass approach:
  1. Sample edge pixels to AUTO-DETECT the background color
     (works for both white AND dark/navy backgrounds from Gemini)
  2. BFS flood-fill from all 4 edges using color-distance threshold
  3. Convert filled pixels to transparent (alpha = 0)
  4. Smooth alpha channel edges
  5. Crop → scale → center on a clean 1024x1024 canvas

Works for all Gemini studio images regardless of background color.
"""

import io
import logging
from collections import deque, Counter
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# Max color distance from background color to be considered "background"
COLOR_DISTANCE_THRESHOLD = 45


def _color_distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two RGB colors."""
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2) ** 0.5


def _detect_bg_color(pixels, width: int, height: int) -> tuple:
    """
    Sample all 4 edges and find the most common color cluster.
    Returns (R, G, B) of the dominant background color.
    """
    edge_colors = []
    step = 4  # sample every 4th pixel for speed

    for x in range(0, width, step):
        r,g,b,a = pixels[x, 0]
        edge_colors.append((r,g,b))
        r,g,b,a = pixels[x, height-1]
        edge_colors.append((r,g,b))
    for y in range(0, height, step):
        r,g,b,a = pixels[0, y]
        edge_colors.append((r,g,b))
        r,g,b,a = pixels[width-1, y]
        edge_colors.append((r,g,b))

    # Quantize to buckets of 32 to find dominant cluster
    bucketed = [(r//32*32, g//32*32, b//32*32) for r,g,b in edge_colors]
    most_common = Counter(bucketed).most_common(1)[0][0]
    logger.debug("  Detected background color bucket: RGB%s", most_common)
    return most_common


def remove_background(png_bytes: bytes, output_size: tuple = (1024, 1024)) -> bytes:
    """
    Remove the studio background from a Gemini car image.
    Auto-detects whether the background is white, dark, or any color.

    Args:
        png_bytes:   Raw PNG bytes from Gemini.
        output_size: Final (width, height) of output image.

    Returns:
        WebP bytes with RGBA (transparent background).
    """
    logger.debug("Removing background (%d bytes PNG)...", len(png_bytes))

    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    width, height = img.size
    pixels = img.load()

    # ── Step 1: Auto-detect background color ─────────────────────
    bg_color = _detect_bg_color(pixels, width, height)
    logger.debug("  BG color detected: %s", bg_color)

    # ── Step 2: BFS flood-fill from all 4 edges ───────────────────
    visited = [[False] * height for _ in range(width)]
    queue   = deque()

    seeds = (
        [(x, 0) for x in range(width)]            +  # top edge
        [(x, height - 1) for x in range(width)]   +  # bottom edge
        [(0, y) for y in range(height)]            +  # left edge
        [(width - 1, y) for y in range(height)]       # right edge
    )

    for (x, y) in seeds:
        if not visited[x][y]:
            r, g, b, a = pixels[x, y]
            if _color_distance((r,g,b), bg_color) < COLOR_DISTANCE_THRESHOLD:
                visited[x][y] = True
                queue.append((x, y))

    # BFS: mark all reachable background pixels as transparent
    while queue:
        x, y = queue.popleft()
        r, g, b, a = pixels[x, y]
        pixels[x, y] = (r, g, b, 0)   # fully transparent

        for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            if 0 <= nx < width and 0 <= ny < height:
                if not visited[nx][ny]:
                    r2, g2, b2, a2 = pixels[nx, ny]
                    if _color_distance((r2,g2,b2), bg_color) < COLOR_DISTANCE_THRESHOLD:
                        visited[nx][ny] = True
                        queue.append((nx, ny))

    # ── Step 3: Smooth alpha channel edges ────────────────────────
    r_ch, g_ch, b_ch, a_ch = img.split()
    a_ch = a_ch.filter(ImageFilter.SMOOTH_MORE)
    img  = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_ch))

    # ── Step 4: Crop, Scale, and Center ──────────────────────────
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        target_max = 900
        ratio = min(target_max / img.width, target_max / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

        final_img = Image.new("RGBA", output_size, (0, 0, 0, 0))
        offset_x = (output_size[0] - new_size[0]) // 2
        offset_y = (output_size[1] - new_size[1]) // 2 + 30
        final_img.paste(img, (offset_x, offset_y))
        img = final_img

    # ── Step 5: Export as lossless WebP ──────────────────────────
    buffer = io.BytesIO()
    img.save(buffer, format="WEBP", lossless=True, quality=100)
    webp_bytes = buffer.getvalue()

    logger.debug("  ✓ Background removed → %d bytes WebP (RGBA)", len(webp_bytes))
    return webp_bytes



def quick_validate_alpha(webp_bytes: bytes) -> dict:
    """
    Sanity check on the resulting WebP after background removal.
    Returns {'valid': bool, 'issues': [str], 'transparent_pct': float}
    """
    issues = []

    try:
        img = Image.open(io.BytesIO(webp_bytes))
    except Exception as e:
        return {"valid": False, "issues": [f"Cannot open image: {e}"]}

    if img.mode != "RGBA":
        issues.append(f"Wrong mode: {img.mode} (expected RGBA)")

    if img.width < 256 or img.height < 256:
        issues.append(f"Too small: {img.width}×{img.height}")

    transparent_pct = 0.0
    if img.mode == "RGBA":
        alpha_data     = list(img.split()[3].getdata())
        total_px       = len(alpha_data)
        transparent_px = sum(1 for p in alpha_data if p < 10)
        transparent_pct = transparent_px / total_px * 100

        if transparent_pct > 95:
            issues.append(f"Image is {transparent_pct:.1f}% transparent — likely blank/empty")

    if len(webp_bytes) < 5_000:
        issues.append(f"File too small ({len(webp_bytes)} bytes) — possibly corrupt")

    return {
        "valid":           len(issues) == 0,
        "issues":          issues,
        "transparent_pct": round(transparent_pct, 1),
    }
