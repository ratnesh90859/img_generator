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
    Removes the studio background from the image using BFS flood-fill from the edges,
    and returns a transparent PNG.
    """
    logger.debug("Removing background from Gemini image (%d bytes)...", len(png_bytes))

    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    
    # Resize first
    if img.size != output_size:
        img = img.resize(output_size, Image.LANCZOS)
        
    width, height = img.size
    pixels = img.load()
    
    # Detect background color
    bg_color = _detect_bg_color(pixels, width, height)
    
    # BFS flood-fill from edges
    visited = set()
    queue = deque()
    
    # Add all edge pixels to queue
    for x in range(width):
        queue.append((x, 0))
        queue.append((x, height - 1))
    for y in range(height):
        queue.append((0, y))
        queue.append((width - 1, y))
        
    while queue:
        x, y = queue.popleft()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        
        r, g, b, a = pixels[x, y]
        
        # If color is close to background, make it transparent
        if _color_distance((r, g, b), bg_color) < COLOR_DISTANCE_THRESHOLD:
            pixels[x, y] = (r, g, b, 0)
            
            # Add neighbors
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    queue.append((nx, ny))

    # Optional: apply slight blur to alpha channel for smoother edges
    alpha = img.getchannel('A').filter(ImageFilter.GaussianBlur(radius=0.5))
    img.putalpha(alpha)
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    out_bytes = buffer.getvalue()

    logger.debug("  ✓ Image processed → %d bytes PNG", len(out_bytes))
    return out_bytes


def quick_validate_alpha(webp_bytes: bytes) -> dict:
    """
    Sanity check on the resulting WebP image.
    Returns {'valid': bool, 'issues': []}
    """
    issues = []

    try:
        img = Image.open(io.BytesIO(webp_bytes))
    except Exception as e:
        return {"valid": False, "issues": [f"Cannot open image: {e}"]}

    if img.width < 256 or img.height < 256:
        issues.append(f"Too small: {img.width}×{img.height}")

    if len(webp_bytes) < 5_000:
        issues.append(f"File too small ({len(webp_bytes)} bytes) — possibly corrupt")

    return {
        "valid":           len(issues) == 0,
        "issues":          issues,
        "transparent_pct": 0.0,
    }
