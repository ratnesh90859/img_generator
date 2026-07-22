"""
image_generator.py
──────────────────
Generates 360° car frames using gemini-2.5-flash-image (Vertex AI).

Consistency Strategy (NEW):
────────────────────────────
The #1 cause of inconsistency is conflicting text descriptions.
When we say "car nose turned 210° to RIGHT" but the model interprets
differently, the frame is wrong and all subsequent frames drift.

NEW APPROACH — "Minimal Text, Maximum Image Reference":
1. Frame 0° (Front View): Generated with a very explicit front-view prompt.
   This becomes the MASTER IDENTITY reference for the entire sequence.

2. Every subsequent frame (10°, 20°, ...350°):
   - Send MASTER FRAME (0°) as identity anchor.
   - Send PREVIOUS FRAME (N-1) as rotation anchor.
   - Use a MINIMAL prompt that says ONLY:
       "Rotate the car 10° clockwise from the previous frame.
        The car is on a turntable. Keep identity from master frame."
   - No confusing directional text (no LEFT/RIGHT/screen directions).

3. The visual anchor chain ensures each frame is derived from the previous,
   creating a smooth continuous rotation like a real turntable video.

Why this works better:
  Gemini's vision understanding of "rotate 10° from THIS image" is far more
  reliable than text descriptions like "nose turned 210° to screen-right".
  By removing conflicting text directions, the model focuses only on the images.
"""

import logging
import time
from typing import Optional

from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import GCP_PROJECT_ID

logger = logging.getLogger(__name__)

_REGION   = "us-central1"
_MODEL_ID = "gemini-2.5-flash-image"

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT_ID,
            location=_REGION,
        )
        logger.info("Vertex AI genai client initialised (project=%s)", GCP_PROJECT_ID)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_first_frame_prompt(display_name: str, color: str) -> str:
    """
    Prompt for Frame 0° (Front View). Explicit and detailed.
    This is the most important frame — it anchors identity for all 36 frames.
    """
    return (
        f"Professional automotive studio photograph of a {display_name} in {color}.\n\n"
        f"CAMERA POSITION: Directly in front of the car, slightly elevated (15° above hood level).\n"
        f"CAR ORIENTATION: The car is facing DIRECTLY toward the camera. "
        f"Front headlights, grille, and bumper are fully visible and centered.\n"
        f"BACKGROUND: Pure seamless white studio background (#FFFFFF). "
        f"Soft ambient ground shadow visible under the tires only.\n"
        f"FRAMING: Car fills 80% of frame width. Perfectly centered horizontally and vertically.\n"
        f"LIGHTING: Even, diffused studio lighting. No harsh shadows on car body.\n"
        f"QUALITY: Photorealistic, high resolution, sharp details on headlights, grille badge, alloy wheels.\n\n"
        f"OUTPUT: Only the car on white background. No text, no watermarks, no showroom context."
    )


def _build_rotation_prompt(display_name: str, color: str, angle_deg: int, prev_angle_deg: int) -> str:
    """
    Prompt for all frames after Frame 0°.
    MINIMAL TEXT — relies on image references for consistency.
    Only tells Gemini to rotate 10° clockwise from the previous frame.
    """
    rotation_so_far = angle_deg  # degrees rotated from front view
    return (
        f"Automotive turntable studio photograph of a {display_name} in {color}.\n\n"
        f"TASK: You are given two reference images:\n"
        f"  IMAGE 1 = MASTER REFERENCE (0° front view of this exact car)\n"
        f"  IMAGE 2 = PREVIOUS FRAME ({prev_angle_deg}° turntable position)\n\n"
        f"Generate the NEXT frame at {angle_deg}° by rotating the car "
        f"EXACTLY 10° clockwise on the turntable from IMAGE 2.\n\n"
        f"STRICT RULES:\n"
        f"1. The car's paint color, headlights, grille, alloy wheels, body proportions "
        f"must EXACTLY match IMAGE 1 (master reference).\n"
        f"2. The car's rotation must be a smooth 10° step from IMAGE 2. "
        f"No sudden jumps. No mirroring. No viewpoint change.\n"
        f"3. The turntable rotates CLOCKWISE when viewed from above. "
        f"At {angle_deg}° total rotation from front, the car has turned "
        f"{rotation_so_far}° clockwise.\n"
        f"4. Keep the SAME camera position, height, and focal length as IMAGE 2.\n"
        f"5. Pure white studio background. Same lighting setup as IMAGE 2.\n\n"
        f"OUTPUT: The car at {angle_deg}° turntable position on white studio background."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Frame generator
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(3),
    before_sleep=lambda rs: logger.warning(
        "Retrying image generation (attempt %d/3)...", rs.attempt_number
    ),
)
def generate_frame(
    display_name: str,
    color: str,
    angle_deg: int,
    seed: Optional[int] = None,
    ref_bytes: Optional[bytes] = None,
    master_frame_bytes: Optional[bytes] = None,
    prev_frame_bytes: Optional[bytes] = None,
    prev_angle_deg: Optional[int] = None,
) -> bytes:
    """
    Generate a single 360° turntable frame.

    For Frame 0°: Uses explicit front-view prompt only.
    For all other frames: Uses master frame + previous frame images as
    primary references with minimal rotation instruction text.

    Returns: Raw PNG bytes from Gemini.
    """
    is_first_frame = (angle_deg == 0 or master_frame_bytes is None)

    if is_first_frame:
        prompt_text = _build_first_frame_prompt(display_name, color)
    else:
        prompt_text = _build_rotation_prompt(
            display_name, color, angle_deg,
            prev_angle_deg if prev_angle_deg is not None else angle_deg - 10
        )

    logger.info(
        "Generating frame @ %d° for %s (first=%s, master=%s, prev=%s)",
        angle_deg, display_name,
        is_first_frame,
        master_frame_bytes is not None,
        prev_frame_bytes is not None,
    )

    # Build multimodal content list:
    # Order matters — put images BEFORE the prompt so Gemini sees them first.
    contents = []

    if not is_first_frame:
        # IMAGE 1: Master reference (0° front view)
        if master_frame_bytes:
            contents.append(
                types.Part.from_bytes(data=master_frame_bytes, mime_type="image/png")
            )
            contents.append(
                "IMAGE 1 — MASTER REFERENCE (0° front view): "
                "Use this to lock car identity, paint color, headlights, grille, alloy wheels, and body shape."
            )

        # IMAGE 2: Previous frame (immediate predecessor)
        if prev_frame_bytes:
            contents.append(
                types.Part.from_bytes(data=prev_frame_bytes, mime_type="image/png")
            )
            contents.append(
                f"IMAGE 2 — PREVIOUS FRAME ({prev_angle_deg}°): "
                f"Rotate the car exactly 10° clockwise from this image. "
                f"Maintain the same camera height, distance, and lighting."
            )

    # Optional user-supplied reference image
    if ref_bytes:
        contents.append(types.Part.from_bytes(data=ref_bytes, mime_type="image/png"))
        contents.append("ADDITIONAL REFERENCE: Use for extra styling context.")

    # Main prompt (text instruction)
    contents.append(prompt_text)

    t0 = time.perf_counter()

    response = _get_client().models.generate_content(
        model=_MODEL_ID,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    elapsed = time.perf_counter() - t0

    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            logger.info(
                "  ✓ Frame @ %d° generated in %.1fs (%d bytes)",
                angle_deg, elapsed, len(part.inline_data.data)
            )
            return part.inline_data.data

    raise ValueError(
        f"Gemini returned no image for {display_name} @ {angle_deg}° "
        f"(response: {response.candidates[0].content.parts})"
    )
