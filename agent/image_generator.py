"""
image_generator.py
──────────────────
Generates 360° car frames using gemini-2.5-flash-image (Vertex AI).

Prompt Strategy (User-defined template):
─────────────────────────────────────────
Uses a fixed, minimal studio prompt with explicit angle degrees.

Key consistency rules baked into every prompt:
  • Fixed camera: eye-level, 50mm lens, medium distance — NEVER changes
  • Omnidirectional flat studio lighting — same shadows/highlights on every frame
  • Explicit angle in degrees — Gemini maps this to a specific viewpoint
  • Negative prompt embedded — prevents interior, lens shift, lighting changes

Speed Strategy:
─────────────────
Since prompts are self-contained (no prev_frame image chaining), ALL frames
can be generated in parallel using ThreadPoolExecutor. Pipeline handles this.
Expected time: ~2-3 min for 36 frames vs 50 min sequential.
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
# Prompt Builder — exact user-defined template
# ─────────────────────────────────────────────────────────────────────────────

# What to AVOID in every frame (embedded as hard constraints in the positive prompt)
_NEGATIVE_CONSTRAINTS = (
    "Do NOT show: interior, uneven lighting, changing camera height, "
    "distorted proportions, background objects, different focal length, "
    "motion blur, overexposed highlights, underexposed shadows."
)


def build_frame_prompt(
    display_name: str,
    color: str,
    angle_deg: int,
    prev_angle_deg: Optional[int] = None,
) -> str:
    """
    Builds the exact prompt template per the user spec.
    When prev_angle_deg is given, the prompt also instructs Gemini
    to treat the previous image as the starting point for this rotation step.
    """
    base = (
        f"A photorealistic, highly detailed exterior view of a {display_name} "
        f"painted in {color}. "
        f"Clean, transparent background (pure white studio backdrop). "
        f"The camera is fixed at an exact eye-level height, shot with a 50mm lens "
        f"from a medium distance, maintaining the exact same framing, scale, and perspective. "
        f"The vehicle is positioned at exactly a {angle_deg}-degree viewing angle on a turntable. "
        f"Exterior view only, windows are highly reflective, interior is not visible. "
        f"Flat, even, omnidirectional studio lighting to ensure perfectly consistent "
        f"shadows and highlights across all frames. "
    )

    if prev_angle_deg is not None:
        step = angle_deg - prev_angle_deg
        base += (
            f"CONTINUITY: The attached previous frame shows the car at {prev_angle_deg}°. "
            f"Rotate the car exactly {step}° clockwise from the previous frame — "
            f"do NOT flip, mirror, or change the car's identity. "
        )

    base += _NEGATIVE_CONSTRAINTS
    return base



# ─────────────────────────────────────────────────────────────────────────────
# Frame Generator
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(3),
    before_sleep=lambda rs: logger.warning(
        "Retrying frame generation (attempt %d/3)...", rs.attempt_number
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
    Generate a single 360° turntable frame using the fixed user-defined prompt template.

    The prompt is purely text-based with angle embedded. No prev_frame chaining.
    This allows all frames to run in parallel for maximum speed.

    If master_frame_bytes (0° front view) is supplied, it is passed as a visual
    identity anchor to lock paint color, badge, and wheel design.

    Returns: Raw PNG bytes from Gemini.
    """
    prompt_text = build_frame_prompt(display_name, color, angle_deg, prev_angle_deg)

    logger.info("Generating frame @ %d° for %s", angle_deg, display_name)

    contents = []

    # Optional: Master identity reference (0° front view).
    # Helps Gemini lock paint shade, headlight design, wheel style.
    # Only injected if caller provides it (Frame 0° generates without it).
    if master_frame_bytes:
        contents.append(
            types.Part.from_bytes(data=master_frame_bytes, mime_type="image/png")
        )
        contents.append(
            "IDENTITY REFERENCE (0° front view of this exact car): "
            "Match the paint color, badge, headlights, alloy wheels, and body proportions exactly."
        )

    # Optional: User-supplied reference image for extra context
    if ref_bytes:
        contents.append(types.Part.from_bytes(data=ref_bytes, mime_type="image/png"))
        contents.append("ADDITIONAL REFERENCE: Use for extra styling context.")

    # Previous frame: visual continuity anchor (rotate 10° from THIS image)
    if prev_frame_bytes:
        contents.append(
            types.Part.from_bytes(data=prev_frame_bytes, mime_type="image/png")
        )
        contents.append(
            f"PREVIOUS FRAME ({prev_angle_deg}°): This is the immediately preceding frame. "
            f"Rotate the car exactly 10° clockwise from this image to reach {angle_deg}°. "
            f"Do NOT flip or mirror the vehicle."
        )

    # Main prompt
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
