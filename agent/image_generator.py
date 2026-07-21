"""
image_generator.py
──────────────────
Generates a single car frame using gemini-2.5-flash-image (Vertex AI).

- Model is lazy-loaded (created inside generate_frame, not at import time)
  so the FastAPI server can start without hitting the API immediately.
- Vertex AI region: us-central1 (gemini image models only available there).
- Returns raw PNG bytes; background removal is a separate step.
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

# Imagen models are only available in us-central1 (even for asia-south1 projects)
_IMAGEN_REGION = "us-central1"
_MODEL_ID      = "gemini-2.5-flash-image"

# Lazy-loaded client — initialised once on first call, not at import time
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT_ID,
            location=_IMAGEN_REGION,
        )
        logger.info("Vertex AI genai client initialised (project=%s, location=%s)",
                    GCP_PROJECT_ID, _IMAGEN_REGION)
    return _client


def _build_prompt(display_name: str, color: str, angle_deg: int) -> str:
    """Map rotation angle to a descriptive camera position for consistency."""
    if angle_deg == 0:
        cam = "front view, perfectly centered"
    elif 0 < angle_deg <= 45:
        cam = "front-right three-quarter view"
    elif 45 < angle_deg <= 90:
        cam = "right side profile view"
    elif 90 < angle_deg <= 135:
        cam = "rear-right three-quarter view"
    elif 135 < angle_deg <= 180:
        cam = "rear view, perfectly centered"
    elif 180 < angle_deg <= 225:
        cam = "rear-left three-quarter view"
    elif 225 < angle_deg <= 270:
        cam = "left side profile view"
    elif 270 < angle_deg <= 315:
        cam = "front-left three-quarter view"
    else:
        cam = "front view, slight right angle"

    return (
        f"Photorealistic automotive studio photograph of a {display_name} car in {color} color. "
        f"Camera angle: {cam}. "
        f"CRITICAL REQUIREMENTS: "
        f"1. Scale & Position: The car MUST be exactly centered. The full car must be visible from bumper to bumper without cropping. Wheels must touch the ground line. "
        f"2. Background: MUST be a pure, bright, seamless white (#FFFFFF) studio background. Absolutely NO dark backgrounds, NO navy backgrounds, NO outdoor scenes, NO shadows on the walls. "
        f"3. Consistency: The proportions, wheelbase length, wheel size, and design details MUST be identical to every other angle of this car. The car must not appear squashed or elongated. "
        f"4. Style: Professional, 4K resolution, highly detailed, realistic reflections. No people, no text, no license plate. "
        f"ONLY return the car on a pure white background."
    )


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
) -> bytes:
    """
    Generate a single car image frame at the given rotation angle.

    Args:
        display_name: e.g. "Honda City"
        color:        e.g. "Dashing Silver"
        angle_deg:    0–350 in 10° steps
        seed:         Not used (genai image doesn't support seed yet)

    Returns:
        Raw PNG bytes from Gemini.
    """
    prompt_text = _build_prompt(display_name, color, angle_deg)
    logger.info("Generating frame @ %d° for %s (ref_image=%s)", angle_deg, display_name, ref_bytes is not None)

    contents = []
    if ref_bytes:
        contents.append(types.Part.from_bytes(data=ref_bytes, mime_type='image/png'))
        contents.append(f"This is the reference image for the car. Ensure exact consistency for the {display_name} in {color}.")
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

    # Extract image bytes from response
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            logger.info("  ✓ Frame @ %d° generated in %.1fs (%d bytes)",
                        angle_deg, elapsed, len(part.inline_data.data))
            return part.inline_data.data  # raw PNG bytes

    raise ValueError(
        f"Gemini returned no image for {display_name} @ {angle_deg}° "
        f"(response: {response.candidates[0].content.parts})"
    )
