"""
image_generator.py
──────────────────
Generates a single car frame using gemini-2.5-flash-image (Vertex AI).

Rigid 360° Turntable Rotation Strategy:
- Continuous, non-contradictory 36-frame rotation map.
- Nose of the car moves smoothly across the 4 quadrants:
    0°       : Direct Front (Headlights facing camera)
    10°–80°  : Nose turns progressively toward SCREEN-RIGHT (Passenger side opens)
    90°      : Full Side Profile (Nose points straight to SCREEN-RIGHT)
    100°–170°: Nose turns toward rear-right (Rear opens)
    180°     : Direct Rear (Taillights facing camera)
    190°–260°: Nose turns toward SCREEN-LEFT (Driver side opens)
    270°     : Full Side Profile (Nose points straight to SCREEN-LEFT)
    280°–350°: Nose turns from SCREEN-LEFT back to Front
- Dual Reference Anchors:
    1. Master Reference (Frame 1 @ 0° Front View) — Permanent visual identity
    2. Previous Frame (Frame i-1) — 10° smooth incremental transition
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

_IMAGEN_REGION = "us-central1"
_MODEL_ID      = "gemini-2.5-flash-image"

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


# Exact, physically consistent 36-frame orientation map (0° to 350° in 10° steps)
FRAME_ORIENTATION_MAP = {
    0:   ("Direct Front View", "The front headlights and grille face the camera directly, perfectly centered."),
    10:  ("Front-Right 10°", "Car nose turned 10° toward SCREEN-RIGHT. Headlights slightly angled right."),
    20:  ("Front-Right 20°", "Car nose turned 20° toward SCREEN-RIGHT. Passenger headlight moving right."),
    30:  ("Front-Right 30°", "Car nose turned 30° toward SCREEN-RIGHT. Front-passenger three-quarter view."),
    40:  ("Front-Right 40°", "Car nose turned 40° toward SCREEN-RIGHT. Front-passenger three-quarter view."),
    50:  ("Front-Right 50°", "Car nose turned 50° toward SCREEN-RIGHT. Passenger side profile becoming prominent."),
    60:  ("Front-Right 60°", "Car nose turned 60° toward SCREEN-RIGHT. Mostly passenger side visible."),
    70:  ("Front-Right 70°", "Car nose turned 70° toward SCREEN-RIGHT. Near full passenger side profile."),
    80:  ("Front-Right 80°", "Car nose turned 80° toward SCREEN-RIGHT. Almost pure passenger side profile."),
    90:  ("Pure Side Profile 90°", "Car nose points straight to SCREEN-RIGHT. Full passenger side profile visible."),
    100: ("Rear-Right 100°", "Car nose pointing SCREEN-RIGHT and away. Rear passenger corner rotating into view."),
    110: ("Rear-Right 110°", "Car nose pointing away to the right. Rear-passenger three-quarter view."),
    120: ("Rear-Right 120°", "Car nose pointing away to the right. Rear-passenger three-quarter view."),
    130: ("Rear-Right 130°", "Car nose pointing away. Taillights becoming clearly visible."),
    140: ("Rear-Right 140°", "Car nose pointing away. Rear passenger quarter facing camera."),
    150: ("Rear-Right 150°", "Car nose pointing away. Mostly rear view with right tail visible."),
    160: ("Rear-Right 160°", "Car nose pointing away. Near direct rear view."),
    170: ("Rear-Right 170°", "Car nose pointing away. Almost direct rear view."),
    180: ("Direct Rear View 180°", "The rear bumper and taillights face the camera directly, perfectly centered."),
    190: ("Rear-Left 190°", "Car nose pointing away toward SCREEN-LEFT. Rear driver corner rotating into view."),
    200: ("Rear-Left 200°", "Car nose pointing away toward SCREEN-LEFT. Rear-driver three-quarter view."),
    210: ("Rear-Left 210°", "Car nose pointing away toward SCREEN-LEFT. Rear-driver three-quarter view."),
    220: ("Rear-Left 220°", "Car nose pointing away toward SCREEN-LEFT. Driver side rear quarter visible."),
    230: ("Rear-Left 230°", "Car nose pointing SCREEN-LEFT. Driver side profile opening into view."),
    240: ("Rear-Left 240°", "Car nose pointing SCREEN-LEFT. Mostly driver side visible."),
    250: ("Rear-Left 250°", "Car nose pointing SCREEN-LEFT. Near full driver side profile."),
    260: ("Rear-Left 260°", "Car nose pointing SCREEN-LEFT. Almost pure driver side profile."),
    270: ("Pure Side Profile 270°", "Car nose points straight to SCREEN-LEFT. Full driver side profile visible."),
    280: ("Front-Left 280°", "Car nose pointing SCREEN-LEFT and forward. Front driver corner rotating into view."),
    290: ("Front-Left 290°", "Car nose pointing SCREEN-LEFT and forward. Front-driver three-quarter view."),
    300: ("Front-Left 300°", "Car nose pointing SCREEN-LEFT and forward. Front-driver three-quarter view."),
    310: ("Front-Left 310°", "Car nose pointing SCREEN-LEFT and forward. Headlights rotating toward camera."),
    320: ("Front-Left 320°", "Car nose turned 40° to SCREEN-LEFT. Mostly front view."),
    330: ("Front-Left 330°", "Car nose turned 30° to SCREEN-LEFT. Almost direct front view."),
    340: ("Front-Left 340°", "Car nose turned 20° to SCREEN-LEFT. Almost direct front view."),
    350: ("Front-Left 350°", "Car nose turned 10° to SCREEN-LEFT. Near direct front view completing 360°."),
}


def _build_prompt(
    display_name: str,
    color: str,
    angle_deg: int,
    prev_angle_deg: Optional[int] = None,
) -> str:
    title, desc = FRAME_ORIENTATION_MAP.get(angle_deg, (f"{angle_deg}° View", f"Car rotated {angle_deg}°"))

    prompt = (
        f"Automotive 360-degree turntable photo of a {display_name} in {color} color.\n"
        f"TARGET FRAME: Angle {angle_deg}° ({title}).\n"
        f"ORIENTATIONAL REQUIREMENT: {desc}\n\n"
        f"STRICT TURNTABLE CONSTRAINTS:\n"
        f"1. CAMERA: Fixed horizontal eye-level camera, 0° pitch angle, no high/low camera angle.\n"
        f"2. TURNTABLE: The car is on a center studio turntable rotating in a SINGLE continuous direction.\n"
        f"3. SCALE & FRAMING: The car occupies 75% of the image width, centered on a pure white (#FFFFFF) floor.\n"
    )

    if prev_angle_deg is not None:
        prompt += (
            f"4. CONTINUITY: Attached is the PREVIOUS FRAME ({prev_angle_deg}°). "
            f"Rotate the vehicle by EXACTLY 10° from the previous frame in the continuous rotation path. "
            f"DO NOT FLIP OR MIRROR THE VEHICLE HORIZONTALLY.\n"
        )

    prompt += (
        f"5. IDENTITY: Attached is the MASTER REFERENCE (0° Front View). Match the car paint color, "
        f"headlights, front grille, rims, and body lines exactly.\n"
        f"Output ONLY the photorealistic car on a pure white studio background."
    )

    return prompt


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
    Generate a single car frame maintaining master identity & smooth 10° rotation step.

    Args:
        display_name:       e.g. "Honda City"
        color:              e.g. "Dashing Silver"
        angle_deg:          0–350 in 10° steps
        master_frame_bytes: Frame 1 (0° Front view) PNG bytes — permanent identity anchor
        prev_frame_bytes:   Frame i-1 PNG bytes — smooth rotation step anchor
        prev_angle_deg:     Angle of prev_frame_bytes (angle_deg - 10)
        ref_bytes:          Optional user-supplied reference

    Returns:
        Raw PNG bytes from Gemini.
    """
    prompt_text = _build_prompt(display_name, color, angle_deg, prev_angle_deg)
    logger.info(
        "Generating frame @ %d° for %s (master_anchor=%s, prev_anchor=%s)",
        angle_deg, display_name,
        master_frame_bytes is not None,
        prev_frame_bytes is not None,
    )

    contents = []

    # 1. Master Identity Reference (0° Front View) — Permanent Anchor
    if master_frame_bytes:
        contents.append(types.Part.from_bytes(data=master_frame_bytes, mime_type="image/png"))
        contents.append(
            "MASTER REFERENCE (0° Front View): Use this image to lock car identity, paint color, grille, headlights, rims, and body proportions."
        )

    # 2. Previous Frame Reference (angle - 10°) — Rotation Transition Anchor
    if prev_frame_bytes:
        contents.append(types.Part.from_bytes(data=prev_frame_bytes, mime_type="image/png"))
        contents.append(
            f"PREVIOUS FRAME ({prev_angle_deg}° View): Rotate the car by exactly 10° from this image in the same continuous rotation path. DO NOT FLIP OR MIRROR."
        )

    # 3. User-supplied reference (optional secondary)
    if ref_bytes:
        contents.append(types.Part.from_bytes(data=ref_bytes, mime_type="image/png"))
        contents.append("USER REFERENCE: Additional styling guide.")

    # 4. Main prompt
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
            logger.info("  ✓ Frame @ %d° generated in %.1fs (%d bytes)",
                        angle_deg, elapsed, len(part.inline_data.data))
            return part.inline_data.data

    raise ValueError(
        f"Gemini returned no image for {display_name} @ {angle_deg}° "
        f"(response: {response.candidates[0].content.parts})"
    )
