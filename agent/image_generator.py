"""
image_generator.py
──────────────────
Generates a single car frame using gemini-2.5-flash-image (Vertex AI).

CarDekho Style 360° Studio Turntable Strategy:
- Camera Pitch: Elevated 15° studio camera looking down slightly at the car (CarDekho Object2VR standard).
- Floor/Background: Seamless pure white (#FFFFFF) studio stage with soft ground ambient shadow.
- Directional Mapping (CarDekho Clockwise Turntable):
    0°       : Direct Front View (Headlights & grille facing camera)
    10°–80°  : Car nose turns to SCREEN-LEFT (Driver/Left side profile opens)
    90°      : Full Left Side Profile (Car nose points straight to SCREEN-LEFT)
    100°–170°: Rear-Left quarter transitioning to Rear
    180°     : Direct Rear View (Taillights & rear bumper facing camera)
    190°–260°: Rear-Right quarter transitioning to Right Side
    270°     : Full Right Side Profile (Car nose points straight to SCREEN-RIGHT)
    280°–350°: Front-Right quarter transitioning back to Front
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


# Exact CarDekho-style 36-frame turntable map (0° to 350° in 10° steps)
CARDEKHO_ORIENTATION_MAP = {
    0:   ("Direct Front View", "Headlights and front grille facing camera directly, centered on stage."),
    10:  ("Front 10°", "Car nose turned 10° to SCREEN-LEFT. Driver headlight slightly angled left."),
    20:  ("Front-Left 20°", "Car nose turned 20° to SCREEN-LEFT. Front-driver quarter view."),
    30:  ("Front-Left 30°", "Car nose turned 30° to SCREEN-LEFT. Front-driver quarter view."),
    40:  ("Front-Left 40°", "Car nose turned 40° to SCREEN-LEFT. Front-driver quarter view."),
    50:  ("Front-Left 50°", "Car nose turned 50° to SCREEN-LEFT. Driver side profile opening."),
    60:  ("Front-Left 60°", "Car nose turned 60° to SCREEN-LEFT. Mostly driver side visible."),
    70:  ("Front-Left 70°", "Car nose turned 70° to SCREEN-LEFT. Near full driver side profile."),
    80:  ("Front-Left 80°", "Car nose turned 80° to SCREEN-LEFT. Almost pure driver side profile."),
    90:  ("Pure Left Side 90°", "Car nose points straight to SCREEN-LEFT. Full driver side profile visible."),
    100: ("Rear-Left 100°", "Car nose pointing SCREEN-LEFT and away. Rear driver corner rotating into view."),
    110: ("Rear-Left 110°", "Car nose pointing away to left. Rear-driver three-quarter view."),
    120: ("Rear-Left 120°", "Car nose pointing away to left. Rear-driver three-quarter view."),
    130: ("Rear-Left 130°", "Car nose pointing away. Taillights becoming clearly visible."),
    140: ("Rear-Left 140°", "Car nose pointing away. Rear driver quarter facing camera."),
    150: ("Rear-Left 150°", "Car nose pointing away. Mostly rear view."),
    160: ("Rear-Left 160°", "Car nose pointing away. Near direct rear view."),
    170: ("Rear-Left 170°", "Car nose pointing away. Almost direct rear view."),
    180: ("Direct Rear View 180°", "Taillights and rear bumper facing camera directly, centered on stage."),
    190: ("Rear-Right 190°", "Car nose pointing away toward SCREEN-RIGHT. Rear passenger corner rotating into view."),
    200: ("Rear-Right 200°", "Car nose pointing away toward SCREEN-RIGHT. Rear-passenger three-quarter view."),
    210: ("Rear-Right 210°", "Car nose pointing away toward SCREEN-RIGHT. Rear-passenger three-quarter view."),
    220: ("Rear-Right 220°", "Car nose pointing away toward SCREEN-RIGHT. Passenger side rear quarter visible."),
    230: ("Rear-Right 230°", "Car nose pointing SCREEN-RIGHT. Passenger side profile opening into view."),
    240: ("Rear-Right 240°", "Car nose pointing SCREEN-RIGHT. Mostly passenger side visible."),
    250: ("Rear-Right 250°", "Car nose pointing SCREEN-RIGHT. Near full passenger side profile."),
    260: ("Rear-Right 260°", "Car nose pointing SCREEN-RIGHT. Almost pure passenger side profile."),
    270: ("Pure Right Side 270°", "Car nose points straight to SCREEN-RIGHT. Full passenger side profile visible."),
    280: ("Front-Right 280°", "Car nose pointing SCREEN-RIGHT and forward. Front passenger corner rotating into view."),
    290: ("Front-Right 290°", "Car nose pointing SCREEN-RIGHT and forward. Front-passenger three-quarter view."),
    300: ("Front-Right 300°", "Car nose pointing SCREEN-RIGHT and forward. Front-passenger three-quarter view."),
    310: ("Front-Right 310°", "Car nose pointing SCREEN-RIGHT and forward. Headlights rotating toward camera."),
    320: ("Front-Right 320°", "Car nose turned 40° to SCREEN-RIGHT. Mostly front view."),
    330: ("Front-Right 330°", "Car nose turned 30° to SCREEN-RIGHT. Almost direct front view."),
    340: ("Front-Right 340°", "Car nose turned 20° to SCREEN-RIGHT. Almost direct front view."),
    350: ("Front-Right 350°", "Car nose turned 10° to SCREEN-RIGHT. Near direct front view completing 360°."),
}


def _build_prompt(
    display_name: str,
    color: str,
    angle_deg: int,
    prev_angle_deg: Optional[int] = None,
) -> str:
    title, desc = CARDEKHO_ORIENTATION_MAP.get(angle_deg, (f"{angle_deg}° View", f"Car rotated {angle_deg}°"))

    prompt = (
        f"CarDekho-style 360-degree exterior automotive studio photograph of a {display_name} in {color} color.\n"
        f"TURNTABLE ANGLE: Frame {angle_deg}° ({title}).\n"
        f"CAMERA ANGLE & PITCH: Elevated 15-degree studio camera angle looking slightly down at the car (showing bonnet, roofline, doors, and wheels in clear 3D perspective).\n"
        f"SPECIFIC ORIENTATION: {desc}\n\n"
        f"CARDEKHO STUDIO SETUP CONSTRAINTS:\n"
        f"1. CAMERA: Fixed tripod, 15° pitch down, 100% stationary camera position across all 36 frames.\n"
        f"2. BACKGROUND: Pure, seamless studio white background (#FFFFFF) with soft ambient ground shadow under tires.\n"
        f"3. FRAMING: Car occupies exactly 75% of the frame width, centered horizontally and vertically.\n"
    )

    if prev_angle_deg is not None:
        prompt += (
            f"4. ROTATION CONTINUITY: Attached is the PREVIOUS FRAME ({prev_angle_deg}°). "
            f"Rotate the car by EXACTLY 10° from the previous frame in the continuous rotation path. "
            f"DO NOT FLIP OR MIRROR THE VEHICLE.\n"
        )

    prompt += (
        f"5. MASTER IDENTITY: Attached is the MASTER REFERENCE (0° Front View). Match the car paint color, "
        f"headlights, front grille, alloy wheels, and trim design exactly.\n"
        f"Return ONLY the vehicle image on a pure white studio floor."
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
    Generate a single car frame matching CarDekho's 360° studio rotation standard.
    """
    prompt_text = _build_prompt(display_name, color, angle_deg, prev_angle_deg)
    logger.info(
        "Generating CarDekho-style frame @ %d° for %s (master_anchor=%s, prev_anchor=%s)",
        angle_deg, display_name,
        master_frame_bytes is not None,
        prev_frame_bytes is not None,
    )

    contents = []

    # 1. Master Identity Reference (0° Front View)
    if master_frame_bytes:
        contents.append(types.Part.from_bytes(data=master_frame_bytes, mime_type="image/png"))
        contents.append(
            "MASTER REFERENCE (0° Front View): Use this image to lock car identity, paint color, grille, headlights, rims, and body proportions."
        )

    # 2. Previous Frame Reference (angle - 10°)
    if prev_frame_bytes:
        contents.append(types.Part.from_bytes(data=prev_frame_bytes, mime_type="image/png"))
        contents.append(
            f"PREVIOUS FRAME ({prev_angle_deg}° View): Rotate the car by exactly 10° from this image in the same continuous CarDekho rotation path. DO NOT FLIP OR MIRROR."
        )

    # 3. User-supplied reference
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
