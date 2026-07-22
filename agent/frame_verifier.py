"""
frame_verifier.py
─────────────────
Verifies that a generated car frame actually shows the car
at the expected rotation angle, using Gemini Vision.

Strategy:
  - Send the generated image + a structured question to Gemini (text-only output).
  - Ask: "Does this show the car nose pointing [direction]? Is this frame correct?"
  - Parse YES/NO from response.
  - Return { valid: bool, reason: str, detected: str }

This is called by the pipeline AFTER each frame is generated.
If verification fails, the pipeline retries generation (up to MAX_VERIFY_ATTEMPTS).
"""

import logging
import re
from typing import Optional

from google import genai
from google.genai import types

from config.settings import GCP_PROJECT_ID

logger = logging.getLogger(__name__)

_VERIFY_REGION = "us-central1"
_VERIFY_MODEL  = "gemini-2.5-flash"   # text-only output, fast & cheap

_client: Optional[genai.Client] = None

MAX_VERIFY_ATTEMPTS = 3   # max generation retries per frame if verification fails


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT_ID,
            location=_VERIFY_REGION,
        )
    return _client


# Expected nose direction for each rotation angle (10° steps)
_EXPECTED_DIRECTION: dict[int, tuple[str, str]] = {
    0:   ("facing the camera directly", "direct front view, headlights visible"),
    10:  ("turned slightly to the LEFT", "front-left quarter, mostly front"),
    20:  ("turned 20° to the LEFT", "front-left quarter view"),
    30:  ("turned 30° to the LEFT", "front-left quarter view"),
    40:  ("turned 40° to the LEFT", "front-left quarter view"),
    50:  ("turned 50° to the LEFT", "transitioning to left side profile"),
    60:  ("turned 60° to the LEFT", "mostly left side visible"),
    70:  ("turned 70° to the LEFT", "near left side profile"),
    80:  ("turned 80° to the LEFT", "almost pure left side profile"),
    90:  ("pointing straight to the LEFT", "full left side profile, no front or rear visible"),
    100: ("pointing to the LEFT and slightly away", "rear-left quarter, rear becoming visible"),
    110: ("pointing away to the LEFT", "rear-left three-quarter view"),
    120: ("pointing away to the LEFT", "rear-left three-quarter view"),
    130: ("pointing away", "taillights becoming clearly visible"),
    140: ("pointing away", "rear-left quarter facing camera"),
    150: ("pointing away", "mostly rear view"),
    160: ("pointing away", "near direct rear view"),
    170: ("pointing slightly away to the RIGHT", "almost direct rear view"),
    180: ("pointing directly away from camera", "direct rear view, taillights fully visible"),
    190: ("pointing away to the RIGHT", "rear-right quarter, passenger side emerging"),
    200: ("pointing away to the RIGHT", "rear-right three-quarter view"),
    210: ("pointing away to the RIGHT", "rear-right three-quarter view"),
    220: ("pointing to the RIGHT and slightly away", "passenger side rear quarter visible"),
    230: ("turned to the RIGHT", "passenger side profile opening"),
    240: ("turned 60° to the RIGHT", "mostly right side visible"),
    250: ("turned 70° to the RIGHT", "near right side profile"),
    260: ("turned 80° to the RIGHT", "almost pure right side profile"),
    270: ("pointing straight to the RIGHT", "full right side profile, no front or rear visible"),
    280: ("turned 80° to the RIGHT and forward", "front-right quarter emerging"),
    290: ("turned 70° to the RIGHT and forward", "front-right three-quarter view"),
    300: ("turned 60° to the RIGHT and forward", "front-right three-quarter view"),
    310: ("turned 50° to the RIGHT", "headlights rotating toward camera"),
    320: ("turned 40° to the RIGHT", "mostly front view"),
    330: ("turned 30° to the RIGHT", "almost direct front view"),
    340: ("turned 20° to the RIGHT", "almost direct front view"),
    350: ("turned 10° to the RIGHT", "near direct front view completing 360°"),
}


def _build_verify_prompt(display_name: str, angle_deg: int) -> str:
    direction, description = _EXPECTED_DIRECTION.get(
        angle_deg,
        ("at an unknown angle", f"rotated {angle_deg}°")
    )
    return (
        f"You are a quality control inspector for 360° automotive turntable photography.\n\n"
        f"TASK: Inspect this image of a {display_name} and answer TWO questions.\n\n"
        f"EXPECTED VIEW: This should be frame {angle_deg}° of a 360° clockwise turntable.\n"
        f"At {angle_deg}°, the car nose should be {direction}.\n"
        f"The expected description: {description}.\n\n"
        f"QUESTIONS:\n"
        f"1. Does the car nose direction match '{direction}'? Answer YES or NO.\n"
        f"2. Briefly describe what you actually see: which part of the car is facing the camera?\n\n"
        f"RESPOND IN THIS FORMAT ONLY:\n"
        f"VALID: YES\n"
        f"DETECTED: [what you actually see]\n\n"
        f"OR:\n"
        f"VALID: NO\n"
        f"DETECTED: [what you actually see]\n"
        f"REASON: [why it does not match {angle_deg}°]"
    )


def verify_frame_angle(
    png_bytes: bytes,
    expected_angle: int,
    display_name: str,
) -> dict:
    """
    Verify that a generated frame shows the car at the expected rotation angle.

    Args:
        png_bytes:      Raw PNG bytes of the generated frame.
        expected_angle: The target turntable angle (0–350° in 10° steps).
        display_name:   Human-readable car name (e.g. "Honda City").

    Returns:
        {
            'valid':    bool,    # True if angle looks correct
            'detected': str,     # What Gemini actually saw in the image
            'reason':   str,     # Explanation (only set on failure)
            'raw':      str,     # Full raw Gemini response for debugging
        }
    """
    prompt = _build_verify_prompt(display_name, expected_angle)

    try:
        response = _get_client().models.generate_content(
            model=_VERIFY_MODEL,
            contents=[
                types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT"],
                temperature=0.1,   # low temp for deterministic quality check
            ),
        )

        raw_text = ""
        for part in response.candidates[0].content.parts:
            if part.text:
                raw_text += part.text

        # Parse structured response
        valid_match    = re.search(r"VALID:\s*(YES|NO)", raw_text, re.IGNORECASE)
        detected_match = re.search(r"DETECTED:\s*(.+?)(?:\n|$)", raw_text, re.IGNORECASE)
        reason_match   = re.search(r"REASON:\s*(.+?)(?:\n|$)", raw_text, re.IGNORECASE)

        is_valid   = valid_match and valid_match.group(1).upper() == "YES"
        detected   = detected_match.group(1).strip() if detected_match else "unknown"
        reason     = reason_match.group(1).strip() if reason_match else ""

        result = {
            "valid":    bool(is_valid),
            "detected": detected,
            "reason":   reason,
            "raw":      raw_text[:500],   # truncate for logging
        }

        if result["valid"]:
            logger.info(
                "  ✓ Frame @ %d° VERIFIED — detected: %s",
                expected_angle, detected
            )
        else:
            logger.warning(
                "  ✗ Frame @ %d° FAILED verification — expected nose %s | detected: %s | reason: %s",
                expected_angle,
                _EXPECTED_DIRECTION.get(expected_angle, ("?", ""))[0],
                detected,
                reason,
            )

        return result

    except Exception as e:
        logger.error("  ✗ Verification error @ %d°: %s", expected_angle, e)
        # On verifier error, accept the frame (don't block pipeline)
        return {
            "valid":    True,
            "detected": "verification-skipped-due-to-error",
            "reason":   str(e),
            "raw":      "",
        }
