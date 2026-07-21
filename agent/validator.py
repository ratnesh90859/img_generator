"""
validator.py
────────────
Three-level validation for generated vehicle frames:

  Level 1 — Frame count   : Are all expected frames present in GCS?
  Level 2 — Image quality : Is each frame a valid RGBA WebP?
  Level 3 — Audit         : Full health check across all Firestore models.
"""

import logging
from typing import Optional
from io import BytesIO

from PIL import Image
from google.cloud import firestore, storage
from config.settings import (
    GCS_BUCKET_NAME,
    TOTAL_FRAMES,
    FIRESTORE_COLLECTION,
)
from agent.gcs_uploader import list_existing_frames

logger = logging.getLogger(__name__)

_fs_client  = firestore.Client()
_gcs_client = storage.Client()


# ─────────────────────────────────────────────────────────────────
# Level 1 — Frame Count Validation
# ─────────────────────────────────────────────────────────────────

def validate_frame_count(model_id: str, expected: int = TOTAL_FRAMES) -> dict:
    """
    Check how many frames exist in GCS vs how many were expected.

    Returns:
        {
            "valid": bool,
            "uploaded": int,
            "expected": int,
            "missing_count": int,
            "missing_frames": [1, 12, 31, ...]   ← frame numbers
        }
    """
    existing = set(list_existing_frames(model_id))
    expected_set = set(range(1, expected + 1))
    missing = sorted(expected_set - existing)

    result = {
        "valid": len(missing) == 0,
        "uploaded": len(existing),
        "expected": expected,
        "missing_count": len(missing),
        "missing_frames": missing,
    }

    if missing:
        logger.warning("[%s] Missing %d frames: %s", model_id, len(missing), missing)
    else:
        logger.info("[%s] ✓ All %d frames present", model_id, expected)

    return result


# ─────────────────────────────────────────────────────────────────
# Level 2 — Image Quality Validation
# ─────────────────────────────────────────────────────────────────

def validate_single_frame(model_id: str, frame_num: int) -> dict:
    """
    Download a frame from GCS and run quality checks on it.

    Returns:
        {
            "frame_num": int,
            "valid": bool,
            "issues": [str, ...]
        }
    """
    issues = []
    bucket  = _gcs_client.bucket(GCS_BUCKET_NAME)
    gcs_path = f"processed/{model_id}/frame_{str(frame_num).zfill(3)}.webp"
    blob = bucket.blob(gcs_path)

    try:
        webp_bytes = blob.download_as_bytes()
    except Exception as e:
        return {"frame_num": frame_num, "valid": False, "issues": [f"GCS download failed: {e}"]}

    try:
        img = Image.open(BytesIO(webp_bytes))
    except Exception as e:
        return {"frame_num": frame_num, "valid": False, "issues": [f"Cannot open image: {e}"]}

    # Check 1: Valid mode (RGB or RGBA)
    if img.mode not in ("RGB", "RGBA"):
        issues.append(f"Invalid mode: {img.mode}")

    # Check 2: Minimum size
    if img.width < 256 or img.height < 256:
        issues.append(f"Too small: {img.width}×{img.height}")

    # Check 3: File size sanity
    if len(webp_bytes) < 5_000:
        issues.append(f"File too small ({len(webp_bytes)} bytes) — possibly corrupt")

    return {
        "frame_num": frame_num,
        "valid": len(issues) == 0,
        "issues": issues,
        "size_bytes": len(webp_bytes),
        "dimensions": f"{img.width}×{img.height}",
    }


def validate_all_frames(model_id: str) -> dict:
    """
    Run quality validation on every frame that exists in GCS for a model.

    Returns a summary report with any frames that failed quality checks.
    """
    existing_frames = list_existing_frames(model_id)
    quality_issues  = []
    passed          = 0

    for frame_num in existing_frames:
        result = validate_single_frame(model_id, frame_num)
        if result["valid"]:
            passed += 1
        else:
            quality_issues.append(result)
            logger.warning(
                "[%s] Frame %02d FAILED quality check: %s",
                model_id, frame_num, result["issues"]
            )

    return {
        "model_id":       model_id,
        "total_checked":  len(existing_frames),
        "passed":         passed,
        "failed":         len(quality_issues),
        "quality_issues": quality_issues,
    }


# ─────────────────────────────────────────────────────────────────
# Level 3 — Full Validation Report (count + quality combined)
# ─────────────────────────────────────────────────────────────────

def full_validation_report(model_id: str) -> dict:
    """
    Run Level 1 (count) + Level 2 (quality) and return a unified report.
    Also writes the report back to Firestore for dashboard visibility.
    """
    logger.info("Running full validation for %s ...", model_id)

    count_result   = validate_frame_count(model_id)
    quality_result = validate_all_frames(model_id)

    overall_valid = count_result["valid"] and quality_result["failed"] == 0

    report = {
        "model_id":        model_id,
        "overall_valid":   overall_valid,
        "frame_count":     count_result,
        "quality":         quality_result,
        "can_retry":       not overall_valid,
    }

    # Write report to Firestore
    doc_ref = _fs_client.collection(FIRESTORE_COLLECTION).document(model_id)
    doc_ref.set({"validation_report": report}, merge=True)

    if overall_valid:
        logger.info("[%s] ✅ Full validation PASSED", model_id)
    else:
        logger.warning("[%s] ⚠️  Full validation FAILED — see report", model_id)

    return report


# ─────────────────────────────────────────────────────────────────
# Audit — check ALL models in Firestore
# ─────────────────────────────────────────────────────────────────

def audit_all_models() -> list[dict]:
    """
    Run frame count validation across every 'completed' model in Firestore.
    Returns a list of models that have issues.

    Run this as a cron / admin script to detect silent failures.
    """
    logger.info("Starting audit of all models in Firestore...")
    docs    = _fs_client.collection(FIRESTORE_COLLECTION).stream()
    issues  = []
    checked = 0

    for doc in docs:
        data = doc.to_dict()
        if data.get("status") != "completed":
            continue
        checked += 1
        count = validate_frame_count(data["model_id"])
        if not count["valid"]:
            issues.append({
                "model_id": data["model_id"],
                "issue":    f"Missing {count['missing_count']} frames",
                "missing":  count["missing_frames"],
            })

    logger.info(
        "Audit complete: %d models checked, %d have issues.",
        checked, len(issues)
    )
    return issues
