"""
pipeline.py
───────────
Orchestrates the full 360° image generation pipeline for a single vehicle model.

Generation Strategy — SEQUENTIAL (better accuracy):
  Frames are generated one at a time: 0° → 10° → 20° → … → 350°.
  Each frame receives:
    • The exact angle in the prompt (user-defined template)
    • Master Frame (0° front view) as identity anchor
    • Previous Frame (N-1) as rotation continuity reference

  This chain of references ensures each frame is a smooth 10° step
  from the one before it, giving the best consistency achievable with
  a generative model.

Pipeline Steps:
  1. Generate Frame 0° (front view, no references).
  2. For each subsequent frame (10°→350°), generate sequentially passing
     master frame + previous frame as visual references.
  3. Post-process: convert to PNG, upload to GCS.
  4. Track progress in Firestore.
  5. Upload thumbnail.
  6. Run validation.
  7. Mark completed / partial / failed in Firestore.
"""

import logging
import time
import base64
from datetime import datetime, timezone
from typing import Optional

from google.cloud import firestore

from config.settings import (
    TOTAL_FRAMES,
    FRAME_ANGLE_STEP,
    FIRESTORE_COLLECTION,
    DEFAULT_COLOR,
    gcs_frame_path,
    gcs_thumbnail_path,
)
from agent.image_generator import generate_frame
from agent.background_remover import remove_background, quick_validate_alpha
from agent.gcs_uploader import (
    upload_frame,
    upload_thumbnail,
    frame_exists,
    delete_frames,
)
from agent.validator import full_validation_report

logger = logging.getLogger(__name__)
_fs = firestore.Client()


def _doc_ref(model_id: str):
    return _fs.collection(FIRESTORE_COLLECTION).document(model_id)


def get_vehicle_status(model_id: str) -> Optional[dict]:
    """Fetch the Firestore document for a model. Returns None if not found."""
    doc = _doc_ref(model_id).get()
    return doc.to_dict() if doc.exists else None


def run_pipeline(
    model_id: str,
    display_name: str,
    color: str = DEFAULT_COLOR,
    force_regenerate: bool = False,
    ref_front: Optional[str] = None,
    ref_right: Optional[str] = None,
    ref_rear: Optional[str] = None,
    ref_left: Optional[str] = None,
) -> dict:
    """
    Main pipeline entry point. Called as a FastAPI background task.

    Sequential generation: 0° → 10° → 20° → … → 350°
    Each frame uses master frame + previous frame as visual anchors.

    Args:
        model_id:          e.g. "tata-nexon"
        display_name:      e.g. "Tata Nexon"
        color:             e.g. "Dashing Silver"
        force_regenerate:  If True, delete existing GCS frames and regenerate all.
        ref_front/right/rear/left: Optional base64 reference images per quadrant.
    """
    doc_ref    = _doc_ref(model_id)
    started_at = datetime.now(timezone.utc)
    logger.info("=== Pipeline START: %s (sequential) ===", model_id)

    # ── Force regenerate ─────────────────────────────────────────────────────
    if force_regenerate:
        logger.info("Force regenerate — deleting existing GCS frames for %s", model_id)
        delete_frames(model_id)

    # ── Mark as generating ───────────────────────────────────────────────────
    doc_ref.set({
        "model_id":         model_id,
        "display_name":     display_name,
        "color":            color,
        "status":           "generating",
        "total_frames":     TOTAL_FRAMES,
        "generated_frames": 0,
        "failed_frames":    [],
        "started_at":       started_at.isoformat(),
        "frame_urls":       [],
    }, merge=True)

    # ── Decode user-supplied quadrant references ─────────────────────────────
    ref_bytes_map = {
        "front": base64.b64decode(ref_front.split(",")[-1]) if ref_front else None,
        "right": base64.b64decode(ref_right.split(",")[-1]) if ref_right else None,
        "rear":  base64.b64decode(ref_rear.split(",")[-1])  if ref_rear  else None,
        "left":  base64.b64decode(ref_left.split(",")[-1])  if ref_left  else None,
    }

    frame_urls:    list[Optional[str]] = [None] * TOTAL_FRAMES
    failed_frames: list[dict]          = []
    generated:     int                 = 0

    # Sequential state — passed to each generation call
    master_frame_bytes: Optional[bytes] = None   # Frame 1 @ 0° — permanent identity
    prev_frame_bytes:   Optional[bytes] = None   # Frame N-1   — rotation continuity
    prev_angle_deg:     Optional[int]   = None

    # ── Sequential loop: 0° → 10° → … → 350° ────────────────────────────────
    for frame_num in range(1, TOTAL_FRAMES + 1):
        angle_deg = (frame_num - 1) * FRAME_ANGLE_STEP

        # Select quadrant reference image
        if angle_deg <= 45 or angle_deg >= 315:
            current_ref = ref_bytes_map["front"]
        elif 45 < angle_deg <= 135:
            current_ref = ref_bytes_map["right"]
        elif 135 < angle_deg <= 225:
            current_ref = ref_bytes_map["rear"]
        else:
            current_ref = ref_bytes_map["left"]

        # Skip frames already in GCS (resumable)
        if not force_regenerate and frame_exists(model_id, frame_num):
            gcs_path = gcs_frame_path(model_id, frame_num)
            frame_urls[frame_num - 1] = gcs_path
            generated += 1
            logger.info("  [%02d/%d] Skipped (cached) — %s", frame_num, TOTAL_FRAMES, gcs_path)

            # Still load cached frame as reference for next iteration
            if prev_frame_bytes is None or (frame_num == 1 and master_frame_bytes is None):
                try:
                    from google.cloud import storage as _gcs
                    from config.settings import GCS_BUCKET_NAME
                    cached_bytes = _gcs.Client().bucket(GCS_BUCKET_NAME).blob(gcs_path).download_as_bytes()
                    if frame_num == 1:
                        master_frame_bytes = cached_bytes
                    prev_frame_bytes = cached_bytes
                    prev_angle_deg   = angle_deg
                except Exception as e:
                    logger.warning("  Could not load cached frame as reference: %s", e)
            continue

        try:
            logger.info(
                "  [%02d/%d] Generating @ %d° (master=%s, prev=%s @ %s°)",
                frame_num, TOTAL_FRAMES, angle_deg,
                master_frame_bytes is not None,
                prev_frame_bytes is not None,
                prev_angle_deg,
            )

            # Generate frame with master + previous anchors
            png_bytes = generate_frame(
                display_name,
                color,
                angle_deg,
                ref_bytes=current_ref,
                master_frame_bytes=master_frame_bytes,
                prev_frame_bytes=prev_frame_bytes,
                prev_angle_deg=prev_angle_deg,
            )

            # Post-process
            out_bytes = remove_background(png_bytes)

            # Quality check
            qc = quick_validate_alpha(out_bytes)
            if not qc["valid"]:
                raise ValueError(f"Quality check failed: {qc['issues']}")

            # Upload to GCS
            gcs_path = upload_frame(model_id, frame_num, out_bytes)

            # Update progress
            frame_urls[frame_num - 1] = gcs_path
            generated += 1
            doc_ref.update({"generated_frames": generated, "status": "generating"})

            # Lock Frame 1 as master identity anchor
            if frame_num == 1:
                master_frame_bytes = png_bytes
                logger.info("  ✓ Master identity frame set (Frame 1 @ 0°)")

            # Always update previous frame reference
            prev_frame_bytes = png_bytes
            prev_angle_deg   = angle_deg

        except Exception as e:
            logger.error("  [%02d/%d] FAILED @ %d°: %s", frame_num, TOTAL_FRAMES, angle_deg, e)
            failed_frames.append({
                "frame_num": frame_num,
                "angle_deg": angle_deg,
                "error":     str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            doc_ref.update({"failed_frames": failed_frames})
            # Keep last good prev_frame for next iteration

    # ── Filter out None slots ─────────────────────────────────────────────────
    gcs_paths = [u for u in frame_urls if u is not None]

    # ── Upload thumbnail (frame 1) ────────────────────────────────────────────
    thumbnail_path = None
    try:
        if gcs_paths:
            from google.cloud import storage as _gcs
            from config.settings import GCS_BUCKET_NAME
            thumb_raw = _gcs.Client().bucket(GCS_BUCKET_NAME).blob(
                gcs_frame_path(model_id, 1)
            ).download_as_bytes()
            thumbnail_path = upload_thumbnail(model_id, thumb_raw)
    except Exception as e:
        logger.warning("Thumbnail upload failed: %s", e)

    # ── Final status ──────────────────────────────────────────────────────────
    completed_at = datetime.now(timezone.utc)
    elapsed_s    = (completed_at - started_at).total_seconds()
    final_status = (
        "completed" if len(failed_frames) == 0 else
        "partial"   if len(gcs_paths) > 0     else
        "failed"
    )

    final_doc = {
        "status":            final_status,
        "generated_frames":  generated_counter[0],
        "failed_frames":     failed_frames,
        "gcs_paths":         gcs_paths,
        "thumbnail_path":    thumbnail_path,
        "completed_at":      completed_at.isoformat(),
        "elapsed_seconds":   round(elapsed_s, 1),
    }
    doc_ref.update(final_doc)

    # ── Run full validation ───────────────────────────────────────────────────
    if final_status in ("completed", "partial"):
        try:
            full_validation_report(model_id)
        except Exception as e:
            logger.warning("Validation report failed: %s", e)

    logger.info(
        "=== Pipeline END: %s | status=%s | frames=%d/%d | %.0fs ===",
        model_id, final_status, generated_counter[0], TOTAL_FRAMES, elapsed_s
    )

    return {**final_doc, "model_id": model_id}


def retry_failed_frames(model_id: str, display_name: str, color: str = DEFAULT_COLOR) -> dict:
    """Re-run only the frames listed in failed_frames. Skips successful ones."""
    doc = get_vehicle_status(model_id)
    if not doc:
        raise ValueError(f"No Firestore record found for {model_id}")

    failed = doc.get("failed_frames", [])
    if not failed:
        logger.info("[%s] No failed frames to retry.", model_id)
        return doc

    logger.info("[%s] Retrying %d failed frames: %s",
                model_id, len(failed), [f["frame_num"] for f in failed])

    _doc_ref(model_id).update({"failed_frames": [], "status": "generating"})
    return run_pipeline(model_id, display_name, color, force_regenerate=False)
