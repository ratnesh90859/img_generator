"""
pipeline.py
───────────
Orchestrates the full on-demand generation pipeline for a single vehicle model:

  1. Check Firestore — already completed? Return cached URLs immediately.
  2. Mark status = "generating" in Firestore.
  3. For each frame (1–36):
       a. Check if already in GCS (resumable — skip if exists).
       b. Generate via Vertex AI Imagen.
       c. Remove background with rembg.
       d. Quick alpha validation.
       e. Upload to GCS.
       f. Update progress in Firestore.
  4. Upload thumbnail (frame 1 resized).
  5. Run full validation report.
  6. Mark status = "completed" (or "failed") in Firestore.

The FastAPI route calls `run_pipeline()` as a background task.
"""

import logging
import time
import base64
import concurrent.futures
import threading
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
    """
    Fetch the Firestore document for a model.
    Returns None if the model has never been requested.
    """
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
    Main pipeline entry point. Designed to run as a background task.

    Args:
        model_id:          e.g. "honda-city"
        display_name:      e.g. "Honda City"
        color:             e.g. "Dashing Silver"
        force_regenerate:  If True, delete existing GCS frames and regenerate all.
        ref_front:         Optional base64 image string for front view (315-45 deg)
        ref_right:         Optional base64 image string for right view (45-135 deg)
        ref_rear:          Optional base64 image string for rear view (135-225 deg)
        ref_left:          Optional base64 image string for left view (225-315 deg)

    Returns:
        Final Firestore document dict.
    """
    doc_ref    = _doc_ref(model_id)
    started_at = datetime.now(timezone.utc)
    logger.info("=== Pipeline START: %s ===", model_id)

    # ── If force regenerate, clean up first ──────────────────────────────
    if force_regenerate:
        logger.info("Force regenerate — deleting existing GCS frames for %s", model_id)
        delete_frames(model_id)

    # ── Mark as generating ───────────────────────────────────────────────
    doc_ref.set({
        "model_id":       model_id,
        "display_name":   display_name,
        "color":          color,
        "status":         "generating",
        "total_frames":   TOTAL_FRAMES,
        "generated_frames": 0,
        "failed_frames":  [],
        "started_at":     started_at.isoformat(),
        "frame_urls":     [],
    }, merge=True)

    # We allocate a fixed list so we can preserve frame order.
    frame_urls = [None] * TOTAL_FRAMES
    failed_frames = []
    generated = 0
    lock = threading.Lock()

    # ── Decode references if provided ────────────────────────────────────
    ref_bytes = {
        "front": base64.b64decode(ref_front.split(",")[-1]) if ref_front else None,
        "right": base64.b64decode(ref_right.split(",")[-1]) if ref_right else None,
        "rear":  base64.b64decode(ref_rear.split(",")[-1])  if ref_rear  else None,
        "left":  base64.b64decode(ref_left.split(",")[-1])  if ref_left  else None,
    }

    # ── Generate frames concurrently ─────────────────────────────────────
    def process_frame(frame_num: int):
        nonlocal generated
        angle_deg = (frame_num - 1) * FRAME_ANGLE_STEP

        # Determine reference image by quadrant
        current_ref = None
        if angle_deg <= 45 or angle_deg >= 315:
            current_ref = ref_bytes["front"]
        elif 45 < angle_deg <= 135:
            current_ref = ref_bytes["right"]
        elif 135 < angle_deg <= 225:
            current_ref = ref_bytes["rear"]
        elif 225 < angle_deg <= 315:
            current_ref = ref_bytes["left"]

        # Skip frames already in GCS (resumable pipeline)
        if not force_regenerate and frame_exists(model_id, frame_num):
            gcs_path = gcs_frame_path(model_id, frame_num)
            with lock:
                frame_urls[frame_num - 1] = gcs_path
                generated += 1
            logger.info("  [%02d/%d] Skipped (cached) — %s", frame_num, TOTAL_FRAMES, gcs_path)
            return

        try:
            # Step 1: Generate
            png_bytes = generate_frame(display_name, color, angle_deg, ref_bytes=current_ref)

            # Step 2: Remove background (now with auto-centering)
            webp_bytes = remove_background(png_bytes)

            # Step 3: Quick quality check
            qc = quick_validate_alpha(webp_bytes)
            if not qc["valid"]:
                raise ValueError(f"Quality check failed: {qc['issues']}")

            # Step 4: Upload to GCS
            gcs_path = upload_frame(model_id, frame_num, webp_bytes)

            # Step 5: Update progress safely
            with lock:
                frame_urls[frame_num - 1] = gcs_path
                generated += 1
                # Periodically update Firestore so frontend gets progress, but throttle it slightly 
                # or just update it since Firestore can handle ~1 write/sec per document
                doc_ref.update({
                    "generated_frames": generated,
                    "status":           "generating",
                })

        except Exception as e:
            logger.error("  [%02d/%d] FAILED @ %d°: %s", frame_num, TOTAL_FRAMES, angle_deg, e)
            with lock:
                failed_frames.append({
                    "frame_num":   frame_num,
                    "angle_deg":   angle_deg,
                    "error":       str(e),
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                })
                doc_ref.update({"failed_frames": failed_frames})

    # Execute concurrent generation — 8 workers = ~45-60s total generation time
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_frame, fn) for fn in range(1, TOTAL_FRAMES + 1)]
        concurrent.futures.wait(futures)

    # Filter out None values in case of failed frames
    frame_urls = [u for u in frame_urls if u is not None]

    # ── Upload thumbnail (frame 1) ────────────────────────────────────────
    thumbnail_path = None
    try:
        if frame_urls:
            from google.cloud import storage as _gcs
            from config.settings import GCS_BUCKET_NAME
            bucket = _gcs.Client().bucket(GCS_BUCKET_NAME)
            thumb_bytes = bucket.blob(gcs_frame_path(model_id, 1)).download_as_bytes()
            thumbnail_path = upload_thumbnail(model_id, thumb_bytes)  # returns gcs_path
    except Exception as e:
        logger.warning("Thumbnail upload failed: %s", e)

    # ── Final status ──────────────────────────────────────────────────────
    completed_at  = datetime.now(timezone.utc)
    elapsed_s     = (completed_at - started_at).total_seconds()
    final_status  = "completed" if len(failed_frames) == 0 else (
                    "partial"   if len(frame_urls) > 0 else "failed"
                   )

    final_doc = {
        "status":            final_status,
        "generated_frames":  generated,
        "failed_frames":     failed_frames,
        "gcs_paths":         frame_urls,      # GCS object paths — signed at serve time
        "thumbnail_path":    thumbnail_path,
        "completed_at":      completed_at.isoformat(),
        "elapsed_seconds":   round(elapsed_s, 1),
    }
    doc_ref.update(final_doc)

    # ── Run full validation report ────────────────────────────────────────
    if final_status in ("completed", "partial"):
        try:
            full_validation_report(model_id)
        except Exception as e:
            logger.warning("Validation report failed: %s", e)

    logger.info("=== Pipeline END: %s | status=%s | frames=%d/%d | %.0fs ===",
                model_id, final_status, generated, TOTAL_FRAMES, elapsed_s)

    return {**final_doc, "model_id": model_id}


def retry_failed_frames(model_id: str, display_name: str, color: str = DEFAULT_COLOR) -> dict:
    """
    Re-run the pipeline only for frames listed in failed_frames.
    Useful for targeted retry without regenerating successful frames.
    """
    doc = get_vehicle_status(model_id)
    if not doc:
        raise ValueError(f"No Firestore record found for {model_id}")

    failed = doc.get("failed_frames", [])
    if not failed:
        logger.info("[%s] No failed frames to retry.", model_id)
        return doc

    logger.info("[%s] Retrying %d failed frames: %s",
                model_id, len(failed), [f["frame_num"] for f in failed])

    # Clear failed list before retry
    _doc_ref(model_id).update({"failed_frames": [], "status": "generating"})

    # Re-run full pipeline — existing GCS frames will be skipped automatically
    return run_pipeline(model_id, display_name, color, force_regenerate=False)
