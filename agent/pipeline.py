"""
pipeline.py
───────────
Orchestrates the full 360° image generation pipeline for a single vehicle model.

Speed Strategy — PARALLEL Generation:
  Since all frame prompts are self-contained (angle baked into text, no chaining),
  all frames can be generated simultaneously in a ThreadPoolExecutor.

  Old sequential: 36 frames × ~25s each = ~15–50 min
  New parallel:   36 frames / 8 workers × ~25s = ~2–4 min  ✅

Pipeline Steps:
  1. Generate Frame 0° (front view) — this becomes the Master Identity Reference.
  2. Generate ALL remaining frames in parallel (8 workers), each gets:
       - The user prompt with angle baked in
       - Master Frame (0°) as optional visual identity anchor
  3. Post-process each frame: PNG → upload to GCS.
  4. Track progress in Firestore.
  5. Upload thumbnail.
  6. Run validation.
  7. Mark completed / partial / failed in Firestore.
"""

import logging
import time
import base64
import concurrent.futures
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

# How many Gemini API calls to run concurrently.
# Gemini Vertex AI quota is ~10 RPM per model on free tier, 60+ on paid.
# Set to 8 to stay comfortably within quota while maximising speed.
PARALLEL_WORKERS = 8


def _doc_ref(model_id: str):
    return _fs.collection(FIRESTORE_COLLECTION).document(model_id)


def get_vehicle_status(model_id: str) -> Optional[dict]:
    """Fetch the Firestore document for a model. Returns None if not found."""
    doc = _doc_ref(model_id).get()
    return doc.to_dict() if doc.exists else None


def _generate_and_upload_frame(
    model_id: str,
    frame_num: int,
    angle_deg: int,
    display_name: str,
    color: str,
    master_frame_bytes: Optional[bytes],
    ref_bytes: Optional[bytes],
    doc_ref,
    frame_urls: list,
    failed_frames: list,
    generated_counter: list,   # [int] — mutable counter for thread-safe increment
    lock: "concurrent.futures.thread._worker_adjustor",  # threading.Lock
) -> bool:
    """
    Worker function: generate one frame, post-process, upload to GCS.
    Called by ThreadPoolExecutor.
    Returns True on success, False on failure.
    """
    import threading
    try:
        # Generate
        png_bytes = generate_frame(
            display_name,
            color,
            angle_deg,
            ref_bytes=ref_bytes,
            master_frame_bytes=master_frame_bytes,
        )

        # Post-process (PNG output, white bg preserved / transparent)
        out_bytes = remove_background(png_bytes)

        # Quality check
        qc = quick_validate_alpha(out_bytes)
        if not qc["valid"]:
            raise ValueError(f"Quality check failed: {qc['issues']}")

        # Upload to GCS
        gcs_path = upload_frame(model_id, frame_num, out_bytes)

        # Update shared state (thread-safe)
        with lock:
            frame_urls[frame_num - 1] = gcs_path
            generated_counter[0] += 1
            count = generated_counter[0]

        # Update Firestore progress
        doc_ref.update({
            "generated_frames": count,
            "status": "generating",
        })

        logger.info("  ✅ [%02d/%d] @ %d° uploaded → %s", frame_num, TOTAL_FRAMES, angle_deg, gcs_path)
        return True

    except Exception as e:
        logger.error("  ❌ [%02d/%d] FAILED @ %d°: %s", frame_num, TOTAL_FRAMES, angle_deg, e)
        with lock:
            failed_frames.append({
                "frame_num": frame_num,
                "angle_deg": angle_deg,
                "error":     str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        doc_ref.update({"failed_frames": failed_frames})
        return False


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

    Generates all 36 frames in parallel for maximum speed (~2-4 min vs 50 min sequential).

    Args:
        model_id:          e.g. "tata-nexon"
        display_name:      e.g. "Tata Nexon"
        color:             e.g. "Dashing Silver"
        force_regenerate:  If True, delete existing GCS frames and regenerate all.
        ref_front/right/rear/left: Optional base64 reference images per quadrant.

    Returns:
        Final Firestore document dict.
    """
    import threading

    doc_ref    = _doc_ref(model_id)
    started_at = datetime.now(timezone.utc)
    logger.info("=== Pipeline START: %s | parallel_workers=%d ===", model_id, PARALLEL_WORKERS)

    # ── Force regenerate: wipe existing GCS frames ───────────────────────────
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

    # Shared state (written by parallel workers)
    frame_urls        = [None] * TOTAL_FRAMES
    failed_frames     = []
    generated_counter = [0]   # mutable list for thread-safe int
    lock              = threading.Lock()

    # ── STEP 1: Generate Frame 0° (Front View) first ─────────────────────────
    # This gives us a master identity reference for all subsequent frames.
    # Frame 1 = angle 0°, frame_num = 1.
    master_frame_bytes: Optional[bytes] = None

    if not force_regenerate and frame_exists(model_id, 1):
        logger.info("  [01/%d] Skipped (cached) — using existing frame 1 as master", TOTAL_FRAMES)
        from google.cloud import storage as _gcs
        from config.settings import GCS_BUCKET_NAME
        try:
            master_frame_bytes = _gcs.Client().bucket(GCS_BUCKET_NAME).blob(
                gcs_frame_path(model_id, 1)
            ).download_as_bytes()
        except Exception:
            master_frame_bytes = None
        frame_urls[0] = gcs_frame_path(model_id, 1)
        with lock:
            generated_counter[0] += 1
    else:
        logger.info("  [01/%d] Generating MASTER FRAME @ 0° (front view)...", TOTAL_FRAMES)
        ref_0 = ref_bytes_map["front"]
        success = _generate_and_upload_frame(
            model_id, 1, 0, display_name, color,
            None,   # no master ref yet — this IS the master
            ref_0, doc_ref, frame_urls, failed_frames, generated_counter, lock
        )
        if success:
            # Download the just-uploaded frame to use as master reference
            from google.cloud import storage as _gcs
            from config.settings import GCS_BUCKET_NAME
            try:
                master_frame_bytes = _gcs.Client().bucket(GCS_BUCKET_NAME).blob(
                    gcs_frame_path(model_id, 1)
                ).download_as_bytes()
                logger.info("  ✓ Master identity frame loaded (%d bytes)", len(master_frame_bytes))
            except Exception as e:
                logger.warning("  Could not load master frame for reference: %s", e)
                master_frame_bytes = None

    # ── STEP 2: Generate ALL remaining frames IN PARALLEL ────────────────────
    # Frames 2–36 (angles 10°–350°) run simultaneously.
    # Each frame gets: angle prompt + master frame (0°) as identity anchor.

    remaining_tasks = []  # list of (frame_num, angle_deg, ref_quadrant)

    for frame_num in range(2, TOTAL_FRAMES + 1):
        angle_deg = (frame_num - 1) * FRAME_ANGLE_STEP

        # Skip frames already in GCS (resumable pipeline)
        if not force_regenerate and frame_exists(model_id, frame_num):
            gcs_path = gcs_frame_path(model_id, frame_num)
            frame_urls[frame_num - 1] = gcs_path
            with lock:
                generated_counter[0] += 1
            logger.info("  [%02d/%d] Skipped (cached)", frame_num, TOTAL_FRAMES)
            continue

        # Select quadrant reference
        if angle_deg <= 45 or angle_deg >= 315:
            qref = ref_bytes_map["front"]
        elif 45 < angle_deg <= 135:
            qref = ref_bytes_map["right"]
        elif 135 < angle_deg <= 225:
            qref = ref_bytes_map["rear"]
        else:
            qref = ref_bytes_map["left"]

        remaining_tasks.append((frame_num, angle_deg, qref))

    if remaining_tasks:
        logger.info(
            "  Launching %d frames in parallel (%d workers)...",
            len(remaining_tasks), PARALLEL_WORKERS
        )
        t_parallel_start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {
                executor.submit(
                    _generate_and_upload_frame,
                    model_id, frame_num, angle_deg, display_name, color,
                    master_frame_bytes, qref,
                    doc_ref, frame_urls, failed_frames, generated_counter, lock
                ): frame_num
                for frame_num, angle_deg, qref in remaining_tasks
            }
            for future in concurrent.futures.as_completed(futures):
                frame_num = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error("  Unexpected error in frame %d: %s", frame_num, e)

        parallel_elapsed = time.perf_counter() - t_parallel_start
        logger.info(
            "  Parallel batch done — %d frames in %.1fs",
            len(remaining_tasks), parallel_elapsed
        )

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
