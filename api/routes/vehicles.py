"""
routes/vehicles.py
──────────────────
All vehicle-related API endpoints:

  GET  /api/vehicles                       — list all models from Firestore
  GET  /api/vehicles/{model_id}            — get status + frame URLs (triggers generation if new)
  GET  /api/vehicles/{model_id}/status     — lightweight status poll (for frontend progress bar)
  POST /api/vehicles/generate              — trigger generation (body: GenerateRequest)
  POST /api/vehicles/{model_id}/retry      — retry only failed frames
  GET  /api/vehicles/{model_id}/validate   — run full validation report
  DELETE /api/vehicles/{model_id}          — delete model + GCS frames (admin)
"""

import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.models import VehicleStatusResponse, GenerateRequest, ValidationReport
from agent.pipeline import (
    run_pipeline,
    get_vehicle_status,
    retry_failed_frames,
)
from agent.validator import full_validation_report, audit_all_models
from agent.gcs_uploader import delete_frames, generate_signed_urls_for_model, generate_signed_url
from config.settings import FIRESTORE_COLLECTION, TOTAL_FRAMES, DEFAULT_COLOR
from google.cloud import firestore

router = APIRouter(prefix="/api/vehicles", tags=["vehicles"])
logger = logging.getLogger(__name__)
_fs    = firestore.Client()


# ─────────────────────────────────────────────────────────────────
# LIST all vehicles
# ─────────────────────────────────────────────────────────────────
@router.get("/", summary="List all vehicle models")
def list_vehicles():
    docs = _fs.collection(FIRESTORE_COLLECTION).stream()
    vehicles = []
    for doc in docs:
        d = doc.to_dict()
        vehicles.append({
            "model_id":     d.get("model_id"),
            "display_name": d.get("display_name"),
            "status":       d.get("status"),
            "thumbnail_url": d.get("thumbnail_url"),
        })
    return {"vehicles": vehicles, "total": len(vehicles)}


# ─────────────────────────────────────────────────────────────────
# GET vehicle — cache-first, triggers generation if new
# ─────────────────────────────────────────────────────────────────
@router.get("/{model_id}", summary="Get vehicle frames (triggers generation if first request)")
def get_vehicle(model_id: str, background_tasks: BackgroundTasks):
    doc = get_vehicle_status(model_id)

    # ── Already completed → return immediately ────────────────────
    if doc and doc.get("status") == "completed":
        return _build_response(doc, 200)

    # ── Currently generating → return progress ────────────────────
    if doc and doc.get("status") in ("generating", "pending"):
        return _build_response(doc, 202)

    # ── Partial/failed → return what we have, allow retry ─────────
    if doc and doc.get("status") in ("partial", "failed"):
        return _build_response(doc, 206)

    # ── Never seen before → trigger background generation ─────────
    raise HTTPException(
        status_code=404,
        detail={
            "message": f"Vehicle '{model_id}' not found. "
                       "POST to /api/vehicles/generate to start generation.",
            "model_id": model_id,
        }
    )


# ─────────────────────────────────────────────────────────────────
# STATUS — lightweight poll endpoint
# ─────────────────────────────────────────────────────────────────
@router.get("/{model_id}/status", summary="Poll generation status")
def get_status(model_id: str):
    doc = get_vehicle_status(model_id)
    if not doc:
        raise HTTPException(404, detail=f"Vehicle '{model_id}' not found")

    generated = doc.get("generated_frames", 0)
    total     = doc.get("total_frames", TOTAL_FRAMES)
    progress  = round((generated / total) * 100, 1) if total > 0 else 0

    return {
        "model_id":         model_id,
        "status":           doc.get("status"),
        "generated_frames": generated,
        "total_frames":     total,
        "progress_pct":     progress,
        "failed_count":     len(doc.get("failed_frames", [])),
        "frame_urls":       doc.get("frame_urls", []) if doc.get("status") == "completed" else [],
    }


# ─────────────────────────────────────────────────────────────────
# GENERATE — trigger on-demand generation
# ─────────────────────────────────────────────────────────────────
@router.post("/generate", summary="Trigger on-demand image generation")
def trigger_generation(req: GenerateRequest, background_tasks: BackgroundTasks):
    doc = get_vehicle_status(req.model_id)

    # Already completed and not forced → return cached result
    if doc and doc.get("status") == "completed" and not req.force:
        return {
            "message":  "Already generated — returning cached result.",
            "status":   "completed",
            "model_id": req.model_id,
            "frame_urls": doc.get("frame_urls", []),
        }

    # Already running → don't double-trigger
    if doc and doc.get("status") in ("generating", "pending") and not req.force:
        return {
            "message":  "Generation already in progress.",
            "status":   "generating",
            "model_id": req.model_id,
            "poll_url": f"/api/vehicles/{req.model_id}/status",
        }

    # Create/update Firestore doc with pending status immediately
    _fs.collection(FIRESTORE_COLLECTION).document(req.model_id).set({
        "model_id":         req.model_id,
        "display_name":     req.display_name,
        "color":            req.color or DEFAULT_COLOR,
        "status":           "pending",
        "total_frames":     TOTAL_FRAMES,
        "generated_frames": 0,
        "failed_frames":    [],
        "frame_urls":       [],
    }, merge=True)

    # Run pipeline in the background (non-blocking)
    background_tasks.add_task(
        run_pipeline,
        req.model_id,
        req.display_name,
        req.color or DEFAULT_COLOR,
        req.force or False,
        req.ref_front,
        req.ref_right,
        req.ref_rear,
        req.ref_left,
    )

    logger.info("Generation triggered for %s", req.model_id)
    return {
        "message":  "Generation started.",
        "status":   "pending",
        "model_id": req.model_id,
        "poll_url": f"/api/vehicles/{req.model_id}/status",
    }


# ─────────────────────────────────────────────────────────────────
# RETRY — only re-run failed frames
# ─────────────────────────────────────────────────────────────────
@router.post("/{model_id}/retry", summary="Retry only failed frames")
def retry_vehicle(model_id: str, background_tasks: BackgroundTasks):
    doc = get_vehicle_status(model_id)
    if not doc:
        raise HTTPException(404, detail=f"Vehicle '{model_id}' not found")

    failed = doc.get("failed_frames", [])
    if not failed:
        return {"message": "No failed frames to retry.", "model_id": model_id}

    background_tasks.add_task(
        retry_failed_frames,
        model_id,
        doc.get("display_name", model_id),
        doc.get("color", DEFAULT_COLOR),
    )
    return {
        "message":      f"Retrying {len(failed)} failed frame(s).",
        "model_id":     model_id,
        "retry_frames": [f["frame_num"] for f in failed],
        "poll_url":     f"/api/vehicles/{model_id}/status",
    }


# ─────────────────────────────────────────────────────────────────
# VALIDATE — run full 3-level validation report
# ─────────────────────────────────────────────────────────────────
@router.get("/{model_id}/validate", summary="Run full validation report")
def validate_vehicle(model_id: str):
    doc = get_vehicle_status(model_id)
    if not doc:
        raise HTTPException(404, detail=f"Vehicle '{model_id}' not found")

    report = full_validation_report(model_id)
    return report


# ─────────────────────────────────────────────────────────────────
# AUDIT — health check across all models
# ─────────────────────────────────────────────────────────────────
@router.get("/audit/all", summary="Audit all completed models for missing frames")
def audit_vehicles():
    issues = audit_all_models()
    return {
        "issues_found": len(issues),
        "models":       issues,
    }


# ─────────────────────────────────────────────────────────────────
# DELETE — admin: remove model + GCS frames
# ─────────────────────────────────────────────────────────────────
@router.delete("/{model_id}", summary="Delete model and all GCS frames (admin)")
def delete_vehicle(model_id: str):
    delete_frames(model_id)
    _fs.collection(FIRESTORE_COLLECTION).document(model_id).delete()
    return {"message": f"Vehicle '{model_id}' deleted from GCS and Firestore."}


# ── Helper: build response, signing GCS paths → URLs ──────────────────────────────────
def _build_response(doc: dict, status_code: int) -> dict:
    """
    Convert stored GCS object paths into fresh signed URLs before returning
    to the frontend. Signing only happens for completed models.
    """
    gcs_paths     = doc.get("gcs_paths", [])
    thumbnail_path = doc.get("thumbnail_path")
    status         = doc.get("status")

    # Only sign URLs if generation is done
    if status == "completed" and gcs_paths:
        signed_frame_urls = generate_signed_urls_for_model(doc.get("model_id", ""), gcs_paths)
    else:
        signed_frame_urls = []   # frontend should poll /status, not use these yet

    signed_thumbnail = generate_signed_url(thumbnail_path) if thumbnail_path else None

    return {
        "model_id":         doc.get("model_id"),
        "display_name":     doc.get("display_name"),
        "status":           status,
        "total_frames":     doc.get("total_frames", TOTAL_FRAMES),
        "generated_frames": doc.get("generated_frames", 0),
        "frame_urls":       signed_frame_urls,
        "thumbnail_url":    signed_thumbnail,
        "failed_frames":    doc.get("failed_frames", []),
        "elapsed_seconds":  doc.get("elapsed_seconds"),
        "note":             "Frame URLs are signed and valid for 60 minutes."
                            if signed_frame_urls else None,
    }
