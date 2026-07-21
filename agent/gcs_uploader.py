"""
gcs_uploader.py
───────────────
Handles all Google Cloud Storage interactions.

NOTE: The testing-ratnesh project has Public Access Prevention enforced by
org policy. Files are kept PRIVATE in GCS. The API layer generates
time-limited Signed URLs (default 60 min) for the frontend to access frames.
"""

import logging
import datetime
from io import BytesIO

from google.cloud import storage
from config.settings import (
    GCS_BUCKET_NAME,
    gcs_frame_path,
    gcs_thumbnail_path,
    SIGNED_URL_EXPIRY_MINUTES,
)

logger = logging.getLogger(__name__)

_client = storage.Client()
_bucket = _client.bucket(GCS_BUCKET_NAME)


# ─────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────

def upload_frame(model_id: str, frame_num: int, webp_bytes: bytes) -> str:
    """
    Upload a single WebP frame to GCS (private).

    Returns:
        The GCS object path (not a URL — signing happens at API layer).
    """
    gcs_path = gcs_frame_path(model_id, frame_num)
    blob = _bucket.blob(gcs_path)
    blob.upload_from_string(webp_bytes, content_type="image/webp")
    logger.info("  ✓ Uploaded frame %02d → gs://%s/%s", frame_num, GCS_BUCKET_NAME, gcs_path)
    return gcs_path   # return path, not URL


def upload_thumbnail(model_id: str, webp_bytes: bytes) -> str:
    """Upload thumbnail (frame 1 resized to 400×400). Returns GCS path."""
    from PIL import Image
    img = Image.open(BytesIO(webp_bytes)).convert("RGBA")
    img.thumbnail((400, 400), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="WEBP", lossless=False, quality=85)

    gcs_path = gcs_thumbnail_path(model_id)
    blob = _bucket.blob(gcs_path)
    blob.upload_from_string(buf.getvalue(), content_type="image/webp")
    logger.info("  ✓ Uploaded thumbnail → gs://%s/%s", GCS_BUCKET_NAME, gcs_path)
    return gcs_path


# ─────────────────────────────────────────────────────────────────
# Signed URL generation  (called by API layer)
# ─────────────────────────────────────────────────────────────────

def generate_signed_url(gcs_path: str, expiry_minutes: int = SIGNED_URL_EXPIRY_MINUTES) -> str:
    """
    Generate a time-limited signed URL for a private GCS object.
    Valid for `expiry_minutes` (default 60 min).
    Uses Application Default Credentials — no service account key needed.
    """
    import google.auth
    import google.auth.transport.requests
    from google.auth import impersonated_credentials

    blob = _bucket.blob(gcs_path)

    # Use ADC with token refresh for signing
    credentials, _ = google.auth.default()
    if hasattr(credentials, 'with_scopes'):
        credentials = credentials.with_scopes(
            ['https://www.googleapis.com/auth/cloud-platform']
        )
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)

    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=expiry_minutes),
        method="GET",
        credentials=credentials,
    )
    return url


def generate_signed_urls_for_model(model_id: str, frame_paths: list[str]) -> list[str]:
    """
    Bulk generate signed URLs for all frames of a model.
    Called when the frontend requests a completed vehicle.
    """
    signed_urls = []
    for path in frame_paths:
        try:
            url = generate_signed_url(path)
            signed_urls.append(url)
        except Exception as e:
            logger.error("Failed to sign URL for %s: %s", path, e)
            signed_urls.append("")   # empty string marks failed signing
    return signed_urls


# ─────────────────────────────────────────────────────────────────
# Cache checks  (used by pipeline for resume support)
# ─────────────────────────────────────────────────────────────────

def frame_exists(model_id: str, frame_num: int) -> bool:
    """Check if a frame already exists in GCS (for resumable generation)."""
    return _bucket.blob(gcs_frame_path(model_id, frame_num)).exists()


def list_existing_frames(model_id: str) -> list[int]:
    """
    Return sorted list of frame numbers that already exist in GCS.
    """
    prefix = f"processed/{model_id}/"
    blobs = list(_client.list_blobs(GCS_BUCKET_NAME, prefix=prefix))
    frame_nums = []
    for blob in blobs:
        filename = blob.name.split("/")[-1]
        if filename.startswith("frame_") and filename.endswith(".webp"):
            try:
                num = int(filename.replace("frame_", "").replace(".webp", ""))
                frame_nums.append(num)
            except ValueError:
                pass
    return sorted(frame_nums)


def delete_frames(model_id: str):
    """Delete all frames for a model (used on force-regenerate)."""
    prefix = f"processed/{model_id}/"
    blobs = list(_client.list_blobs(GCS_BUCKET_NAME, prefix=prefix))
    for blob in blobs:
        blob.delete()
    logger.info("Deleted %d frames for %s", len(blobs), model_id)
