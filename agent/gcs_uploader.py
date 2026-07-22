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

def upload_frame(model_id: str, frame_num: int, png_bytes: bytes) -> str:
    """
    Upload a single PNG frame to GCS (private).

    Returns:
        The GCS object path (not a URL — signing happens at API layer).
    """
    gcs_path = gcs_frame_path(model_id, frame_num)
    blob = _bucket.blob(gcs_path)
    blob.upload_from_string(png_bytes, content_type="image/png")
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

import concurrent.futures
import threading

_signing_credentials = None
_credentials_lock = threading.Lock()


def _get_signing_credentials():
    global _signing_credentials
    with _credentials_lock:
        if _signing_credentials is not None:
            return _signing_credentials

        import os
        import google.auth
        import google.auth.transport.requests
        from google.auth import iam as google_iam
        from google.oauth2 import service_account as sa_module

        # Step 1: get ADC credentials with cloud-platform scope
        credentials, project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        auth_request = google.auth.transport.requests.Request()
        credentials.refresh(auth_request)

        # Step 2: if credentials already have signing capability (SA key file),
        # use them directly.
        if hasattr(credentials, "service_account_email") and hasattr(credentials, "_signer"):
            _signing_credentials = credentials
            return _signing_credentials

        # Step 3: User ADC credentials (OAuth token) don't have a private key.
        # Use the IAM signBlob API via google.auth.iam.Signer.
        signing_sa_email = os.environ.get("SERVICE_ACCOUNT_EMAIL", "")

        if not signing_sa_email:
            raise RuntimeError(
                "Signed URL generation requires a service account.\n"
                "Add SERVICE_ACCOUNT_EMAIL=<sa>@<project>.iam.gserviceaccount.com to your .env\n"
                "and grant your user account roles/iam.serviceAccountTokenCreator on that SA."
            )

        signer = google_iam.Signer(
            request=auth_request,
            credentials=credentials,
            service_account_email=signing_sa_email,
        )
        _signing_credentials = sa_module.Credentials(
            signer=signer,
            service_account_email=signing_sa_email,
            token_uri="https://oauth2.googleapis.com/token",
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return _signing_credentials


def generate_signed_url(gcs_path: str, expiry_minutes: int = SIGNED_URL_EXPIRY_MINUTES) -> str:
    """
    Generate a time-limited signed URL for a private GCS object.
    Valid for `expiry_minutes` (default 60 min).
    """
    signing_credentials = _get_signing_credentials()
    blob = _bucket.blob(gcs_path)
    return blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=expiry_minutes),
        method="GET",
        credentials=signing_credentials,
    )


def generate_signed_urls_for_model(model_id: str, frame_paths: list[str]) -> list[str]:
    """
    Bulk generate signed URLs for all frames of a model in parallel.
    Called when the frontend requests a completed vehicle.
    """
    # Prime credentials first to avoid concurrent initialization overhead
    try:
        _get_signing_credentials()
    except Exception as e:
        logger.error("Failed to initialize signing credentials: %s", e)
        return [""] * len(frame_paths)

    signed_urls = [None] * len(frame_paths)

    def sign_one(index, path):
        try:
            url = generate_signed_url(path)
            signed_urls[index] = url
        except Exception as e:
            logger.error("Failed to sign URL for %s: %s", path, e)
            signed_urls[index] = ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(sign_one, i, path) for i, path in enumerate(frame_paths)]
        concurrent.futures.wait(futures)

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
