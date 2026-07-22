"""
Central configuration for the 360° Vehicle Image Generation POC.
All values are read from environment variables with safe defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# GCP
# ---------------------------------------------------------------------------
GCP_PROJECT_ID  = os.getenv("GCP_PROJECT_ID", "testing-ratnesh")
GCP_REGION      = os.getenv("GCP_REGION", "asia-south1")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "kotak-vehicles-360")

# ---------------------------------------------------------------------------
# Firestore
# ---------------------------------------------------------------------------
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "vehicles")

# ---------------------------------------------------------------------------
# Vertex AI / Imagen
# NOTE: gemini image models only available in us-central1 (not asia-south1)
#       The generator module handles this internally.
# ---------------------------------------------------------------------------
IMAGEN_MODEL    = os.getenv("IMAGEN_MODEL", "gemini-2.5-flash-image")
IMAGE_WIDTH     = int(os.getenv("IMAGE_WIDTH", "1024"))
IMAGE_HEIGHT    = int(os.getenv("IMAGE_HEIGHT", "1024"))
TOTAL_FRAMES    = int(os.getenv("TOTAL_FRAMES", "36"))        # 36 × 10° = 360°
FRAME_ANGLE_STEP = 360 // TOTAL_FRAMES                         # 10°

# ---------------------------------------------------------------------------
# Vehicle Models (POC — 4 cars)
# ---------------------------------------------------------------------------
POC_VEHICLES: List[dict] = [
    {"model_id": "honda-city",       "display_name": "Honda City",           "brand": "Honda"},
    {"model_id": "hyundai-creta",    "display_name": "Hyundai Creta",        "brand": "Hyundai"},
    {"model_id": "maruti-baleno",    "display_name": "Maruti Suzuki Baleno", "brand": "Maruti Suzuki"},
    {"model_id": "tata-harrier",     "display_name": "Tata Harrier",         "brand": "Tata"},
]

DEFAULT_COLOR = os.getenv("DEFAULT_CAR_COLOR", "Dashing Silver")

# ---------------------------------------------------------------------------
# GCS path helpers
# ---------------------------------------------------------------------------
def gcs_frame_path(model_id: str, frame_num: int) -> str:
    """Returns the GCS object path for a given model frame (PNG format)."""
    return f"processed/{model_id}/frame_{str(frame_num).zfill(3)}.png"

def gcs_thumbnail_path(model_id: str) -> str:
    return f"thumbnails/{model_id}.webp"

# NOTE: Public access is blocked by org policy on testing-ratnesh.
# The API generates Signed URLs (1-hour expiry) instead.
# These helpers below are used by the API layer — NOT the agent pipeline.
SIGNED_URL_EXPIRY_MINUTES = int(os.getenv("SIGNED_URL_EXPIRY_MINUTES", "60"))
