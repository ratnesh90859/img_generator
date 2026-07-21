"""
Pydantic response models for the FastAPI layer.
"""

from pydantic import BaseModel
from typing import Optional, List


class VehicleStatusResponse(BaseModel):
    model_id:         str
    display_name:     str
    status:           str              # pending | generating | completed | partial | failed
    total_frames:     int
    generated_frames: int
    frame_urls:       List[str]        # empty until completed
    thumbnail_url:    Optional[str]    = None
    failed_frames:    List[dict]       = []
    elapsed_seconds:  Optional[float]  = None
    started_at:       Optional[str]    = None
    completed_at:     Optional[str]    = None


class GenerateRequest(BaseModel):
    model_id:     str
    display_name: str
    color:        Optional[str] = "Dashing Silver"
    force:        Optional[bool] = False   # set True to regenerate even if cached
    ref_front:    Optional[str] = None     # Base64 encoded image
    ref_right:    Optional[str] = None     # Base64 encoded image
    ref_rear:     Optional[str] = None     # Base64 encoded image
    ref_left:     Optional[str] = None     # Base64 encoded image


class ValidationReport(BaseModel):
    model_id:      str
    overall_valid: bool
    frame_count:   dict
    quality:       dict
    can_retry:     bool


class AuditReport(BaseModel):
    total_checked: int
    issues_found:  int
    models:        List[dict]
