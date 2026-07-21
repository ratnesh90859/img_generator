"""
FastAPI application entry point.
"""

# Load .env FIRST — before any other module reads os.environ
from dotenv import load_dotenv
load_dotenv()

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.vehicles import router as vehicles_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

app = FastAPI(
    title="Kotak 360° Vehicle Image API",
    description="On-demand vehicle image generation for Kotak Mahindra Bank POC",
    version="1.0.0",
)

# ── CORS — allow your custom frontend origin ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────
app.include_router(vehicles_router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "service": "kotak-360-vehicle-api"}


@app.get("/", tags=["health"])
def root():
    return {
        "message": "Kotak 360° Vehicle Image Generation API",
        "docs":    "/docs",
        "health":  "/health",
    }
