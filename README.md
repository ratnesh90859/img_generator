# 🚗 360° Vehicle Image Generator

AI-powered 360° car image generation using **Google Gemini** (Vertex AI) with a live interactive viewer. Generate 36 consistent frames of any car model, served via a FastAPI backend and a beautiful drag-to-spin frontend.

---

## ✨ Features

- **AI Generation**: Uses `gemini-2.5-flash-image` (Vertex AI) to generate photorealistic car images
- **Reference Images**: Upload 4 reference images (Front/Right/Rear/Left) to enforce visual consistency
- **Smart Background Removal**: Auto-detects and removes any background color (white or dark)
- **Parallel Generation**: 8 concurrent workers — 36 frames in ~60–90 seconds
- **360° Viewer**: Drag-to-rotate + scrubber slider, Car Dekho–style UI
- **GCS + Firestore**: All frames stored in GCS, metadata in Firestore, served via Signed URLs
- **Resumable**: Pipeline skips already-generated frames on restart

---

## 📁 Project Structure

```
img_generator/
├── api/
│   ├── main.py                # FastAPI app entry point
│   ├── models.py              # Pydantic request/response models
│   └── routes/
│       └── vehicles.py        # All REST endpoints
├── agent/
│   ├── image_generator.py     # Gemini multimodal image generation
│   ├── background_remover.py  # Smart BFS background removal (Pillow)
│   ├── gcs_uploader.py        # GCS upload + Signed URL generation
│   ├── pipeline.py            # Orchestration (parallel, resumable)
│   └── validator.py           # Frame validation + audit reports
├── config/
│   └── settings.py            # All GCP config loaded from .env
├── frontend/
│   └── index.html             # 360° viewer (pure HTML/CSS/JS)
├── scripts/
│   ├── setup_gcp.sh           # One-time GCP infrastructure setup
│   └── audit.py               # Admin health check script
├── .env.example               # Environment variable template
├── Dockerfile                 # Container definition
├── requirements.txt
└── README.md
```

---

## 🧰 Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| gcloud CLI | Latest | [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install) |
| Git | Any | `brew install git` |
| GCP Project | — | Must have billing enabled |

---

## 🚀 Mac Setup (Step by Step)

### 1. Clone the Repository

```bash
git clone https://github.com/ratnesh90859/img_generator.git
cd img_generator
```

### 2. Create a Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure GCP Authentication

```bash
# Install gcloud if not already installed
brew install --cask google-cloud-sdk

# Log in with Application Default Credentials
gcloud auth application-default login

# Set your GCP project
gcloud config set project testing-ratnesh
```

### 5. Set Up GCP Infrastructure (run once)

This creates the GCS bucket, Firestore database, and enables all required APIs:

```bash
bash scripts/setup_gcp.sh
```

> **What it enables:**
> - Vertex AI API
> - Cloud Storage API  
> - Firestore API
> - Cloud Run API (for future deployment)

### 6. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=asia-south1
GCS_BUCKET_NAME=kotak-vehicles-360
FIRESTORE_COLLECTION=vehicles
TOTAL_FRAMES=36
DEFAULT_CAR_COLOR=Dashing Silver
```

### 7. Run the API Server

Open **Terminal 1**:

```bash
source venv/bin/activate
export PYTHONPATH=$(pwd)
uvicorn api.main:app --port 8080 --log-level warning
```

### 8. Run the Frontend

Open **Terminal 2**:

```bash
python3 -m http.server 3000 --directory frontend
```

### 9. Open the App

Open your browser and go to: **http://localhost:3000**

---

## 🎮 How to Use

### Generate a 360° View

1. Open **http://localhost:3000**
2. Select a vehicle from the dropdown
3. *(Optional)* Click **⚙️ Advanced** to upload 4 reference images for consistency
4. Click **⚡ Generate**
5. Watch the progress bar — ~60–90 seconds to generate all 36 frames
6. Click the card to open the **360° Viewer**
7. **Drag** the car or use the **scrubber slider** to spin it

### Force Regenerate

Check **"Force Regenerate"** in the Advanced panel before clicking Generate to delete old cached images and regenerate fresh.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/vehicles/` | List all generated models |
| `GET` | `/api/vehicles/{id}` | Get model status + signed frame URLs |
| `GET` | `/api/vehicles/{id}/status` | Lightweight progress poll |
| `POST` | `/api/vehicles/generate` | Trigger on-demand generation |
| `POST` | `/api/vehicles/{id}/retry` | Retry only failed frames |
| `GET` | `/api/vehicles/{id}/validate` | Full validation report |
| `GET` | `/api/vehicles/audit/all` | Health check all models |
| `DELETE` | `/api/vehicles/{id}` | Delete model + all GCS frames |

### Example: Trigger Generation

```bash
curl -X POST http://localhost:8080/api/vehicles/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "honda-city",
    "display_name": "Honda City",
    "color": "Dashing Silver",
    "force": true
  }'
```

### Example: With Reference Images

```bash
curl -X POST http://localhost:8080/api/vehicles/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "honda-city",
    "display_name": "Honda City",
    "color": "Dashing Silver",
    "ref_front": "data:image/jpeg;base64,/9j/4AAQ...",
    "ref_right": "data:image/jpeg;base64,/9j/4AAQ...",
    "ref_rear":  "data:image/jpeg;base64,/9j/4AAQ...",
    "ref_left":  "data:image/jpeg;base64,/9j/4AAQ..."
  }'
```

### Example: Poll Status

```bash
curl http://localhost:8080/api/vehicles/honda-city/status
```

```json
{
  "status": "generating",
  "generated_frames": 24,
  "total_frames": 36,
  "progress_pct": 66.7
}
```

---

## 🏗️ Architecture

```
Browser (frontend/)
    │
    ▼
FastAPI (api/) ─────────────────── Firestore
    │                               (status / metadata)
    ▼
Pipeline (agent/pipeline.py)
    │ 8 parallel workers
    ├─► Gemini (Vertex AI) ──── generates PNG
    ├─► Background Remover ──── removes any BG color
    ├─► Validator ─────────────── quality check
    └─► GCS Uploader ─────────── stores RGBA WebP
                                   │
                                   ▼
                              Signed URL ──► Browser
```

---

## 🔧 Troubleshooting

### "API Unreachable" on frontend

```bash
# Make sure both servers are running
# Terminal 1 (API):
export PYTHONPATH=$(pwd)
uvicorn api.main:app --port 8080

# Terminal 2 (Frontend):
python3 -m http.server 3000 --directory frontend
```

### Authentication Errors

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Vertex AI Permission Errors

```bash
# Grant yourself the Vertex AI User role
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:YOUR_EMAIL" \
  --role="roles/aiplatform.user"
```

### GCS Access Denied (Org Policy)

This project is configured to use **Signed URLs** — public access is not required. The org-level "Public Access Prevention" policy is compatible with this setup.

---

## 🐳 Docker (Optional)

```bash
# Build
docker build -t img-generator .

# Run
docker run -p 8080:8080 \
  -e GCP_PROJECT_ID=testing-ratnesh \
  -v ~/.config/gcloud:/root/.config/gcloud \
  img-generator
```

---

## 🌐 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT_ID` | `testing-ratnesh` | Your GCP project ID |
| `GCP_REGION` | `asia-south1` | GCP region for Firestore |
| `GCS_BUCKET_NAME` | `kotak-vehicles-360` | GCS bucket for frames |
| `FIRESTORE_COLLECTION` | `vehicles` | Firestore collection name |
| `TOTAL_FRAMES` | `36` | Number of frames per 360° spin |
| `DEFAULT_CAR_COLOR` | `Dashing Silver` | Default car color if not specified |

---

## 📝 Notes

- **Gemini model** is locked to `us-central1` region (image generation only available there)
- **Firestore** is in `asia-south1` (low latency for India)
- **rembg is NOT used** — replaced with a custom Pillow BFS flood-fill that auto-detects background color
- Generated frames are stored as **lossless RGBA WebP** at 1024×1024
- All GCS URLs are **Signed** (60-minute expiry) — no public access needed

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

*Built with ❤️ using Google Gemini, FastAPI, and Vertex AI*
