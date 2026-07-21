#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_gcp.sh — One-time GCP setup for the Kotak 360° Vehicle POC
#
# Run this ONCE before starting the API.
# Requirements: gcloud CLI installed and authenticated
#   gcloud auth application-default login
#   gcloud config set project testing-ratnesh
# ─────────────────────────────────────────────────────────────────────────────

set -e

PROJECT_ID="testing-ratnesh"
REGION="asia-south1"
BUCKET_NAME="kotak-vehicles-360"

echo "=== Kotak 360° POC — GCP Setup ==="
echo "Project : $PROJECT_ID"
echo "Region  : $REGION"
echo "Bucket  : $BUCKET_NAME"
echo ""

# ── 1. Set active project ─────────────────────────────────────────────────
echo "[1/6] Setting active project..."
gcloud config set project $PROJECT_ID

# ── 2. Enable required APIs ───────────────────────────────────────────────
echo "[2/6] Enabling APIs (this may take 1-2 minutes)..."
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  firestore.googleapis.com \
  run.googleapis.com \
  --project=$PROJECT_ID

echo "  ✓ APIs enabled"

# ── 3. Create GCS bucket ──────────────────────────────────────────────────
echo "[3/6] Creating GCS bucket..."
if gsutil ls gs://$BUCKET_NAME &>/dev/null; then
  echo "  ⚠️  Bucket $BUCKET_NAME already exists — skipping creation"
else
  gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME
  echo "  ✓ Bucket created: gs://$BUCKET_NAME"
fi

# Make bucket publicly readable (for POC — images served via public URLs)
gsutil iam ch allUsers:objectViewer gs://$BUCKET_NAME
echo "  ✓ Bucket set to public read"

# Enable CORS on bucket (for custom frontend)
cat > /tmp/cors.json << 'EOF'
[
  {
    "origin": ["*"],
    "method": ["GET", "HEAD"],
    "responseHeader": ["Content-Type", "Cache-Control"],
    "maxAgeSeconds": 3600
  }
]
EOF
gsutil cors set /tmp/cors.json gs://$BUCKET_NAME
echo "  ✓ CORS configured on bucket"

# ── 4. Create Firestore database ──────────────────────────────────────────
echo "[4/6] Setting up Firestore..."
# Create in Native mode (needed for real-time + queries)
gcloud firestore databases create \
  --location=$REGION \
  --type=firestore-native \
  --project=$PROJECT_ID 2>/dev/null || echo "  ⚠️  Firestore already exists — skipping"
echo "  ✓ Firestore ready"

# ── 5. Create Firestore index for status queries ──────────────────────────
echo "[5/6] Creating Firestore composite index..."
cat > /tmp/firestore.indexes.json << 'EOF'
{
  "indexes": [
    {
      "collectionGroup": "vehicles",
      "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "status",     "order": "ASCENDING" },
        { "fieldPath": "display_name", "order": "ASCENDING" }
      ]
    }
  ]
}
EOF
gcloud firestore indexes composite create \
  --database="(default)" \
  --collection-group=vehicles \
  --field-config="field-path=status,order=ascending" \
  --field-config="field-path=display_name,order=ascending" \
  --project=$PROJECT_ID 2>/dev/null || echo "  ⚠️  Index may already exist — skipping"
echo "  ✓ Firestore index created"

# ── 6. Summary ────────────────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ✅ ==="
echo ""
echo "GCS Bucket    : https://storage.googleapis.com/$BUCKET_NAME/"
echo "Firestore     : https://console.firebase.google.com/project/$PROJECT_ID/firestore"
echo "Vertex AI     : https://console.cloud.google.com/vertex-ai?project=$PROJECT_ID"
echo ""
echo "Next step: Run the API locally with:"
echo "  cd c:/kotak"
echo "  pip install -r requirements.txt"
echo "  uvicorn api.main:app --reload --port 8080"
