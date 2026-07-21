# 🪟 Windows Setup Guide — 360° Vehicle Image Generator

This guide provides step-by-step instructions for setting up, authenticating, and running the **360° Vehicle Image Generator** on a **Windows** machine.

---

## 🧰 Prerequisites for Windows

Before starting, ensure you have the following installed on your Windows system:

| Tool | Recommended Version | Download Link / Command |
|---|---|---|
| **Python** | 3.11 or newer | [python.org](https://www.python.org/downloads/windows/) *(Check "Add Python to PATH" during installation!)* |
| **Git for Windows** | Latest | [git-scm.com](https://git-scm.com/download/win) |
| **Google Cloud SDK** | Latest | [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install#windows) |

---

## 🚀 Step-by-Step Setup on Windows

### 1. Clone the Repository

Open **PowerShell** or **Command Prompt (cmd)**:

```powershell
git clone https://github.com/ratnesh90859/img_generator.git
cd img_generator
```

---

### 2. Create and Activate Virtual Environment

**In PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*(Note: If PowerShell blocks script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*

**In Command Prompt (cmd):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

---

### 3. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

---

### 4. Authenticate with Google Cloud (GCP)

Run the following two commands to authenticate your gcloud CLI and Python Google Client Libraries (ADC):

```powershell
# 1. Authenticate gcloud CLI
gcloud auth login

# 2. Authenticate Application Default Credentials (ADC) for Python backend
gcloud auth application-default login

# 3. Set target GCP Project
gcloud config set project testing-ratnesh
```

---

### 5. Configure Environment Variables (`.env`)

Create or edit the `.env` file in the root `img_generator` directory:

```powershell
copy .env.example .env
```

Ensure your `.env` contains:

```env
GCP_PROJECT_ID=testing-ratnesh
GCP_REGION=asia-south1
GCS_BUCKET_NAME=kotak-vehicles-360
SERVICE_ACCOUNT_EMAIL=infra-agent-sa@testing-ratnesh.iam.gserviceaccount.com
FIRESTORE_COLLECTION=vehicles
IMAGEN_MODEL=gemini-2.5-flash-image
IMAGE_WIDTH=1024
IMAGE_HEIGHT=1024
TOTAL_FRAMES=36
DEFAULT_CAR_COLOR=Dashing Silver
```

---

### 6. Set Up GCP Infrastructure (One-Time Run)

Run the setup script using **Git Bash** (included with Git for Windows):

```bash
# Open Git Bash in project folder:
bash scripts/setup_gcp.sh
```

---

### 7. Run the Backend API Server (Terminal 1)

Open **Terminal 1** (PowerShell/cmd with virtual environment activated):

**In PowerShell:**
```powershell
$env:PYTHONPATH="."
.\venv\Scripts\uvicorn api.main:app --port 8080 --log-level warning
```

**In Command Prompt (cmd):**
```cmd
set PYTHONPATH=.
venv\Scripts\uvicorn api.main:app --port 8080 --log-level warning
```

Your API backend is now live at **http://localhost:8080** (API Docs: **http://localhost:8080/docs**).

---

### 8. Run the Frontend Web App (Terminal 2)

Open **Terminal 2** (PowerShell/cmd in the `img_generator` folder):

```powershell
python -m http.server 3000 --directory frontend
```

Your frontend app is now live at **http://localhost:3000**.

---

## 🎮 Using the Application

1. Open **[http://localhost:3000](http://localhost:3000)** in your browser.
2. Select a vehicle model (e.g. *Honda City*, *Hyundai Creta*, *Tata Harrier*).
3. *(Optional)* Click **⚙️ Advanced** to upload reference images or check **Force Regenerate**.
4. Click **⚡ Generate**.
5. Once complete, click the vehicle card to open the interactive **360° Drag-to-Rotate Viewer**.

---

## 🔧 Windows Troubleshooting

### 1. `Execution of scripts is disabled on this system` in PowerShell
Run this command in your PowerShell session before activating the virtual environment:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### 2. `AttributeError: you need a private key to sign credentials`
Ensure `gcloud auth application-default login` was executed and `SERVICE_ACCOUNT_EMAIL=infra-agent-sa@testing-ratnesh.iam.gserviceaccount.com` is present in your `.env` file.

### 3. `ModuleNotFoundError: No module named 'api'`
Make sure `PYTHONPATH` is set to the current directory before launching uvicorn:
- PowerShell: `$env:PYTHONPATH="."`
- CMD: `set PYTHONPATH=.`
