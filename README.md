# 👁️ OculoAI • Clinical Dry Eye Diagnostic Suite

[![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B.svg?style=flat&logo=Streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?style=flat&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/indukurunaveenreddy/Dry-Eye-disease?quickstart=1)
[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=indukurunaveenreddy/Dry-Eye-disease&branch=main&mainModule=app.py)

An advanced clinical artificial intelligence platform for **Dry Eye Disease (DED)** screening and multi-source ocular evidence assessment. Combines infrared clinical meibography, real-time facial and ocular tracking, 12-item OSDI symptom burden neural modeling, and 6-stage blink dynamics.

---

## 🌟 Key Capabilities

- **👁️ Infrared Meibography Screening**: GLCM spatial texture extraction, gland dropout detection, and ML classification distinguishing pathology from healthy gland architecture.
- **📹 Live Camera Ocular Tracking**: Real-time Haar cascade facial landmark detection with automated ocular cropping and gaze classification.
- **📋 12-Item OSDI Neural Modeling**: Multi-layer Perceptron (MLP) classification predicting symptom severity (Normal, Mild, Moderate, Severe).
- **⚡ Blink Completeness & Gaze Dynamics**: Deep transfer learning (VGG-19) classifying 6 ocular orientations (Eye Closed, Eye Opened, Partially Open, Forward Look, Left Look, Right Look).
- **🤖 OculoAI Clinical Chatbot**: Interactive clinical consultation assistant with specialized dry eye guidance and 20-20-20 visual hygiene recommendations.
- **📊 Multimodal Clinical Datasets Hub**: Integrated explorer for **11,989 multimodal records** (999 infrared scans, 10,760 video frames, 230 survey patients) with filtering and export tools.

---


## ☁️ Run Directly on GitHub (No Local Installation)

You can launch and run this application immediately in the cloud without installing anything on your computer:

### Option 1: 1-Click Cloud Run via GitHub Codespaces ⚡ (Recommended)

Click the button below to launch an instant, interactive cloud container directly inside your browser:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/indukurunaveenreddy/Dry-Eye-disease?quickstart=1)

1. Click **Create new codespace** (or click the badge above).
2. GitHub Codespaces automatically provisions a Linux environment, installs all dependencies (`requirements.txt`), and launches the application.
3. When prompted, click **Open in Browser** (or visit the **Ports** tab and click port `8501`).
4. Log in instantly with persona credentials (e.g. `dr.chen` / `oculo2026`) or click **1-Click Instant Access**.

---

### Option 2: 1-Click Deployment on Streamlit Community Cloud 🌐

Host a permanent public web link connected directly to this GitHub repository:

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=indukurunaveenreddy/Dry-Eye-disease&branch=main&mainModule=app.py)

1. Click the **Deploy to Streamlit** badge above (or visit [share.streamlit.io](https://share.streamlit.io)).
2. Sign in with GitHub.
3. Confirm repository settings:
   - **Repository**: `indukurunaveenreddy/Dry-Eye-disease`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!** Your app will be live at `https://[your-app-name].streamlit.app`.

---

## 🚀 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/indukurunaveenreddy/Dry-Eye-disease.git
cd Dry-Eye-disease
```

### 2. Install Dependencies
```bash
cd "Dry Eye Disease/Source Code/Eye DIsease"
pip install -r requirements.txt
# Or install core dependencies:
pip install streamlit opencv-python scikit-learn scikit-image matplotlib pandas openpyxl streamlit-option-menu
```

### 3. Launch Application
```bash
streamlit run Login.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Repository Structure

```
Dry Eye Disease/
├── Source Code/
│   └── Eye DIsease/
│       ├── Login.py              # Glassmorphic Clinician Auth Portal
│       ├── Prediction.py         # Main Clinical Diagnostic Suite & Datasets Hub
│       ├── ui_theme.py           # Unified Glassmorphism Design System & CSS
│       ├── eye_model.py          # Real-time Haar cascade & ocular tracking model
│       ├── chatbot.py            # Clinical symptom advisor chatbot engine
│       ├── Dataset.xlsx          # 230-Patient OSDI-12 clinical survey cohort
│       ├── Dataset/              # 999 Non-contact infrared meibography scans
│       │   ├── Affected/         # Pathology scans (549 images)
│       │   └── Not/              # Normal / healthy scans (450 images)
│       └── Blink/                # 10,760 Gaze & blink video frames
│           ├── Closed/           # 109 frames
│           ├── Open/             # 109 frames
│           ├── Partial/          # 10 frames
│           ├── forward_look/     # 3,457 frames
│           ├── left_look/        # 3,498 frames
│           └── right_look/       # 3,577 frames
├── DOC/                          # Clinical documentation and study reports
├── PPT/                          # Presentation decks
└── Screenshots/                  # High-resolution application screenshots
```

---

## 🔒 Security & Privacy
- **100% On-Device Processing**: Local SQLite authentication with zero external cloud telemetry.
- **HIPAA / GDPR Design Readiness**: Localized model inference ensures sensitive patient imagery remains strictly on the clinician's machine.

---

## 👨‍⚕️ Author & Credits
Developed by **Naveen Reddy Indukuru** ([@indukurunaveenreddy](https://github.com/indukurunaveenreddy)).
