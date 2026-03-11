<div align="center">

```
 ██╗  ██╗███████╗ █████╗ ██╗  ████████╗██╗  ██╗
 ██║  ██║██╔════╝██╔══██╗██║  ╚══██╔══╝██║  ██║
 ███████║█████╗  ███████║██║     ██║   ███████║
 ██╔══██║██╔══╝  ██╔══██║██║     ██║   ██╔══██║
 ██║  ██║███████╗██║  ██║███████╗██║   ██║  ██║
 ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝   ╚═╝  ╚═╝
      G U A R D I A N  ·  AI Health System
```

### 🏥 AI-Powered Disease Prediction & Health Assistant

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-4285f4?style=flat-square&logo=google&logoColor=white)](https://aistudio.google.com)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-f7931e?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![SDG3](https://img.shields.io/badge/UN_SDG-Goal_3-4c9f38?style=flat-square)](https://sdgs.un.org/goals/goal3)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**[Features](#-features) · [Quick Start](#-quick-start) · [Dataset](#-dataset) · [Model](#-ml-model) · [Screenshots](#-screenshots) · [Structure](#-project-structure)**

</div>

---

## 🌍 About

**AI Health Guardian** is an intelligent health prediction platform built for **UN SDG Goal 3 — Good Health and Well-Being**. It combines a Random Forest ML model trained on 4,920 clinical samples with Google's Gemini AI to deliver:

- Instant disease prediction from 131 symptoms across 41 diseases
- Personalised health risk scoring with radar chart visualisation
- Real-time AI health chatbot (Dr. AI) powered by Gemini Flash
- Medicine information lookup, health dashboards, and PDF-ready reports

> *Making quality health information accessible to everyone.*

---

## ✨ Features

| Module | Description | Tech |
|--------|-------------|------|
| 🔬 **Symptom Checker** | Select from 131 symptoms → top-5 disease predictions with confidence scores | Random Forest |
| ❤️ **Risk Assessment** | Personal health risk score (0–100) with radar chart breakdown | Custom scoring |
| 🤖 **AI Chatbot** | Chat with Dr. AI for health queries, diet advice, symptom explanations | Gemini Flash |
| 💊 **Medicine Info** | Search any drug: uses, dosage, side effects, warnings | Gemini Flash |
| 📊 **Dashboard** | Disease distribution, BMI analytics, global burden choropleth map | Plotly |
| 📄 **Health Report** | Beautiful downloadable HTML health report with patient vitals | Streamlit |

---

## 🚀 Quick Start

### Prerequisites
- Python **3.9 or higher**
- A free **Gemini API key** → [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- The Kaggle dataset (instructions below)

---

### Step 1 — Clone & Enter Directory

```bash
# If using git
git clone https://github.com/YOUR_USERNAME/ai-health-guardian.git
cd ai-health-guardian

# Or just enter the project folder you already have
cd ai_health_guardian
```

---

### Step 2 — Create Virtual Environment

```bash
# Create
python -m venv venv

# Activate — Windows PowerShell
venv\Scripts\activate

# Activate — Mac / Linux
source venv/bin/activate

# You should see (venv) in your terminal prompt ✅
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ Takes 2–4 minutes. All packages install from PyPI.

---

### Step 4 — Add Your Gemini API Key

**Option A — `.env` file (recommended)**
```bash
# Copy the template
cp .env.example .env      # Mac/Linux
copy .env.example .env    # Windows

# Open .env in any editor and set:
GEMINI_API_KEY=AIzaSy...your_key_here
```

**Option B — Paste directly in the app sidebar**
No file editing needed — just paste your key in the sidebar when the app opens.

---

### Step 5 — Add the Dataset

**Manual download (easiest):**
1. Go to → [kaggle.com/datasets/itachi9604/disease-symptom-description-dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
2. Click **Download**
3. Extract the ZIP
4. Copy `dataset.csv` into the `data/` folder of this project

```
ai_health_guardian/
└── data/
    └── dataset.csv   ← place it here
```

**Or via Kaggle API:**
```bash
# Place your kaggle.json at ~/.kaggle/kaggle.json first
python download_dataset.py
```

---

### Step 6 — Train the ML Model

```bash
python train_model.py
```

**Expected output:**
```
✔  Loaded: data/dataset.csv  →  4,920 rows × 18 columns
✔  Unique symptoms found: 131
✔  Binary matrix: 4,920 rows × 131 symptom features
✔  Total classes: 41

  Random Forest       ██████████████████████████████  100.00%  ★ BEST
  Decision Tree       ██████████████████████████████  100.00%
  Naive Bayes         ██████████████████████████████  100.00%
  Gradient Boosting   ██████████████████████████████  100.00%

🏆 Best: Random Forest  →  100.00% accuracy
🎉 Training Complete!
```

> This creates `model/disease_model.pkl` — run only once.

---

### Step 7 — Launch the App 🎉

```bash
streamlit run app.py
```

Open your browser at **[http://localhost:8501](http://localhost:8501)**

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — itachi9604](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) |
| Rows | 4,920 |
| Columns | 18 (Disease + Symptom_1 … Symptom_17) |
| Diseases | 41 unique |
| Symptoms | 131 unique (after cleaning & one-hot encoding) |
| Format | Text symptom columns → converted to binary matrix |

**All 41 diseases covered:**

<details>
<summary>Click to expand full list</summary>

| # | Disease | # | Disease |
|---|---------|---|---------|
| 1 | Fungal infection | 22 | Hepatitis D |
| 2 | Allergy | 23 | Hepatitis E |
| 3 | GERD | 24 | Alcoholic hepatitis |
| 4 | Chronic cholestasis | 25 | Tuberculosis |
| 5 | Drug Reaction | 26 | Common Cold |
| 6 | Peptic ulcer disease | 27 | Pneumonia |
| 7 | AIDS | 28 | Dimorphic haemorrhoids |
| 8 | Diabetes | 29 | Heart attack |
| 9 | Gastroenteritis | 30 | Varicose veins |
| 10 | Bronchial Asthma | 31 | Hypothyroidism |
| 11 | Hypertension | 32 | Hyperthyroidism |
| 12 | Migraine | 33 | Hypoglycemia |
| 13 | Cervical spondylosis | 34 | Osteoarthritis |
| 14 | Paralysis (brain hemorrhage) | 35 | Arthritis |
| 15 | Jaundice | 36 | Vertigo / BPPV |
| 16 | Malaria | 37 | Acne |
| 17 | Chicken pox | 38 | Urinary tract infection |
| 18 | Dengue | 39 | Psoriasis |
| 19 | Typhoid | 40 | Impetigo |
| 20 | Hepatitis A | 41 | hepatitis A |
| 21 | Hepatitis B | — | — |

</details>

---

## 🤖 ML Model

### Architecture

```
Raw CSV  →  Text Cleaning  →  One-Hot Encoding (131 features)
         →  Train/Test Split 80/20
         →  4 Models Trained in Parallel
         →  Best Model Auto-Selected
         →  Saved as .pkl
```

### Model Comparison

| Model | Accuracy | CV (5-fold) | Training Time |
|-------|----------|-------------|---------------|
| ✅ **Random Forest (200 trees)** | **100.00%** | **100.00%** | ~0.4s |
| Decision Tree | 100.00% | 100.00% | ~0.1s |
| Naive Bayes | 100.00% | 100.00% | ~0.0s |
| Gradient Boosting | 100.00% | 100.00% | ~31s |

> **Why 100%?** The dataset has clearly distinct symptom sets per disease with minimal overlap, making it perfectly separable for tree-based models. This is expected and consistent with published results on this Kaggle dataset.

### Top Predictive Symptoms

```
muscle_pain          ████████████████████████████  0.01781
family_history       ████████████████████████░░░░  0.01589
fatigue              ███████████████████████░░░░░  0.01509
dark_urine           ███████████████████████░░░░░  0.01497
diarrhoea            ███████████████████████░░░░░  0.01479
yellowing_of_eyes    ███████████████████████░░░░░  0.01476
```

---

## 📁 Project Structure

```
ai_health_guardian/
│
├── 📱 app.py                    Main Streamlit application (UI + all pages)
├── 🧠 train_model.py            ML training pipeline with colored output
├── 🤖 gemini_helper.py          Gemini AI integration with fallback models
├── 📦 download_dataset.py       Kaggle dataset downloader
│
├── 📋 requirements.txt          Python dependencies
├── 🔑 .env.example              Environment variable template
├── 🚫 .gitignore                Git ignore rules (keys, models, data)
├── 📖 README.md                 This file
│
├── data/
│   ├── dataset.csv              Kaggle disease-symptom dataset ← you add this
│   └── disease_info.json        Disease descriptions, treatments, diets (auto-generated)
│
└── model/
    ├── disease_model.pkl        Trained Random Forest (auto-generated)
    ├── label_encoder.pkl        Label encoder (auto-generated)
    ├── symptom_columns.json     131 symptom feature names (auto-generated)
    └── model_info.json          Model performance metadata (auto-generated)
```

---

## 🛠️ Tech Stack

```
Frontend    →  Streamlit + Custom CSS (Syne font, biopunk dark theme)
ML Model    →  Scikit-learn (Random Forest, Decision Tree, Naive Bayes, Gradient Boosting)
AI / NLP    →  Google Gemini 2.0 Flash Lite (free tier) with automatic fallback
Charts      →  Plotly (bar, pie, radar, choropleth, histogram)
Data        →  Pandas + NumPy
Environment →  python-dotenv
```

---

## ⚙️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `python train_model.py` — KeyError: 'prognosis' | Your CSV uses 'Disease' as target — already handled automatically |
| `ValueError: invalid literal for int()` | Text-format dataset — already handled, the script converts it |
| Gemini 404 model not found | Update `FREE_MODELS` list in `gemini_helper.py` |
| Gemini 429 quota exceeded | App auto-retries and falls back to next model; wait 1 min |
| Model not found error in app | Run `python train_model.py` first |
| Port already in use | `streamlit run app.py --server.port 8502` |
| `venv` won't activate on Windows | Run PowerShell as Admin → `Set-ExecutionPolicy RemoteSigned` |

---

## 🌐 SDG 3 Impact

This project directly addresses **UN Sustainable Development Goal 3 — Good Health and Well-Being** by:

- 🔍 **Early Detection** — helping users identify possible diseases before seeing a doctor
- 📚 **Health Literacy** — providing accessible information about 41 diseases and 131 symptoms  
- 💊 **Medicine Access** — instant drug information for informed decision-making
- 📊 **Risk Awareness** — personalised health risk scoring encourages preventive action
- 🌍 **Universal Access** — free, open-source, runs locally with no subscription needed

---

## ⚠️ Disclaimer

This tool is built for **educational purposes** as a student project for the SDG 3 initiative.

- ❌ Not a substitute for professional medical advice
- ❌ Not for clinical diagnosis or treatment decisions  
- ✅ For learning, awareness, and informational purposes only
- ✅ Always consult a qualified healthcare professional

---

## 📜 License

MIT License — free to use, modify, and distribute with attribution.

---

<div align="center">

**Built with ❤️ for SDG 3 · Good Health and Well-Being**

*Random Forest + Gemini AI + Streamlit · 41 diseases · 131 symptoms · 100% accuracy*

</div>