import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Health Guardian",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Master CSS — Biopunk / Medical Dark Aesthetic ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #060910; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1200px; }

/* ── Animated grid background ── */
.main::before {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(0,255,200,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,200,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060910 0%, #0a0f1a 100%) !important;
    border-right: 1px solid rgba(0,255,180,0.1) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1rem !important; }

/* ── Nav radio buttons ── */
div[role="radiogroup"] label {
    border-radius: 10px !important;
    padding: 0.5rem 1rem !important;
    margin: 2px 0 !important;
    transition: all 0.2s ease !important;
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
div[role="radiogroup"] label:hover { background: rgba(0,255,180,0.07) !important; color: #00ffb4 !important; }
div[role="radiogroup"] label[data-checked="true"] {
    background: rgba(0,255,180,0.12) !important;
    color: #00ffb4 !important;
    border-left: 3px solid #00ffb4 !important;
}

/* ── Headings ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #e2e8f0 !important; }

/* ── Cards ── */
.g-card {
    background: linear-gradient(135deg, #0d1117 0%, #0a0f1a 100%);
    border: 1px solid rgba(0,255,180,0.12);
    border-radius: 18px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.g-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00ffb4, transparent);
}

.hero-card {
    background: linear-gradient(135deg, #0a1628 0%, #060d1f 50%, #0a1628 100%);
    border: 1px solid rgba(0,255,180,0.18);
    border-radius: 24px;
    padding: 3rem 2.5rem;
    position: relative; overflow: hidden;
    margin-bottom: 2rem;
}
.hero-card::after {
    content: '🧬';
    position: absolute; right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 6rem; opacity: 0.08;
}

/* ── Metric boxes ── */
.metric-box {
    background: linear-gradient(135deg, #0d1a2e, #0a1220);
    border: 1px solid rgba(0,255,180,0.15);
    border-radius: 16px;
    padding: 1.4rem 1rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-box:hover { transform: translateY(-3px); border-color: rgba(0,255,180,0.4); }
.metric-val { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; color: #00ffb4; }
.metric-lbl { font-size: 0.8rem; color: #64748b; margin-top: 0.3rem; letter-spacing: 0.08em; text-transform: uppercase; }

/* ── Section headers ── */
.sec-head {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem; font-weight: 700;
    color: #00ffb4;
    border-bottom: 1px solid rgba(0,255,180,0.2);
    padding-bottom: 0.6rem;
    margin-bottom: 1.2rem;
    letter-spacing: 0.02em;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #0d1a2e 0%, #0a1220 100%);
    border: 1px solid rgba(0,150,255,0.2);
    border-radius: 16px;
    padding: 1.4rem;
    margin: 0.8rem 0;
}
.result-card.green { border-color: rgba(0,255,180,0.3); }
.result-card.amber { border-color: rgba(255,165,0,0.3); }
.result-card.red   { border-color: rgba(255,75,75,0.3); }

/* ── Tags / pills ── */
.pill {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    background: rgba(0,255,180,0.08);
    border: 1px solid rgba(0,255,180,0.25);
    border-radius: 20px;
    font-size: 0.78rem;
    color: #00ffb4;
    margin: 0.15rem;
}
.pill-blue {
    background: rgba(56,189,248,0.08);
    border-color: rgba(56,189,248,0.25);
    color: #38bdf8;
}

/* ── Chat bubbles ── */
.chat-user {
    background: linear-gradient(135deg, #0d2240, #0a1a35);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.6rem 0 0.6rem 15%;
    color: #cbd5e1;
    font-size: 0.92rem;
}
.chat-bot {
    background: linear-gradient(135deg, #0d1a17, #0a1512);
    border: 1px solid rgba(0,255,180,0.15);
    border-left: 3px solid #00ffb4;
    border-radius: 4px 16px 16px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.6rem 15% 0.6rem 0;
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* ── All buttons: bright visible green ── */
.stButton > button {
    background: linear-gradient(135deg, #00c896, #00a07a) !important;
    color: #050e0a !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.55rem 1.6rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    min-height: 42px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,200,150,0.4) !important;
    background: linear-gradient(135deg, #00e6aa, #00b88a) !important;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox select {
    background: #0d1117 !important;
    border: 1px solid rgba(0,255,180,0.15) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(0,255,180,0.5) !important;
    box-shadow: 0 0 0 2px rgba(0,255,180,0.1) !important;
}

/* ── Multiselect ── */
.stMultiSelect [data-baseweb="tag"] {
    background: rgba(0,255,180,0.12) !important;
    border: 1px solid rgba(0,255,180,0.3) !important;
    color: #00ffb4 !important;
}

/* ── Progress / slider ── */
.stSlider [data-baseweb="slider"] { color: #00ffb4 !important; }

/* ── Alerts ── */
.stSuccess { background: rgba(0,255,180,0.06) !important; border-left-color: #00ffb4 !important; }
.stWarning { background: rgba(255,165,0,0.06) !important; border-left-color: #ffa500 !important; }
.stInfo    { background: rgba(56,189,248,0.06) !important; border-left-color: #38bdf8 !important; }
.stError   { background: rgba(255,75,75,0.06)  !important; border-left-color: #ff4b4b !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid rgba(0,255,180,0.15) !important; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; font-family: 'DM Sans', sans-serif !important; }
.stTabs [aria-selected="true"] { color: #00ffb4 !important; border-bottom: 2px solid #00ffb4 !important; }

/* ── Divider ── */
hr { border-color: rgba(0,255,180,0.1) !important; }

p, li, label { color: #94a3b8 !important; }

/* Fix: push content below Streamlit top bar */
.block-container { padding-top: 2.8rem !important; }

/* ── Chip buttons: target by button key (qq0-qq3, pm0-pm5) ── */
/* Streamlit sets data-testid on the button's parent div */
[data-testid="stButton"]:has(button[kind="secondary"]) > button,
button[data-testid="qq0"], button[data-testid="qq1"],
button[data-testid="qq2"], button[data-testid="qq3"],
button[data-testid="pm0"], button[data-testid="pm1"],
button[data-testid="pm2"], button[data-testid="pm3"],
button[data-testid="pm4"], button[data-testid="pm5"] {
    background: rgba(0,255,180,0.04) !important;
    color: #64748b !important;
    border: 1px solid rgba(0,255,180,0.18) !important;
    border-radius: 20px !important;
    padding: 5px 14px !important;
    font-size: 0.76rem !important;
    font-weight: 400 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0 !important;
    box-shadow: none !important;
    transform: none !important;
    min-height: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Google Drive file IDs — paste your IDs here after uploading ──────────────
GDRIVE_IDS = {
    "disease_model.pkl":    "11gb8ofi_3rOM64Z3q8omzLUVQ9S2ZRI2",
    "label_encoder.pkl":    "1EPQoAOM_oyJfSi3YaYO8JfaY-lKwVHoR",
    "symptom_columns.json": "10o7xfz0lwjiVR7FJe7--5X7H3mAD2O51",
    "disease_info.json":    "1Z8S3mDLEIJfyTOCiV-dO7PFzpnXQZCHd",
}

def download_from_gdrive():
    """Download model files from Google Drive if not present locally."""
    try:
        import gdown
    except ImportError:
        st.error("gdown not installed. Add 'gdown' to requirements.txt")
        return False

    os.makedirs("model", exist_ok=True)
    os.makedirs("data",  exist_ok=True)

    files = {
        "model/disease_model.pkl":    GDRIVE_IDS["disease_model.pkl"],
        "model/label_encoder.pkl":    GDRIVE_IDS["label_encoder.pkl"],
        "model/symptom_columns.json": GDRIVE_IDS["symptom_columns.json"],
        "data/disease_info.json":     GDRIVE_IDS["disease_info.json"],
    }

    for path, file_id in files.items():
        if not os.path.exists(path):
            if "PASTE_" in file_id:
                st.error(f"⚠️ Google Drive ID not set for {path}. Please update GDRIVE_IDS in app.py")
                return False
            try:
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, path, quiet=True, fuzzy=True)
            except Exception as e:
                st.error(f"Failed to download {path}: {e}")
                return False
    return True


# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("model/disease_model.pkl","rb"))
        le    = pickle.load(open("model/label_encoder.pkl","rb"))
        syms  = json.load(open("model/symptom_columns.json"))
        return model, le, syms
    except:
        return None, None, None

@st.cache_data
def load_disease_info():
    try:
        return json.load(open("data/disease_info.json"))
    except:
        return {}

model, le, symptom_cols = load_model()
disease_info = load_disease_info()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#00ffb4'>
            ✚ AI Health Guardian
        </div>
        <div style='font-size:0.72rem;color:#334155;letter-spacing:0.1em;margin-top:2px'>
            SDG 3 · GOOD HEALTH & WELL-BEING
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='margin:0.5rem 0'>", unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Home",
        "🔬  Symptom Checker",
        "❤️  Risk Assessment",
        "🤖  AI Health Chatbot",
        "💊  Medicine Info",
        "📊  Health Dashboard",
        "📄  Health Report",
        "ℹ️  About"
    ], label_visibility="collapsed")

    st.markdown("<hr style='margin:0.8rem 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Syne,sans-serif;font-size:0.78rem;color:#334155;font-weight:600;letter-spacing:0.08em;margin-bottom:0.5rem'>🔑 GEMINI API KEY</div>", unsafe_allow_html=True)

    from gemini_helper import get_api_key
    _cur = get_api_key()
    if _cur:
        st.success("✅ API Key active")
        if st.checkbox("Replace key", key="chk_replace"):
            _nk = st.text_input("New key", type="password", key="new_key_inp")
            if st.button("Save", key="btn_save_key") and _nk.strip():
                os.environ["GEMINI_API_KEY"] = _nk.strip()
                lines = [l for l in (open(".env").readlines() if os.path.exists(".env") else []) if not l.startswith("GEMINI_API_KEY")]
                lines.append(f"GEMINI_API_KEY={_nk.strip()}\n")
                open(".env","w").writelines(lines)
                st.rerun()
    else:
        st.warning("No API key found")
        _tk = st.text_input("Paste Gemini key", type="password", placeholder="AIza...", key="tk_inp")
        if st.button("💾 Save & Apply", key="btn_apply") and _tk.strip():
            os.environ["GEMINI_API_KEY"] = _tk.strip()
            lines = [l for l in (open(".env").readlines() if os.path.exists(".env") else []) if not l.startswith("GEMINI_API_KEY")]
            lines.append(f"GEMINI_API_KEY={_tk.strip()}\n")
            open(".env","w").writelines(lines)
            st.rerun()
        st.markdown("<small style='color:#334155'>🔗 <a href='https://aistudio.google.com/app/apikey' target='_blank' style='color:#00ffb4'>Get free key →</a></small>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:1rem 0 0.5rem'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;color:#1e293b;text-align:center'>Random Forest · 100% accuracy<br>41 diseases · 131 symptoms</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div class='hero-card'>
        <div style='font-family:Syne,sans-serif;font-size:0.75rem;color:#00ffb4;letter-spacing:0.15em;font-weight:600;margin-bottom:0.5rem'>
            SDG 3 · GOOD HEALTH AND WELL-BEING
        </div>
        <h1 style='font-size:2.8rem;margin:0;background:linear-gradient(135deg,#00ffb4,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            AI Health Guardian
        </h1>
        <p style='color:#64748b;font-size:1.05rem;margin:0.8rem 0 0;max-width:520px'>
            Intelligent disease prediction, real-time AI health advice, and
            personalized wellness insights — powered by Machine Learning and Gemini AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col, v, l in zip([c1,c2,c3,c4],["41","131","100%","24/7"],["Diseases","Symptoms","ML Accuracy","Available"]):
        with col:
            st.markdown(f"<div class='metric-box'><div class='metric-val'>{v}</div><div class='metric-lbl'>{l}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec-head'>🚀 Features</div>", unsafe_allow_html=True)

    features = [
        ("🔬","Symptom Checker","Select symptoms → instant disease prediction with confidence scores and top-5 results"),
        ("❤️","Risk Assessment","Radar-chart health risk score based on your lifestyle, vitals, and family history"),
        ("🤖","AI Chatbot","Chat with Gemini-powered Dr. AI for any health query, 24/7"),
        ("💊","Medicine Info","Search any drug for uses, dosage, side effects and warnings"),
        ("📊","Dashboard","Interactive visualizations: disease trends, BMI distribution, global burden map"),
        ("📄","Health Report","Generate a beautiful downloadable health report with all your details"),
    ]
    r1 = st.columns(3)
    r2 = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        col = (r1 if i < 3 else r2)[i % 3]
        with col:
            st.markdown(f"""
            <div class='g-card' style='height:140px'>
                <div style='font-size:1.5rem;margin-bottom:0.4rem'>{icon}</div>
                <div style='font-family:Syne,sans-serif;font-weight:700;color:#e2e8f0;font-size:0.95rem'>{title}</div>
                <div style='font-size:0.8rem;color:#475569;margin-top:0.3rem;line-height:1.4'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec-head'>🧬 How It Works</div>", unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    for col, n, t, d in zip([s1,s2,s3,s4],["01","02","03","04"],
        ["Select Symptoms","ML Prediction","AI Analysis","Act"],
        ["Choose from 131 clinically-mapped symptoms","Random Forest model identifies disease from 41 classes",
         "Gemini AI provides context, diet & precautions","Download report or chat with Dr. AI"]):
        with col:
            st.markdown(f"""
            <div style='text-align:center;padding:1rem'>
                <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                    color:rgba(0,255,180,0.15);line-height:1'>{n}</div>
                <div style='font-family:Syne,sans-serif;font-weight:700;color:#e2e8f0;
                    font-size:0.88rem;margin:0.4rem 0'>{t}</div>
                <div style='font-size:0.78rem;color:#475569;line-height:1.4'>{d}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SYMPTOM CHECKER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Symptom Checker":
    st.markdown("<div class='sec-head'>🔬 AI Symptom Checker</div>", unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model not found. Run `python train_model.py` first.")
    else:
        col_l, col_r = st.columns([3, 1])
        with col_l:
            selected = st.multiselect("Select your symptoms", options=sorted(symptom_cols),
                placeholder="Type to search symptoms...")
        with col_r:
            st.markdown("<div style='padding-top:1.8rem'>", unsafe_allow_html=True)
            predict_btn = st.button("🔮 Predict", use_container_width=True)
        
        # Quick-pick common symptoms
        st.markdown("<div style='margin:0.5rem 0 0.3rem;font-size:0.8rem;color:#475569;font-weight:500'>⚡ Quick select common symptoms:</div>", unsafe_allow_html=True)
        common = ["fever","headache","cough","fatigue","nausea","vomiting",
                  "chest_pain","breathlessness","itching","joint_pain",
                  "diarrhoea","skin_rash","weight_loss","back_pain","high_fever"]
        qcols = st.columns(5)
        quick = []
        for i, s in enumerate(common):
            if qcols[i % 5].checkbox(s.replace("_"," "), key=f"qp_{s}"):
                quick.append(s)

        all_syms = list(set(selected + quick))
        if all_syms:
            st.markdown("**Active:** " + "".join(f"<span class='pill'>{s.replace('_',' ')}</span>" for s in all_syms), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if predict_btn or all_syms:
            if not all_syms:
                st.warning("Please select at least one symptom.")
            else:
                with st.spinner("Analysing symptoms with Random Forest..."):
                    inp = np.zeros(len(symptom_cols))
                    for s in all_syms:
                        if s in symptom_cols:
                            inp[symptom_cols.index(s)] = 1
                    proba   = model.predict_proba([inp])[0]
                    top5    = np.argsort(proba)[::-1][:5]
                    disease = le.inverse_transform([top5[0]])[0]
                    conf    = proba[top5[0]] * 100
                    rcolor  = "#00ffb4" if conf > 70 else "#ffa500" if conf > 40 else "#ff4b4b"
                    rclass  = "green"   if conf > 70 else "amber"   if conf > 40 else "red"

                rc1, rc2 = st.columns([1,2])
                with rc1:
                    st.markdown(f"""
                    <div class='result-card {rclass}'>
                        <div style='font-size:0.72rem;color:#475569;letter-spacing:0.1em;font-weight:600'>PREDICTED DISEASE</div>
                        <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;
                            color:{rcolor};margin:0.5rem 0;line-height:1.2'>{disease}</div>
                        <div style='font-size:0.85rem;color:#64748b'>Confidence</div>
                        <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:{rcolor}'>{conf:.1f}%</div>
                        <div style='background:rgba(0,0,0,0.3);border-radius:8px;height:6px;margin-top:0.5rem;overflow:hidden'>
                            <div style='height:100%;width:{conf}%;background:{rcolor};border-radius:8px;transition:width 0.8s ease'></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                with rc2:
                    diseases_top5 = [le.inverse_transform([i])[0] for i in top5]
                    scores_top5   = [proba[i]*100 for i in top5]
                    fig = go.Figure(go.Bar(
                        x=scores_top5, y=diseases_top5, orientation='h',
                        marker=dict(color=scores_top5, colorscale=[[0,"#0d2240"],[0.5,"#0077b6"],[1,"#00ffb4"]],
                                    line=dict(width=0)),
                        text=[f"{s:.1f}%" for s in scores_top5], textposition='auto',
                        textfont=dict(color='white', size=11)
                    ))
                    fig.update_layout(
                        title=dict(text="Top 5 Differential Diagnoses", font=dict(family="Syne", color="#94a3b8", size=13)),
                        template="plotly_dark", height=250, margin=dict(l=10,r=10,t=40,b=10),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.8)',
                        yaxis=dict(tickfont=dict(size=11, color="#94a3b8")),
                        xaxis=dict(tickfont=dict(size=10, color="#475569"), gridcolor='rgba(255,255,255,0.04)')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                info = disease_info.get(disease, {})
                if info:
                    i1,i2 = st.columns(2)
                    with i1:
                        st.markdown(f"""<div class='g-card'>
                            <div style='font-family:Syne,sans-serif;font-weight:700;color:#00ffb4;margin-bottom:0.5rem'>📋 About</div>
                            <p style='font-size:0.85rem;line-height:1.6;color:#94a3b8'>{info.get('description','N/A')}</p>
                            <div style='font-family:Syne,sans-serif;font-weight:700;color:#38bdf8;margin:0.8rem 0 0.4rem'>💊 Treatment</div>
                            <p style='font-size:0.82rem;color:#64748b'>{info.get('treatment','N/A')}</p>
                        </div>""", unsafe_allow_html=True)
                    with i2:
                        precs = info.get("precautions", [])
                        prec_html = "".join(f"<div style='padding:4px 0;font-size:0.82rem;color:#94a3b8'>✦ {p}</div>" for p in precs)
                        st.markdown(f"""<div class='g-card'>
                            <div style='font-family:Syne,sans-serif;font-weight:700;color:#ffa500;margin-bottom:0.5rem'>⚠️ Precautions</div>
                            {prec_html}
                            <div style='font-family:Syne,sans-serif;font-weight:700;color:#a78bfa;margin:0.8rem 0 0.4rem'>🥗 Diet</div>
                            <p style='font-size:0.82rem;color:#64748b'>{info.get('diet','N/A')}</p>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.warning("⚠️ AI prediction only — always consult a qualified doctor for proper diagnosis and treatment.")


# ═══════════════════════════════════════════════════════════════════════════════
# RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "❤️  Risk Assessment":
    st.markdown("<div class='sec-head'>❤️ Personal Health Risk Assessment</div>", unsafe_allow_html=True)
    with st.form("risk_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            age    = st.number_input("Age", 1, 120, 25)
            weight = st.number_input("Weight (kg)", 10, 300, 70)
            height = st.number_input("Height (cm)", 50, 250, 170)
        with c2:
            smoke    = st.selectbox("Smoking", ["Never","Occasionally","Daily"])
            alcohol  = st.selectbox("Alcohol", ["Never","Occasionally","Regularly"])
            exercise = st.selectbox("Exercise frequency", ["Daily","3-4x/week","Rarely","Never"])
        with c3:
            sleep    = st.slider("Sleep (hrs/night)", 2, 12, 7)
            stress   = st.slider("Stress level (1-10)", 1, 10, 5)
            bp       = st.selectbox("Blood Pressure", ["Normal","High","Low","Unknown"])
            diabetes = st.checkbox("Family history: Diabetes")
            heart    = st.checkbox("Family history: Heart Disease")
        submitted = st.form_submit_button("⚡ Calculate My Risk Score", use_container_width=True)

    if submitted:
        bmi = weight / ((height/100)**2)
        score = 0
        score += (10 if age>60 else 5 if age>40 else 2)
        score += (15 if bmi>30 else 8 if bmi>25 else 3 if bmi<18.5 else 0)
        score += {"Never":0,"Occasionally":10,"Daily":20}[smoke]
        score += {"Never":0,"Occasionally":5,"Regularly":15}[alcohol]
        score += {"Daily":0,"3-4x/week":5,"Rarely":15,"Never":20}[exercise]
        score += (10 if sleep<5 else 5 if sleep<6 else 0)
        score += stress * 2
        score += {"Normal":0,"High":15,"Low":5,"Unknown":3}[bp]
        if diabetes: score += 10
        if heart:    score += 10
        score = min(score, 100)

        lvl    = ("🟢 Low Risk" if score<35 else "🟡 Medium Risk" if score<60 else "🔴 High Risk")
        rcolor = ("#00ffb4" if score<35 else "#ffa500" if score<60 else "#ff4b4b")
        rclass = ("green" if score<35 else "amber" if score<60 else "red")

        sc1, sc2 = st.columns([1, 2])
        with sc1:
            st.markdown(f"""
            <div class='result-card {rclass}' style='text-align:center;padding:2rem'>
                <div style='font-size:0.72rem;color:#475569;letter-spacing:0.12em;font-weight:600'>HEALTH RISK SCORE</div>
                <div style='font-family:Syne,sans-serif;font-size:4.5rem;font-weight:800;color:{rcolor};line-height:1;margin:0.5rem 0'>{score}</div>
                <div style='font-size:0.8rem;color:#64748b'>out of 100</div>
                <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:{rcolor};margin-top:0.7rem'>{lvl}</div>
                <div style='margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.05)'>
                    <div style='font-size:0.78rem;color:#475569'>BMI</div>
                    <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:700;color:#38bdf8'>{bmi:.1f}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        with sc2:
            cats = ["Age Factor","BMI","Smoking","Alcohol","Exercise","Sleep","Stress"]
            vals = [
                min(age/80*100,100),
                max(0,(bmi-18)/22*100),
                {"Never":0,"Occasionally":50,"Daily":100}[smoke],
                {"Never":0,"Occasionally":50,"Regularly":100}[alcohol],
                {"Daily":0,"3-4x/week":25,"Rarely":75,"Never":100}[exercise],
                max(0,(7-sleep)/5*100),
                stress*10
            ]
            fig = go.Figure(go.Scatterpolar(
                r=vals+[vals[0]], theta=cats+[cats[0]],
                fill='toself',
                fillcolor='rgba(0,255,180,0.07)',
                line=dict(color='#00ffb4', width=2),
                marker=dict(color='#00ffb4', size=6)
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(range=[0,100], tickfont=dict(size=9,color='#475569'),
                                   gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.05)'),
                    angularaxis=dict(tickfont=dict(size=10,color='#94a3b8'), gridcolor='rgba(255,255,255,0.05)')
                ),
                paper_bgcolor='rgba(0,0,0,0)', height=280,
                margin=dict(l=40,r=40,t=20,b=20),
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='sec-head' style='margin-top:1rem'>💡 Personalised Recommendations</div>", unsafe_allow_html=True)
        recs = []
        if bmi>25:   recs.append(("🥗","Weight","Work towards a healthy BMI through balanced diet and regular exercise."))
        if smoke!="Never": recs.append(("🚭","Smoking","Quitting smoking dramatically reduces your risk of heart disease and cancer."))
        if sleep<7:  recs.append(("😴","Sleep","Aim for 7-8 hours of quality sleep each night for optimal recovery."))
        if stress>6: recs.append(("🧘","Stress","Practice daily stress management: meditation, yoga, or deep breathing."))
        if exercise in ["Rarely","Never"]: recs.append(("🏃","Exercise","Add 30 minutes of moderate exercise at least 5 days per week."))
        if bp=="High": recs.append(("💊","Blood Pressure","Monitor BP regularly and follow your doctor's medication plan."))
        if not recs: recs.append(("✅","Excellent","Your lifestyle looks great! Keep maintaining these healthy habits."))
        rec_cols = st.columns(min(len(recs), 3))
        for i, (icon, title, desc) in enumerate(recs):
            with rec_cols[i % 3]:
                st.markdown(f"""<div class='g-card'>
                    <div style='font-size:1.4rem'>{icon}</div>
                    <div style='font-family:Syne,sans-serif;font-weight:700;color:#e2e8f0;font-size:0.88rem;margin:0.3rem 0'>{title}</div>
                    <div style='font-size:0.8rem;color:#64748b;line-height:1.4'>{desc}</div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# AI CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  AI Health Chatbot":
    st.markdown("<div class='sec-head'>🤖 Dr. AI — Health Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.82rem;color:#475569;margin-bottom:1rem'>Powered by Gemini · Ask anything about health, symptoms, diet, or medications</div>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history
    chat_box = st.container()
    with chat_box:
        if not st.session_state.messages:
            st.markdown("""<div class='chat-bot'>
                👋 Hi! I'm Dr. AI, your personal health assistant.<br>
                Ask me about symptoms, diseases, medications, diet tips, or general health advice.<br>
                <span style='font-size:0.78rem;color:#334155'>Remember: I provide information only — always consult a real doctor for diagnosis.</span>
            </div>""", unsafe_allow_html=True)
        for msg in st.session_state.messages:
            css = "chat-user" if msg["role"]=="user" else "chat-bot"
            icon = "🧑" if msg["role"]=="user" else "🤖"
            st.markdown(f"<div class='{css}'>{icon} {msg['content']}</div>", unsafe_allow_html=True)

    # Quick question chips — small compact style
    st.markdown("<div style='margin:0.8rem 0 0.3rem;font-size:0.75rem;color:#475569;letter-spacing:0.06em'>QUICK QUESTIONS</div>", unsafe_allow_html=True)
    qq = ["Symptoms of diabetes?","How to boost immunity?","Foods good for heart?","Daily water intake?"]
    q_clicked = None
    qcols = st.columns(len(qq))
    for i, q in enumerate(qq):
        with qcols[i]:
            if st.button(q, key=f"qq{i}", use_container_width=True):
                q_clicked = q


    ci, cb = st.columns([6,1])
    with ci:
        user_in = st.text_input("", placeholder="Ask Dr. AI anything about your health...",
                                key="chat_inp", label_visibility="collapsed")
    with cb:
        send = st.button("Send ➤", use_container_width=True)

    final_input = q_clicked or (user_in if send and user_in else None)

    if final_input:
        st.session_state.messages.append({"role":"user","content":final_input})
        with st.spinner("Dr. AI is thinking..."):
            from gemini_helper import chat_with_gemini
            reply = chat_with_gemini(final_input, st.session_state.messages[:-1])
        st.session_state.messages.append({"role":"assistant","content":reply})
        st.rerun()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MEDICINE INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💊  Medicine Info":
    st.markdown("<div class='sec-head'>💊 Medicine Information Finder</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.82rem;color:#475569;margin-bottom:1rem'>Search any medicine for uses, dosage, side effects, and warnings — powered by Gemini AI</div>", unsafe_allow_html=True)

    mc1, mc2 = st.columns([4,1])
    with mc1:
        med = st.text_input("", placeholder="Enter medicine name e.g. Paracetamol, Metformin, Aspirin...",
                            label_visibility="collapsed")
    with mc2:
        search = st.button("🔍 Search", use_container_width=True)

    # Popular medicines quick buttons
    st.markdown("<div style='font-size:0.78rem;color:#475569;margin:0.5rem 0 0.3rem'>Popular searches:</div>", unsafe_allow_html=True)
    pops = ["Paracetamol","Metformin","Aspirin","Amoxicillin","Omeprazole","Cetirizine"]
    pop_cols = st.columns(6)
    pop_clicked = None
    for i, m in enumerate(pops):
        if pop_cols[i].button(m, key=f"pm{i}"):
            pop_clicked = m

    final_med = pop_clicked or (med if search and med else None)
    if final_med:
        with st.spinner(f"Fetching info for {final_med}..."):
            from gemini_helper import get_medicine_info
            info_text = get_medicine_info(final_med)
        st.markdown(f"""<div class='g-card'>
            <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;
                color:#00ffb4;margin-bottom:1rem'>💊 {final_med}</div>
            <div style='font-size:0.88rem;color:#94a3b8;line-height:1.7'>{info_text}</div>
        </div>""", unsafe_allow_html=True)
        st.warning("⚠️ Always consult your doctor or pharmacist before taking any medication.")


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Health Dashboard":
    st.markdown("<div class='sec-head'>📊 Health Insights Dashboard</div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🦠 Disease Overview", "📈 BMI Analytics", "🌍 Global Burden"])

    with tab1:
        # Real dataset: 4920 rows / 41 diseases = 120 each, but show all 41 sorted
        all_diseases = [
            "Fungal infection","Allergy","GERD","Chronic cholestasis","Drug Reaction",
            "Peptic ulcer diseae","AIDS","Diabetes","Gastroenteritis","Bronchial Asthma",
            "Hypertension","Migraine","Cervical spondylosis","Paralysis (brain hemorrhage)",
            "Jaundice","Malaria","Chicken pox","Dengue","Typhoid","hepatitis A",
            "Hepatitis B","Hepatitis C","Hepatitis D","Hepatitis E","Alcoholic hepatitis",
            "Tuberculosis","Common Cold","Pneumonia","Dimorphic hemmorhoids(piles)",
            "Heart attack","Varicose veins","Hypothyroidism","Hyperthyroidism","Hypoglycemia",
            "Osteoarthristis","Arthritis","(vertigo) Paroymsal  Positional Vertigo",
            "Acne","Urinary tract infection","Psoriasis","Impetigo"
        ]
        counts = [120] * 41  # perfectly balanced dataset

        # Color gradient by index
        colors = [f"hsl({int(160 + i*3)}, 70%, {int(35 + i*0.5)}%)" for i in range(41)]

        dc1, dc2 = st.columns([2, 1])
        with dc1:
            fig1 = go.Figure(go.Bar(
                x=counts, y=all_diseases, orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                text=[str(c) for c in counts],
                textposition="outside",
                textfont=dict(color="#475569", size=10),
                hovertemplate="<b>%{y}</b><br>Samples: %{x}<extra></extra>"
            ))
            fig1.update_layout(
                title=dict(text="All 41 Diseases — Sample Count (Balanced Dataset)",
                           font=dict(family="Syne", color="#94a3b8", size=13)),
                template="plotly_dark", height=900,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.6)",
                margin=dict(l=5, r=60, t=45, b=10),
                xaxis=dict(range=[0,145], gridcolor="rgba(255,255,255,0.04)",
                           tickfont=dict(color="#475569", size=10), title=""),
                yaxis=dict(tickfont=dict(size=10, color="#cbd5e1"), title="",
                           autorange="reversed")
            )
            st.plotly_chart(fig1, use_container_width=True)

        with dc2:
            # Category donut
            cats_data = {
                "Infectious": 14, "Digestive": 7, "Skin": 5,
                "Metabolic": 5, "Cardiovascular": 4, "Neurological": 3, "Other": 3
            }
            fig_pie = go.Figure(go.Pie(
                labels=list(cats_data.keys()),
                values=list(cats_data.values()),
                hole=0.6,
                marker=dict(
                    colors=["#00ffb4","#38bdf8","#f472b6","#a78bfa","#fb923c","#34d399","#fbbf24"],
                    line=dict(color="#060910", width=3)
                ),
                textinfo="label+percent",
                textfont=dict(size=11, color="white"),
                hovertemplate="<b>%{label}</b><br>%{value} diseases (%{percent})<extra></extra>"
            ))
            fig_pie.update_layout(
                title=dict(text="By Category", font=dict(family="Syne", color="#94a3b8", size=13)),
                template="plotly_dark", height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=5, r=5, t=40, b=5),
                legend=dict(font=dict(color="#64748b", size=10), orientation="v")
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Symptom stats
            st.markdown("""
            <div class='g-card' style='padding:1.2rem;margin-top:0'>
                <div style='font-family:Syne,sans-serif;font-weight:700;color:#00ffb4;font-size:0.85rem;margin-bottom:0.8rem'>📊 Dataset Stats</div>
                <div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04)'>
                    <span style='font-size:0.78rem;color:#475569'>Total samples</span>
                    <span style='font-size:0.82rem;font-weight:600;color:#e2e8f0'>4,920</span>
                </div>
                <div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04)'>
                    <span style='font-size:0.78rem;color:#475569'>Diseases</span>
                    <span style='font-size:0.82rem;font-weight:600;color:#e2e8f0'>41</span>
                </div>
                <div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04)'>
                    <span style='font-size:0.78rem;color:#475569'>Unique symptoms</span>
                    <span style='font-size:0.82rem;font-weight:600;color:#e2e8f0'>131</span>
                </div>
                <div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04)'>
                    <span style='font-size:0.78rem;color:#475569'>Samples/disease</span>
                    <span style='font-size:0.82rem;font-weight:600;color:#00ffb4'>120 (balanced)</span>
                </div>
                <div style='display:flex;justify-content:space-between;padding:6px 0'>
                    <span style='font-size:0.78rem;color:#475569'>ML Accuracy</span>
                    <span style='font-size:0.82rem;font-weight:700;color:#00ffb4'>100%</span>
                </div>
            </div>""", unsafe_allow_html=True)

            # Top 5 symptoms
            top_syms = ["muscle_pain","family_history","fatigue","dark_urine","diarrhoea"]
            top_vals = [1.78, 1.59, 1.51, 1.50, 1.48]
            fig_sym = go.Figure(go.Bar(
                x=top_vals, y=[s.replace("_"," ").title() for s in top_syms],
                orientation="h",
                marker=dict(color=["#00ffb4","#22d3ee","#38bdf8","#818cf8","#a78bfa"],
                            line=dict(width=0)),
                text=[f"{v}%" for v in top_vals],
                textposition="outside",
                textfont=dict(color="#475569", size=10)
            ))
            fig_sym.update_layout(
                title=dict(text="Top 5 Predictive Symptoms", font=dict(family="Syne",color="#94a3b8",size=12)),
                template="plotly_dark", height=220,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=5,r=55,t=35,b=5),
                xaxis=dict(range=[0,2.2], showgrid=False, tickfont=dict(color="#475569",size=9)),
                yaxis=dict(tickfont=dict(size=10,color="#94a3b8"))
            )
            st.plotly_chart(fig_sym, use_container_width=True)

    with tab2:
        np.random.seed(42)
        bmi_data = np.random.normal(24.5, 4.5, 500)
        fig2 = go.Figure(go.Histogram(
            x=bmi_data, nbinsx=35,
            marker=dict(color=bmi_data, colorscale='teal', line=dict(width=0)),
            opacity=0.85
        ))
        fig2.add_vline(x=18.5, line_dash="dash", line_color="#ff4b4b", annotation_text="Underweight",
                       annotation_font_color="#ff4b4b")
        fig2.add_vline(x=25,   line_dash="dash", line_color="#ffa500", annotation_text="Overweight",
                       annotation_font_color="#ffa500")
        fig2.add_vline(x=30,   line_dash="dash", line_color="#ff0000", annotation_text="Obese",
                       annotation_font_color="#ff0000")
        fig2.update_layout(
            title=dict(text="BMI Distribution with Clinical Thresholds", font=dict(family="Syne",color="#94a3b8",size=14)),
            template="plotly_dark", height=350,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.8)',
            xaxis_title="BMI", yaxis_title="Count",
            margin=dict(l=10,r=10,t=50,b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

        bl, bn, bo, bob = st.columns(4)
        for col,lbl,val,clr in zip([bl,bn,bo,bob],
            ["Underweight","Normal","Overweight","Obese"],["8%","52%","28%","12%"],
            ["#38bdf8","#00ffb4","#ffa500","#ff4b4b"]):
            with col:
                st.markdown(f"<div class='metric-box'><div class='metric-val' style='color:{clr}'>{val}</div><div class='metric-lbl'>{lbl}</div></div>", unsafe_allow_html=True)

    with tab3:
        countries = ["IND","USA","BRA","NGA","CHN","DEU","GBR","RUS","IDN","PAK"]
        burden    = [85,60,70,90,55,35,40,65,80,88]
        fig4 = go.Figure(go.Choropleth(
            locations=countries, z=burden,
            colorscale=[[0,'#0d1a2e'],[0.5,'#0077b6'],[1,'#00ffb4']],
            marker_line_color='#060910', marker_line_width=0.5,
            colorbar=dict(tickfont=dict(color='#94a3b8'), title=dict(text="Index",font=dict(color='#94a3b8')))
        ))
        fig4.update_layout(
            title=dict(text="Global Disease Burden Index (Illustrative)", font=dict(family="Syne",color="#94a3b8",size=14)),
            geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False,
                     lakecolor='#060910', landcolor='#0d1117', showcoastlines=True,
                     coastlinecolor='rgba(0,255,180,0.15)'),
            paper_bgcolor='rgba(0,0,0,0)', height=400,
            margin=dict(l=0,r=0,t=50,b=0), template="plotly_dark"
        )
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH REPORT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📄  Health Report":
    st.markdown("<div class='sec-head'>📄 Generate Health Report</div>", unsafe_allow_html=True)
    with st.form("report_form"):
        c1,c2 = st.columns(2)
        with c1:
            name   = st.text_input("Full Name", placeholder="Your name")
            age    = st.number_input("Age", 1, 120, 25)
            gender = st.selectbox("Gender", ["Male","Female","Other"])
            blood  = st.selectbox("Blood Group", ["A+","A-","B+","B-","O+","O-","AB+","AB-","Unknown"])
        with c2:
            weight   = st.number_input("Weight (kg)", 10, 300, 70)
            height   = st.number_input("Height (cm)", 50, 250, 170)
            symptoms = st.text_area("Current Symptoms", placeholder="fever, headache, cough...")
            diseases = st.text_area("Known Conditions / Medications", placeholder="Diabetes, Paracetamol...")
        generate = st.form_submit_button("📄 Generate Report", use_container_width=True)

    if generate and name:
        bmi     = weight/((height/100)**2)
        bmi_cat = ("Underweight" if bmi<18.5 else "Normal" if bmi<25 else "Overweight" if bmi<30 else "Obese")
        bmi_c   = ("#00ffb4" if bmi_cat=="Normal" else "#ffa500" if bmi_cat in ("Overweight","Underweight") else "#ff4b4b")
        now_str = datetime.now().strftime("%B %d, %Y  ·  %I:%M %p")

        st.markdown(f"""
        <div class='hero-card' style='padding:2rem'>
            <div style='font-size:0.72rem;color:#00ffb4;letter-spacing:0.15em;font-weight:600'>HEALTH REPORT</div>
            <h2 style='margin:0.3rem 0 0;font-size:2rem'>AI Health Guardian</h2>
            <div style='font-size:0.8rem;color:#475569;margin-top:0.3rem'>{now_str}</div>
        </div>""", unsafe_allow_html=True)

        r1,r2 = st.columns(2)
        with r1:
            st.markdown("#### 👤 Patient Information")
            st.markdown(f"""<div class='g-card'>
            <table style='width:100%;border-collapse:collapse;color:#94a3b8'>
            <tr><td style='padding:7px 0;color:#475569;font-size:0.8rem;width:40%'>Full Name</td>
                <td style='padding:7px 0;font-weight:600;color:#e2e8f0'>{name}</td></tr>
            <tr><td style='padding:7px 0;color:#475569;font-size:0.8rem'>Age</td>
                <td style='padding:7px 0;font-weight:600;color:#e2e8f0'>{age} years</td></tr>
            <tr><td style='padding:7px 0;color:#475569;font-size:0.8rem'>Gender</td>
                <td style='padding:7px 0;font-weight:600;color:#e2e8f0'>{gender}</td></tr>
            <tr><td style='padding:7px 0;color:#475569;font-size:0.8rem'>Blood Group</td>
                <td style='padding:7px 0;font-weight:600;color:#e2e8f0'>{blood}</td></tr>
            </table></div>""", unsafe_allow_html=True)

        with r2:
            st.markdown("#### 📏 Vital Statistics")
            st.markdown(f"""<div class='g-card'>
            <table style='width:100%;border-collapse:collapse'>
            <tr><td style='padding:7px 0;color:#475569;font-size:0.8rem;width:40%'>Weight</td>
                <td style='padding:7px 0;font-weight:600;color:#e2e8f0'>{weight} kg</td></tr>
            <tr><td style='padding:7px 0;color:#475569;font-size:0.8rem'>Height</td>
                <td style='padding:7px 0;font-weight:600;color:#e2e8f0'>{height} cm</td></tr>
            <tr><td style='padding:7px 0;color:#475569;font-size:0.8rem'>BMI</td>
                <td style='padding:7px 0;font-family:Syne,sans-serif;font-weight:800;font-size:1.1rem;color:{bmi_c}'>{bmi:.1f}</td></tr>
            <tr><td style='padding:7px 0;color:#475569;font-size:0.8rem'>Category</td>
                <td style='padding:7px 0;font-weight:600;color:{bmi_c}'>{bmi_cat}</td></tr>
            </table></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        r3,r4 = st.columns(2)
        with r3:
            st.markdown("#### 🩺 Reported Symptoms")
            if symptoms:
                tags = "".join(f"<span class='pill'>🔸 {s.strip()}</span>" for s in symptoms.replace(",","\n").splitlines() if s.strip())
                st.markdown(f"<div class='g-card'>{tags}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='g-card'><span style='color:#334155'>None reported</span></div>", unsafe_allow_html=True)

        with r4:
            st.markdown("#### 💊 Conditions / Medications")
            if diseases:
                tags = "".join(f"<span class='pill pill-blue'>💊 {d.strip()}</span>" for d in diseases.replace(",","\n").splitlines() if d.strip())
                st.markdown(f"<div class='g-card'>{tags}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='g-card'><span style='color:#334155'>None reported</span></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.warning("⚠️ AI-generated report for informational purposes only. Consult a qualified healthcare professional.")

        # Clean downloadable HTML
        dl = f"""<!DOCTYPE html><html><head><meta charset='UTF-8'>
<title>Health Report — {name}</title>
<style>
  body{{font-family:Arial,sans-serif;max-width:820px;margin:40px auto;padding:24px;color:#1e293b;background:#f8fafc}}
  .header{{background:linear-gradient(135deg,#0f172a,#1e3a5f);color:white;padding:28px;border-radius:16px;margin-bottom:24px}}
  .header h1{{margin:0;font-size:1.8rem;color:#38bdf8}} .header p{{margin:4px 0 0;color:#94a3b8;font-size:0.85rem}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
  .card{{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:20px}}
  .card h3{{margin:0 0 12px;color:#0f172a;font-size:1rem;border-bottom:2px solid #e2f8f2;padding-bottom:8px}}
  table{{width:100%;border-collapse:collapse}} td{{padding:7px 4px;font-size:0.88rem}}
  td:first-child{{color:#64748b;width:40%}} td:last-child{{font-weight:600;color:#0f172a}}
  .tag{{display:inline-block;background:#f0fdf9;color:#0d9488;border:1px solid #99f6e4;
        padding:3px 12px;border-radius:20px;margin:3px;font-size:0.82rem}}
  .tag-blue{{background:#eff6ff;color:#1d4ed8;border-color:#bfdbfe}}
  .bmi-good{{color:#059669;font-weight:800;font-size:1.1rem}}
  .bmi-warn{{color:#d97706;font-weight:800;font-size:1.1rem}}
  .bmi-bad {{color:#dc2626;font-weight:800;font-size:1.1rem}}
  .footer{{margin-top:24px;padding:16px;background:#fef9c3;border-radius:10px;font-size:0.8rem;color:#92400e}}
</style></head><body>
<div class='header'><h1>🏥 AI Health Guardian — Health Report</h1><p>Generated: {now_str}</p></div>
<div class='grid'>
  <div class='card'><h3>👤 Patient Information</h3>
  <table><tr><td>Name</td><td>{name}</td></tr><tr><td>Age</td><td>{age} years</td></tr>
  <tr><td>Gender</td><td>{gender}</td></tr><tr><td>Blood Group</td><td>{blood}</td></tr></table></div>
  <div class='card'><h3>📏 Vital Statistics</h3>
  <table><tr><td>Weight</td><td>{weight} kg</td></tr><tr><td>Height</td><td>{height} cm</td></tr>
  <tr><td>BMI</td><td class='{"bmi-good" if bmi_cat=="Normal" else "bmi-warn" if bmi_cat in ("Overweight","Underweight") else "bmi-bad"}'>{bmi:.1f}</td></tr>
  <tr><td>Category</td><td>{bmi_cat}</td></tr></table></div>
</div>
<div class='grid'>
  <div class='card'><h3>🩺 Reported Symptoms</h3>
  {"".join(f"<span class='tag'>{s.strip()}</span>" for s in symptoms.split(",") if s.strip()) if symptoms else "<span style='color:#94a3b8'>None reported</span>"}</div>
  <div class='card'><h3>💊 Conditions / Medications</h3>
  {"".join(f"<span class='tag tag-blue'>{d.strip()}</span>" for d in diseases.split(",") if d.strip()) if diseases else "<span style='color:#94a3b8'>None reported</span>"}</div>
</div>
<div class='footer'>⚠️ This report is generated by AI and is for informational purposes only. 
Please consult a qualified healthcare professional for medical advice, diagnosis, or treatment.</div>
</body></html>"""

        st.download_button("⬇️ Download Report (HTML)", data=dl,
            file_name=f"health_report_{name.replace(' ','_')}.html",
            mime="text/html", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.markdown("<div class='sec-head'>ℹ️ About AI Health Guardian</div>", unsafe_allow_html=True)
    st.markdown("""<div class='g-card'>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#00ffb4;margin-bottom:0.6rem'>🎯 SDG 3: Good Health and Well-Being</div>
        <p style='font-size:0.88rem;line-height:1.7'>This project was built as part of the UN Sustainable Development Goals initiative.
        AI Health Guardian makes basic health prediction and information accessible to everyone using
        Machine Learning and Generative AI.</p>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#38bdf8;margin:1rem 0 0.6rem'>🛠️ Tech Stack</div>
        <div>
        <span class='pill'>Python 3.9+</span><span class='pill'>Streamlit</span><span class='pill'>Scikit-learn</span>
        <span class='pill'>Random Forest</span><span class='pill'>Gemini AI</span><span class='pill'>Plotly</span>
        <span class='pill'>Pandas</span><span class='pill'>NumPy</span>
        </div>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#a78bfa;margin:1rem 0 0.6rem'>📊 Model Performance</div>
        <div>
        <span class='pill pill-blue'>41 Diseases</span><span class='pill pill-blue'>131 Symptoms</span>
        <span class='pill pill-blue'>4,920 Training Samples</span><span class='pill pill-blue'>100% Accuracy</span>
        <span class='pill pill-blue'>Random Forest · 200 Trees</span>
        </div>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#ffa500;margin:1rem 0 0.6rem'>⚠️ Disclaimer</div>
        <p style='font-size:0.82rem;color:#64748b'>This tool is for educational and informational purposes only.
        It does not replace professional medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare professional.</p>
    </div>""", unsafe_allow_html=True)