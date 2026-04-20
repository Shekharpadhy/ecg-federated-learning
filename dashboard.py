"""Streamlit dashboard — ECG Federated Learning.   Run: streamlit run dashboard.py"""

import io, json, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

RESULTS_DIR = Path("results")
PROJECT_DIR = Path(__file__).parent
RECORDS_ALL = ["100","101","103","105","106","108","109","111","112","113"]

CLASS_INFO = {
    "N":(  "Normal beat",                       "#00c9a7"),
    "L":(  "Left bundle branch block",          "#3b82f6"),
    "V":(  "Premature ventricular contraction", "#ef4444"),
    "A":(  "Atrial premature beat",             "#a855f7"),
    "a":(  "Aberrated atrial premature beat",   "#8b5cf6"),
    "F":(  "Fusion of ventricular & normal",    "#f97316"),
    "+":(  "Paced beat",                        "#06b6d4"),
    "~":(  "Signal quality change",             "#64748b"),
    "|":(  "Isolated QRS-like artifact",        "#94a3b8"),
    "x":(  "Non-conducted P-wave",              "#dc2626"),
    "Q":(  "Unclassifiable beat",               "#475569"),
}

st.set_page_config(
    page_title="ECG — Federated Learning",
    layout="wide",
    page_icon="🫀",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════════
# MASTER CSS
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#07101e; --surface:#0d1a2e; --surface-2:#13243c;
  --border:rgba(148,163,184,0.08); --border-hi:rgba(0,201,167,0.28);
  --accent:#00c9a7; --accent-rgb:0,201,167;
  --red:#ff4757; --blue:#3b82f6;
  --text:#e2e8f0; --muted:#4e657a; --muted-2:#7a92a8;
  --font:'Outfit',sans-serif; --mono:'JetBrains Mono',monospace;
  --r:14px; --r-sm:8px; --r-lg:22px;
}

*,*::before,*::after{box-sizing:border-box;}

html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"],.main{
  font-family:var(--font)!important;
  background:var(--bg)!important;
  color:var(--text)!important;
}
.main .block-container{padding:2rem 2.5rem 4rem!important;max-width:1440px!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--surface-2);border-radius:2px;}
::-webkit-scrollbar-thumb:hover{background:var(--muted);}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{
  background:var(--surface)!important;
  border-right:1px solid var(--border)!important;
}
[data-testid="stSidebar"]>div:first-child{background:transparent!important;padding:1.5rem 1.1rem!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stSidebar"] hr{border-color:var(--border)!important;opacity:1!important;margin:1rem 0!important;}
[data-testid="stSidebar"] label{
  color:var(--muted-2)!important;font-size:0.72rem!important;
  font-weight:600!important;letter-spacing:0.07em!important;text-transform:uppercase!important;
}
[data-testid="stSidebar"] .stSelectbox>div>div{
  background:var(--surface-2)!important;border:1px solid var(--border)!important;
  border-radius:var(--r-sm)!important;color:var(--text)!important;font-family:var(--font)!important;
}

/* ── BUTTONS ── */
[data-testid="baseButton-primary"]{
  background:var(--accent)!important;color:#07101e!important;border:none!important;
  border-radius:var(--r-sm)!important;font-family:var(--font)!important;
  font-weight:700!important;font-size:0.875rem!important;letter-spacing:0.01em!important;
  transition:all .2s cubic-bezier(.16,1,.3,1)!important;
}
[data-testid="baseButton-primary"]:hover{
  background:#00e6bf!important;transform:translateY(-1px)!important;
  box-shadow:0 4px 22px rgba(var(--accent-rgb),.32)!important;
}
[data-testid="baseButton-primary"]:active{transform:translateY(0) scale(.99)!important;}
[data-testid="baseButton-secondary"]{
  background:var(--surface-2)!important;border:1px solid var(--border)!important;
  color:var(--text)!important;border-radius:var(--r-sm)!important;
  font-family:var(--font)!important;font-weight:500!important;
  transition:all .2s cubic-bezier(.16,1,.3,1)!important;
}
[data-testid="baseButton-secondary"]:hover{border-color:var(--border-hi)!important;}

/* ── TABS ── */
[data-baseweb="tab-list"]{
  background:var(--surface)!important;border-radius:var(--r)!important;
  padding:4px!important;border:1px solid var(--border)!important;gap:2px!important;
}
[data-baseweb="tab"]{
  background:transparent!important;border-radius:10px!important;
  color:var(--muted-2)!important;font-family:var(--font)!important;
  font-size:0.82rem!important;font-weight:500!important;
  padding:7px 16px!important;transition:all .18s ease!important;border:none!important;
}
[data-baseweb="tab"]:hover{color:var(--text)!important;background:var(--surface-2)!important;}
[aria-selected="true"][data-baseweb="tab"]{
  background:var(--surface-2)!important;color:var(--accent)!important;font-weight:600!important;
}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important;}

/* ── METRICS (native st.metric) ── */
[data-testid="metric-container"]{
  background:var(--surface)!important;border:1px solid var(--border)!important;
  border-radius:var(--r)!important;padding:1.2rem 1.4rem!important;
}
[data-testid="metric-container"] label{
  color:var(--muted-2)!important;font-size:0.7rem!important;font-weight:600!important;
  text-transform:uppercase!important;letter-spacing:0.07em!important;font-family:var(--font)!important;
}
[data-testid="stMetricValue"]{
  color:var(--text)!important;font-size:1.6rem!important;font-weight:700!important;
  font-family:var(--mono)!important;font-variant-numeric:tabular-nums!important;
}
[data-testid="stMetricDelta"]{font-family:var(--mono)!important;font-size:0.75rem!important;}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"]{border-radius:var(--r)!important;overflow:hidden!important;border:1px solid var(--border)!important;}
.dvn-scroller{background:var(--surface)!important;}
[data-testid="stDataFrame"] th{
  background:var(--surface-2)!important;color:var(--muted-2)!important;
  font-size:0.69rem!important;font-weight:600!important;letter-spacing:0.07em!important;
  text-transform:uppercase!important;border-bottom:1px solid var(--border)!important;
  font-family:var(--font)!important;
}
[data-testid="stDataFrame"] td{
  font-size:0.83rem!important;border-bottom:1px solid rgba(148,163,184,.04)!important;
  color:var(--text)!important;font-family:var(--font)!important;
}
[data-testid="stDataFrame"] tr:hover td{background:rgba(var(--accent-rgb),.04)!important;}

/* ── TABLE ── */
[data-testid="stTable"] table{
  background:var(--surface)!important;border-collapse:collapse!important;
  width:100%!important;font-family:var(--font)!important;border-radius:var(--r)!important;overflow:hidden!important;
}
[data-testid="stTable"] th{
  background:var(--surface-2)!important;color:var(--muted-2)!important;
  font-size:0.69rem!important;text-transform:uppercase!important;letter-spacing:0.07em!important;
  padding:10px 16px!important;border-bottom:1px solid var(--border)!important;font-weight:600!important;
}
[data-testid="stTable"] td{
  padding:9px 16px!important;border-bottom:1px solid rgba(148,163,184,.04)!important;
  font-size:0.83rem!important;color:var(--text)!important;
}

/* ── ALERTS ── */
[data-testid="stAlert"]{border-radius:var(--r-sm)!important;border:none!important;font-family:var(--font)!important;font-size:0.875rem!important;}

/* ── CODE ── */
.stCodeBlock>div{background:var(--surface)!important;border-radius:var(--r-sm)!important;border:1px solid var(--border)!important;}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"]{
  background:var(--surface)!important;border:1px dashed rgba(var(--accent-rgb),.22)!important;
  border-radius:var(--r)!important;transition:border-color .2s!important;
}
[data-testid="stFileUploader"]:hover{border-color:rgba(var(--accent-rgb),.45)!important;}
[data-testid="stFileUploadDropzone"]{background:transparent!important;border:none!important;}

/* ── PROGRESS ── */
[data-testid="stProgressBar"]>div{background:var(--surface-2)!important;border-radius:99px!important;}
[data-testid="stProgressBar"]>div>div{background:var(--accent)!important;border-radius:99px!important;}

/* ── EXPANDER ── */
[data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:var(--r-sm)!important;}
[data-testid="stExpander"] summary{font-family:var(--font)!important;color:var(--muted-2)!important;font-size:0.83rem!important;}

/* ── DOWNLOAD ── */
[data-testid="stDownloadButton"] button{
  background:var(--surface-2)!important;border:1px solid var(--border)!important;
  color:var(--accent)!important;font-family:var(--font)!important;
  font-weight:500!important;border-radius:var(--r-sm)!important;transition:all .2s!important;
}
[data-testid="stDownloadButton"] button:hover{border-color:var(--accent)!important;}

hr{border-color:var(--border)!important;opacity:1!important;margin:1.5rem 0!important;}
h1,h2,h3,h4{font-family:var(--font)!important;font-weight:700!important;color:var(--text)!important;letter-spacing:-.02em!important;}
p{font-family:var(--font)!important;color:var(--muted-2)!important;}

/* ════ WOW ELEMENT 1 — Animated ECG waveform ════ */
.ecg-container{position:relative;width:100%;height:72px;overflow:hidden;}
.ecg-fade{
  position:absolute;inset:0;
  background:linear-gradient(90deg,var(--bg) 0%,transparent 12%,transparent 88%,var(--bg) 100%);
  z-index:2;pointer-events:none;
}
.ecg-svg{width:200%;height:100%;animation:ecg-scroll 3.2s linear infinite;}
.ecg-line{fill:none;stroke:var(--accent);stroke-width:2;stroke-linecap:round;stroke-linejoin:round;filter:drop-shadow(0 0 5px rgba(var(--accent-rgb),.75));}
@keyframes ecg-scroll{from{transform:translateX(0);}to{transform:translateX(-50%);}}
.ecg-unit{font-family:var(--mono);font-size:0.6rem;color:var(--muted);text-align:center;margin-top:3px;letter-spacing:.1em;}

/* ════ WOW ELEMENT 2 — Spinning conic-gradient card borders ════ */
@property --ba{syntax:'<angle>';initial-value:0deg;inherits:false;}
.spin-card{
  position:relative;border-radius:var(--r);padding:1px;
  background:conic-gradient(from var(--ba),transparent 65%,rgba(var(--accent-rgb),.55),transparent 35%);
  animation:spin-card-border 6s linear infinite;
}
@keyframes spin-card-border{to{--ba:360deg;}}
.spin-card-inner{background:var(--surface);border-radius:calc(var(--r) - 1px);padding:1.3rem 1.5rem;height:100%;}
.sc-label{font-size:.68rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:.35rem;font-family:var(--font);}
.sc-value{font-family:var(--mono);font-size:1.55rem;font-weight:700;color:var(--text);font-variant-numeric:tabular-nums;line-height:1.1;}
.sc-value.hi{color:var(--accent);}
.sc-value.warn{color:var(--red);}
.sc-delta{font-family:var(--mono);font-size:.7rem;color:var(--muted);margin-top:.25rem;}
.sc-delta.pos{color:var(--accent);}
.sc-delta.neg{color:var(--red);}

/* ════ WOW ELEMENT 3 — Ambient floating orbs ════ */
.orbs-layer{position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;}
.orb{position:absolute;border-radius:50%;filter:blur(110px);}
.orb-a{width:650px;height:650px;background:var(--accent);opacity:.055;top:-280px;right:-180px;animation:orb-float 28s ease-in-out infinite;}
.orb-b{width:420px;height:420px;background:var(--blue);opacity:.05;bottom:-170px;left:-120px;animation:orb-float 35s ease-in-out infinite reverse;animation-delay:-14s;}
.orb-c{width:280px;height:280px;background:var(--red);opacity:.04;top:42%;left:48%;animation:orb-float 22s ease-in-out infinite;animation-delay:-8s;}
@keyframes orb-float{0%,100%{transform:translate(0,0) scale(1);}33%{transform:translate(45px,-45px) scale(1.07);}66%{transform:translate(-28px,32px) scale(.94);}}

/* Grain overlay */
.grain{
  position:fixed;inset:0;pointer-events:none;z-index:1;opacity:.028;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)'/%3E%3C/svg%3E");
  background-size:200px 200px;
}

/* ── HERO ── */
.hero{display:flex;align-items:center;justify-content:space-between;padding:2.2rem 0 1.8rem;gap:2rem;position:relative;z-index:2;}
.hero-left{flex:1;min-width:0;}
.hero-badge{
  display:inline-flex;align-items:center;gap:6px;
  background:rgba(var(--accent-rgb),.09);border:1px solid rgba(var(--accent-rgb),.22);
  color:var(--accent);font-size:.68rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  padding:4px 12px;border-radius:99px;margin-bottom:.9rem;font-family:var(--font);
}
.badge-dot{width:5px;height:5px;background:var(--accent);border-radius:50%;animation:pdot 2s ease-in-out infinite;}
@keyframes pdot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.3;transform:scale(.6);}}
.hero-h1{font-family:var(--font)!important;font-size:clamp(1.9rem,3.5vw,3rem);font-weight:800;
  color:var(--text)!important;letter-spacing:-.04em;line-height:1.05;margin:0 0 .7rem;text-wrap:balance;}
.hero-h1 span{color:var(--accent);}
.hero-sub{font-family:var(--font);font-size:.88rem;color:var(--muted-2);line-height:1.65;max-width:460px;margin-bottom:1.4rem;}
.hero-stats{display:flex;gap:2rem;flex-wrap:wrap;}
.hs-num{font-family:var(--mono);font-size:1.3rem;font-weight:700;color:var(--text);font-variant-numeric:tabular-nums;}
.hs-lbl{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;font-weight:500;margin-top:1px;font-family:var(--font);}
.hero-right{flex:0 0 400px;}

/* ── SECTION HEADER ── */
.sh{display:flex;align-items:baseline;gap:.65rem;margin-bottom:1.1rem;}
.sh-title{font-family:var(--font);font-size:.92rem;font-weight:700;color:var(--text);letter-spacing:-.01em;}
.sh-pill{
  font-family:var(--mono);font-size:.62rem;font-weight:500;color:var(--muted);
  background:var(--surface-2);border:1px solid var(--border);padding:2px 7px;border-radius:99px;
}

/* ── GLASS PANEL ── */
.gp{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:1.4rem;}

/* ── ACCURACY BANNER ── */
.acc-banner{
  display:flex;align-items:center;gap:1.5rem;padding:1.2rem 1.5rem;
  background:linear-gradient(135deg,rgba(var(--accent-rgb),.07) 0%,transparent 100%);
  border:1px solid rgba(var(--accent-rgb),.14);border-radius:var(--r);margin-bottom:1.4rem;
}
.acc-big{font-family:var(--mono);font-size:2.4rem;font-weight:800;color:var(--accent);font-variant-numeric:tabular-nums;line-height:1;}
.acc-lbl{font-family:var(--font);}
.acc-lbl strong{display:block;color:var(--text);font-size:.92rem;font-weight:600;margin-bottom:2px;}
.acc-lbl span{font-size:.78rem;color:var(--muted-2);}

/* ── LIVE TRAINING ── */
.live-header{display:flex;align-items:center;gap:8px;margin-bottom:.7rem;}
.live-dot{width:7px;height:7px;background:var(--red);border-radius:50%;animation:blink 1s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.15;}}
.live-lbl{font-size:.68rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--muted-2);font-family:var(--font);}

/* ── INLINE METRIC GRID ── */
.imr{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:1px;background:var(--border);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;}
.imc{background:var(--surface);padding:1rem 1.2rem;}
.imc-lbl{font-size:.65rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:3px;font-family:var(--font);}
.imc-val{font-family:var(--mono);font-size:1.15rem;font-weight:700;color:var(--text);font-variant-numeric:tabular-nums;}

/* ── SIDEBAR CONFIG TABLE ── */
.cfg{width:100%;}
.cfg tr{border-bottom:1px solid rgba(148,163,184,.05);}
.cfg td{padding:5px 0;font-family:var(--font);font-size:.78rem;}
.cfg .ck{color:var(--muted-2);}
.cfg .cv{color:var(--accent);font-family:var(--mono);font-size:.75rem;text-align:right;}

/* ── EMPTY STATE ── */
.empty{text-align:center;padding:2.5rem 1rem;}
.empty-icon{width:44px;height:44px;margin:0 auto .9rem;background:var(--surface-2);border-radius:12px;
  display:flex;align-items:center;justify-content:center;border:1px solid var(--border);}
.empty-title{font-size:.9rem;font-weight:600;color:var(--text);font-family:var(--font);margin-bottom:.35rem;}
.empty-sub{font-size:.8rem;color:var(--muted-2);font-family:var(--font);line-height:1.55;}
</style>
""", unsafe_allow_html=True)

# Ambient layer (orbs + grain) — renders once outside tabs
st.markdown("""
<div class="orbs-layer">
  <div class="orb orb-a"></div>
  <div class="orb orb-b"></div>
  <div class="orb orb-c"></div>
</div>
<div class="grain"></div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=5)
def load_baseline():
    p = RESULTS_DIR / "baseline_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None

@st.cache_data(ttl=5)
def load_fed():
    p = RESULTS_DIR / "federated_history.json"
    return json.loads(p.read_text()) if p.exists() else None

@st.cache_resource
def load_model():
    p = RESULTS_DIR / "baseline_model.pt"
    if not p.exists():
        return None, None
    try:
        sys.path.insert(0, str(PROJECT_DIR))
        from src.model import build_model
        ckpt  = torch.load(p, map_location="cpu")
        model = build_model(ckpt["input_dim"], ckpt["num_classes"], ckpt.get("model_type","fc"))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt["classes"]
    except Exception:
        return None, None

bm  = load_baseline()
fed = load_fed()

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.4rem;padding-bottom:1rem;border-bottom:1px solid var(--border)">
      <div style="width:34px;height:34px;background:rgba(0,201,167,.1);border:1px solid rgba(0,201,167,.25);
                  border-radius:9px;display:flex;align-items:center;justify-content:center;flex-shrink:0">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00c9a7" stroke-width="2" stroke-linecap="round">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
        </svg>
      </div>
      <div>
        <div style="font-family:var(--font);font-size:.9rem;font-weight:700;color:var(--text);letter-spacing:-.01em">ECG · FedLearn</div>
        <div style="font-size:.62rem;color:var(--muted);font-weight:400;letter-spacing:.03em">Privacy-Preserving AI</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section-header" style="font-size:.62rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)!important;margin-bottom:.5rem">Run Training</p>', unsafe_allow_html=True)

    mode = st.selectbox("Mode", ["baseline","federated","explain"],
        label_visibility="collapsed",
        format_func=lambda m:{
            "baseline" :"Baseline (Centralized)",
            "federated":"Federated — 5 Hospitals",
            "explain"  :"SHAP Explainability",
        }[m])
    run_btn = st.button("Run", use_container_width=True, type="primary")

    st.divider()

    # Dataset status
    data_dir = PROJECT_DIR / "data" / "mit-bih-arrhythmia-database-1.0.0"
    present  = [r for r in RECORDS_ALL if (data_dir / f"{r}.hea").exists()]
    st.markdown(f'<p style="font-size:.62rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)">Dataset · {len(present)}/{len(RECORDS_ALL)} records</p>', unsafe_allow_html=True)
    st.progress(len(present) / len(RECORDS_ALL))

    if len(present) < len(RECORDS_ALL):
        if st.button("Download MIT-BIH", use_container_width=True):
            with st.spinner("Downloading from PhysioNet…"):
                dl = subprocess.run(
                    [sys.executable,"-c",
                     "import sys;sys.path.insert(0,'.');from src.download_data import download_mitbih;download_mitbih()"],
                    capture_output=True,text=True,cwd=str(PROJECT_DIR))
            if dl.returncode == 0:
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Download failed"); st.code(dl.stderr[-1200:])
    else:
        st.markdown('<p style="font-size:.75rem;color:var(--accent);font-weight:600">All 10 records ready</p>', unsafe_allow_html=True)

    st.divider()

    base_acc = bm["accuracy"] if bm else None
    fed_acc  = fed["rounds"][-1]["accuracy"] if (fed and fed.get("rounds")) else None

    st.markdown(f"""
    <table class="cfg">
      <tr><td class="ck">Model</td><td class="cv">FC Network</td></tr>
      <tr><td class="ck">Epochs</td><td class="cv">7</td></tr>
      <tr><td class="ck">Batch</td><td class="cv">512</td></tr>
      <tr><td class="ck">Clients</td><td class="cv">5 hospitals</td></tr>
      <tr><td class="ck">Fed rounds</td><td class="cv">3</td></tr>
      <tr><td class="ck">Records</td><td class="cv">10 MIT-BIH</td></tr>
      <tr><td class="ck">Beats</td><td class="cv">21,849</td></tr>
      <tr><td class="ck">Centralized</td><td class="cv">{"%.2f%%" % (base_acc*100) if base_acc else "—"}</td></tr>
      <tr><td class="ck">Federated</td><td class="cv">{"%.2f%%" % (fed_acc*100) if fed_acc else "—"}</td></tr>
    </table>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# LIVE TRAINING
# ════════════════════════════════════════════════════════════════════════════════
if run_btn:
    st.markdown(f"""
    <div class="live-header" style="margin-top:1rem">
      <div class="live-dot"></div>
      <span class="live-lbl">Live — running {mode}</span>
    </div>
    """, unsafe_allow_html=True)
    log_box   = st.empty()
    chart_box = st.empty()
    log_lines, epoch_rows = [], []

    proc = subprocess.Popen(
        [sys.executable,"main.py","--mode",mode],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(PROJECT_DIR), bufsize=1,
    )
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line or "urllib3" in line or "NotOpenSSL" in line or "warnings.warn" in line:
            continue
        log_lines.append(line)
        log_box.code("\n".join(log_lines[-20:]), language="bash")
        if "Epoch" in line and "loss=" in line and "acc=" in line:
            try:
                parts = line.split()
                ep   = int(next(p for p in parts if "/" in p).split("/")[0])
                loss = float(next(p for p in parts if p.startswith("loss=")).split("=")[1])
                acc  = float(next(p for p in parts if p.startswith("acc=")).split("=")[1])
                epoch_rows.append({"Epoch":ep,"Loss":loss,"Accuracy":acc})
                if len(epoch_rows) > 1:
                    chart_box.line_chart(pd.DataFrame(epoch_rows).set_index("Epoch"), height=200)
            except Exception:
                pass
    proc.wait()
    if proc.returncode == 0:
        st.cache_data.clear(); st.cache_resource.clear()
        st.rerun()
    else:
        st.error("Training failed — see log above.")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# HERO HEADER  (WOW ELEMENT 1 — animated ECG waveform)
# ════════════════════════════════════════════════════════════════════════════════
base_acc = bm["accuracy"] if bm else None
fed_acc  = fed["rounds"][-1]["accuracy"] if (fed and fed.get("rounds")) else None
n_cls    = len(bm["classes"]) if bm else "11"

# ECG path: two identical copies for seamless translateX(-50%) loop
# Each copy is 600px wide with P-wave + QRS spike + T-wave
ECG_BEAT = "M0,45 L55,45 Q65,31 75,45 L92,45 L98,53 L104,7 L110,60 L128,45 Q148,30 168,45 L300,45"
ECG_PATH = f"{ECG_BEAT} M300,45 L355,45 Q365,31 375,45 L392,45 L398,53 L404,7 L410,60 L428,45 Q448,30 468,45 L600,45"

st.markdown(f"""
<div class="hero">
  <div class="hero-left">
    <div class="hero-badge"><div class="badge-dot"></div>Federated Learning · ECG AI</div>
    <h1 class="hero-h1">Arrhythmia detection<br>across <span>5 hospitals</span></h1>
    <p class="hero-sub">Privacy-preserving classification on 21,849 ECG beats from 10 MIT-BIH records — no raw patient data ever shared.</p>
    <div class="hero-stats">
      <div><div class="hs-num">21,849</div><div class="hs-lbl">Total beats</div></div>
      <div><div class="hs-num">{n_cls}</div><div class="hs-lbl">Arrhythmia classes</div></div>
      <div><div class="hs-num">{"%.1f%%" % (base_acc*100) if base_acc else "—"}</div><div class="hs-lbl">Centralized acc.</div></div>
      <div><div class="hs-num">{"%.1f%%" % (fed_acc*100) if fed_acc else "—"}</div><div class="hs-lbl">Federated acc.</div></div>
    </div>
  </div>
  <div class="hero-right">
    <div class="ecg-container">
      <div class="ecg-fade"></div>
      <svg class="ecg-svg" viewBox="0 0 600 72" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
        <path class="ecg-line" d="{ECG_PATH}"/>
      </svg>
    </div>
    <div class="ecg-unit">LEAD-I · MIT-BIH · 360 Hz</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SPINNING STAT CARDS  (WOW ELEMENT 2 — conic gradient borders)
# ════════════════════════════════════════════════════════════════════════════════
gap  = (base_acc - fed_acc) if (base_acc and fed_acc) else None

cards_html = f"""
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:1.8rem;position:relative;z-index:2">

  <div class="spin-card">
    <div class="spin-card-inner">
      <div class="sc-label">Centralized Acc.</div>
      <div class="sc-value hi">{"%.2f%%" % (base_acc*100) if base_acc else "—"}</div>
      <div class="sc-delta">7 epochs · FC model</div>
    </div>
  </div>

  <div class="spin-card">
    <div class="spin-card-inner">
      <div class="sc-label">Federated Acc.</div>
      <div class="sc-value {"warn" if (fed_acc and fed_acc < 0.85) else "hi"}">{"%.2f%%" % (fed_acc*100) if fed_acc else "—"}</div>
      <div class="sc-delta {"neg" if gap and gap > 0.05 else "pos"}">{"−%.2f%% vs baseline" % (gap*100) if gap else "Run federated mode"}</div>
    </div>
  </div>

  <div class="spin-card">
    <div class="spin-card-inner">
      <div class="sc-label">Training beats</div>
      <div class="sc-value">17,479</div>
      <div class="sc-delta">80% stratified split</div>
    </div>
  </div>

  <div class="spin-card">
    <div class="spin-card-inner">
      <div class="sc-label">Arrhythmia classes</div>
      <div class="sc-value">{n_cls}</div>
      <div class="sc-delta">11 MIT-BIH symbols</div>
    </div>
  </div>

  <div class="spin-card">
    <div class="spin-card-inner">
      <div class="sc-label">Records used</div>
      <div class="sc-value">10</div>
      <div class="sc-delta">of 48 MIT-BIH total</div>
    </div>
  </div>

</div>
"""
st.markdown(cards_html, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ════════════════════════════════════════════════════════════════════════════════
t_ov, t_train, t_fed, t_shap, t_up, t_data = st.tabs([
    "Overview", "Training", "Federated", "SHAP", "Upload & Analyze", "Dataset"
])

# ── TAB 1 : OVERVIEW ─────────────────────────────────────────────────────────
with t_ov:
    if bm and fed and fed.get("rounds"):
        ch1, ch2 = st.columns([1, 1.2])
        with ch1:
            st.markdown('<div class="sh"><span class="sh-title">Centralized vs Federated</span><span class="sh-pill">accuracy</span></div>', unsafe_allow_html=True)
            compare_df = pd.DataFrame({
                "Mode":["Centralized","Federated"],
                "Accuracy":[bm["accuracy"], fed["rounds"][-1]["accuracy"]],
            })
            st.bar_chart(compare_df.set_index("Mode"), height=260, color="#00c9a7")

        with ch2:
            st.markdown('<div class="sh"><span class="sh-title">Federated accuracy per round</span><span class="sh-pill">3 rounds</span></div>', unsafe_allow_html=True)
            rdf = pd.DataFrame(fed["rounds"]).rename(columns={"round":"Round","accuracy":"Federated Accuracy"})
            rdf["Centralized"] = bm["accuracy"]
            st.line_chart(rdf.set_index("Round")[["Federated Accuracy","Centralized"]], height=260)
    else:
        st.markdown("""
        <div class="gp empty">
          <div class="empty-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4e657a" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div>
          <div class="empty-title">No training results yet</div>
          <div class="empty-sub">Run <strong style="color:var(--accent)">Baseline</strong> then <strong style="color:var(--accent)">Federated</strong> from the sidebar to populate charts.</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sh"><span class="sh-title">System architecture</span></div>', unsafe_allow_html=True)
    arch = RESULTS_DIR / "system_architecture.png"
    if arch.exists():
        st.image(str(arch), use_container_width=True)

# ── TAB 2 : TRAINING ─────────────────────────────────────────────────────────
with t_train:
    if bm:
        st.markdown(f"""
        <div class="acc-banner">
          <div class="acc-big">{"%.2f%%" % (bm["accuracy"]*100)}</div>
          <div class="acc-lbl">
            <strong>Centralized baseline accuracy</strong>
            <span>{bm.get("model_type","fc").upper()} · 7 epochs · {bm.get("num_samples",17479):,} training beats · batch 512</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if bm.get("epoch_history"):
            st.markdown('<div class="sh"><span class="sh-title">Per-epoch progress</span><span class="sh-pill">training dynamics</span></div>', unsafe_allow_html=True)
            eh = pd.DataFrame(bm["epoch_history"]).set_index("epoch")
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown('<p style="font-size:.75rem;color:var(--muted-2);font-family:var(--font);margin-bottom:.5rem">Training accuracy</p>', unsafe_allow_html=True)
                st.line_chart(eh[["accuracy"]], height=220, color="#00c9a7")
            with ec2:
                st.markdown('<p style="font-size:.75rem;color:var(--muted-2);font-family:var(--font);margin-bottom:.5rem">Training loss</p>', unsafe_allow_html=True)
                st.line_chart(eh[["loss"]], height=220, color="#ff4757")

            with st.expander("Epoch-by-epoch numbers"):
                disp = eh.copy()
                disp.columns = ["Loss","Accuracy"]
                disp["Accuracy"] = disp["Accuracy"].map("{:.2%}".format)
                disp["Loss"]     = disp["Loss"].map("{:.4f}".format)
                disp.index.name  = "Epoch"
                st.dataframe(disp, use_container_width=True)

        st.divider()
        st.markdown('<div class="sh"><span class="sh-title">Per-class classification report</span><span class="sh-pill">baseline</span></div>', unsafe_allow_html=True)
        rows = []
        for cls, vals in bm.get("report",{}).items():
            if isinstance(vals, dict):
                info = CLASS_INFO.get(cls, (cls,"#aaa"))
                rows.append({
                    "Symbol":cls, "Type":info[0],
                    "Precision":f"{vals['precision']:.3f}",
                    "Recall":f"{vals['recall']:.3f}",
                    "F1-Score":f"{vals['f1-score']:.3f}",
                    "Support":int(vals["support"]),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div class="gp empty">
          <div class="empty-title">No baseline results</div>
          <div class="empty-sub">Select <strong style="color:var(--accent)">Baseline (Centralized)</strong> and click Run.</div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 3 : FEDERATED ────────────────────────────────────────────────────────
with t_fed:
    st.markdown("""
    <div class="gp" style="margin-bottom:1.4rem">
      <div class="sh"><span class="sh-title">How 5-hospital simulation works</span></div>
      <p style="font-size:.83rem;color:var(--muted-2);font-family:var(--font);margin:0;line-height:1.7">
        The 17,479 training beats are split into 5 equal partitions — each representing one hospital's private records.
        Clients train locally for 2 epochs, send <em>only model weights</em> (never raw data) to the server,
        which averages them via <strong style="color:var(--text)">FedAvg</strong> and distributes the updated global model.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if fed and fed.get("rounds"):
        rounds = fed["rounds"]
        st.markdown(f"""
        <div class="imr" style="margin-bottom:1.4rem">
          <div class="imc"><div class="imc-lbl">Rounds</div><div class="imc-val">{fed.get("num_rounds",3)}</div></div>
          <div class="imc"><div class="imc-lbl">Hospitals</div><div class="imc-val">{fed.get("num_clients",5)}</div></div>
          <div class="imc"><div class="imc-lbl">Local epochs</div><div class="imc-val">2</div></div>
          <div class="imc"><div class="imc-lbl">Final accuracy</div><div class="imc-val" style="color:var(--accent)">{"%.2f%%" % (rounds[-1]["accuracy"]*100)}</div></div>
          <div class="imc"><div class="imc-lbl">Round-1 acc.</div><div class="imc-val">{"%.2f%%" % (rounds[0]["accuracy"]*100)}</div></div>
        </div>
        """, unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown('<p style="font-size:.75rem;color:var(--muted-2);font-family:var(--font);margin-bottom:.5rem">Federated accuracy per round</p>', unsafe_allow_html=True)
            rdf = pd.DataFrame(rounds).rename(columns={"round":"Round","accuracy":"Federated Accuracy"})
            if bm: rdf["Centralized"] = bm["accuracy"]
            st.line_chart(rdf.set_index("Round"), height=240)
        with fc2:
            st.markdown('<p style="font-size:.75rem;color:var(--muted-2);font-family:var(--font);margin-bottom:.5rem">Loss per round</p>', unsafe_allow_html=True)
            rdf2 = pd.DataFrame(rounds).rename(columns={"round":"Round","loss":"Loss"})
            st.line_chart(rdf2.set_index("Round")[["Loss"]], height=240, color="#ff4757")

        with st.expander("Round-by-round numbers"):
            display_r = pd.DataFrame(rounds)
            display_r["accuracy"] = display_r["accuracy"].map("{:.2%}".format)
            display_r["loss"]     = display_r["loss"].map("{:.4f}".format)
            display_r.columns     = ["Round","Accuracy","Loss"]
            st.dataframe(display_r.set_index("Round"), use_container_width=True)
    else:
        st.markdown("""
        <div class="gp empty">
          <div class="empty-title">No federated results yet</div>
          <div class="empty-sub">Select <strong style="color:var(--accent)">Federated — 5 Hospitals</strong> and click Run.</div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 4 : SHAP ─────────────────────────────────────────────────────────────
with t_shap:
    shap_img = RESULTS_DIR / "shap_summary.png"
    if shap_img.exists():
        st.image(str(shap_img), use_container_width=True)

        region_file = RESULTS_DIR / "shap_region_summary.json"
        if region_file.exists():
            ri    = json.loads(region_file.read_text())
            total = sum(ri.values())
            rows  = sorted(ri.items(), key=lambda x: x[1], reverse=True)

            st.divider()
            st.markdown('<div class="sh"><span class="sh-title">Importance by ECG region</span><span class="sh-pill">SHAP sum</span></div>', unsafe_allow_html=True)

            cols = st.columns(len(rows))
            region_colors = {"QRS complex":"#ef4444","T-wave":"#22c55e","ST segment":"#f97316",
                             "P-wave":"#3b82f6","PR interval":"#a855f7",
                             "Pre-beat baseline":"#64748b","Post-beat baseline":"#64748b"}
            for i,(region,val) in enumerate(rows):
                pct = val/total*100
                color = region_colors.get(region,"#94a3b8")
                cols[i].markdown(f"""
                <div style="text-align:center;padding:.8rem .5rem;background:var(--surface);border:1px solid var(--border);border-radius:var(--r-sm)">
                  <div style="font-family:var(--mono);font-size:1.1rem;font-weight:700;color:{color};font-variant-numeric:tabular-nums">{pct:.1f}%</div>
                  <div style="font-size:.62rem;color:var(--muted);font-family:var(--font);margin-top:3px;font-weight:500">{region}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="sh"><span class="sh-title">ECG feature map — what each sample index means</span></div>', unsafe_allow_html=True)
        st.table(pd.DataFrame([
            {"Samples":"0 – 49",      "Region":"Pre-beat baseline","Clinical meaning":"Isoelectric reference"},
            {"Samples":"50 – 79",     "Region":"P-wave",           "Clinical meaning":"Atrial depolarisation"},
            {"Samples":"80 – 94",     "Region":"PR interval",      "Clinical meaning":"AV node conduction delay"},
            {"Samples":"95 – 109",    "Region":"QRS complex ★",    "Clinical meaning":"Ventricular depolarisation — highest SHAP"},
            {"Samples":"110 – 139",   "Region":"ST segment",       "Clinical meaning":"Ischaemia / infarction marker"},
            {"Samples":"140 – 169",   "Region":"T-wave",           "Clinical meaning":"Ventricular repolarisation"},
            {"Samples":"170 – 199",   "Region":"Post-beat baseline","Clinical meaning":"Recovery period"},
        ]))
        st.markdown('<p style="font-size:.75rem;color:var(--muted);font-family:var(--font)">★ Feature 103 = sample 3 past the R-peak. High SHAP here confirms the model focuses on the QRS complex — the correct clinical region for arrhythmia detection.</p>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="gp empty">
          <div class="empty-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4e657a" stroke-width="1.5">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
          </div>
          <div class="empty-title">No SHAP analysis yet</div>
          <div class="empty-sub">Select <strong style="color:var(--accent)">SHAP Explainability</strong> and click Run.<br>Generates in ~10 seconds using GradientExplainer.</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        st.markdown('<div class="sh"><span class="sh-title">What are these features?</span></div>', unsafe_allow_html=True)
        st.table(pd.DataFrame([
            {"Samples":"0 – 49",   "Region":"Pre-beat baseline","Clinical meaning":"Isoelectric reference"},
            {"Samples":"50 – 79",  "Region":"P-wave",           "Clinical meaning":"Atrial depolarisation"},
            {"Samples":"80 – 94",  "Region":"PR interval",      "Clinical meaning":"AV conduction delay"},
            {"Samples":"95 – 109", "Region":"QRS complex ★",    "Clinical meaning":"Ventricular depolarisation — highest importance"},
            {"Samples":"110 – 139","Region":"ST segment",       "Clinical meaning":"Ischaemia / infarction marker"},
            {"Samples":"140 – 169","Region":"T-wave",           "Clinical meaning":"Ventricular repolarisation"},
            {"Samples":"170 – 199","Region":"Post-beat baseline","Clinical meaning":"Recovery period"},
        ]))

# ── TAB 5 : UPLOAD & ANALYZE ─────────────────────────────────────────────────
with t_up:
    st.markdown('<div class="sh"><span class="sh-title">Upload & analyze</span><span class="sh-pill">inference</span></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:.83rem;color:var(--muted-2);font-family:var(--font);margin-bottom:1rem">Upload a <strong style="color:var(--text)">CSV</strong> (one ECG beat per row, 200 signal columns) for live predictions, or a <strong style="color:var(--text)">JSON</strong> results file to view its metrics.</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose file", type=["csv","json"], label_visibility="collapsed")

    if uploaded:
        if uploaded.name.endswith(".json"):
            try:
                data = json.load(uploaded)
                if "accuracy" in data:
                    st.markdown(f"""
                    <div class="acc-banner">
                      <div class="acc-big">{"%.2f%%" % (data["accuracy"]*100)}</div>
                      <div class="acc-lbl"><strong>Accuracy from uploaded file</strong><span>{uploaded.name}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                if "epoch_history" in data:
                    eh = pd.DataFrame(data["epoch_history"]).set_index("epoch")
                    uc1, uc2 = st.columns(2)
                    with uc1: st.line_chart(eh[["accuracy"]], height=200, color="#00c9a7")
                    with uc2: st.line_chart(eh[["loss"]], height=200, color="#ff4757")
                if "report" in data:
                    rows = [{"Class":k,"Precision":f"{v['precision']:.3f}","Recall":f"{v['recall']:.3f}",
                             "F1":f"{v['f1-score']:.3f}","Support":int(v["support"])}
                            for k,v in data["report"].items() if isinstance(v,dict)]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                if "rounds" in data:
                    rdf = pd.DataFrame(data["rounds"]).rename(columns={"round":"Round","accuracy":"Accuracy"})
                    st.line_chart(rdf.set_index("Round"), height=200)
            except Exception as e:
                st.error(f"Could not parse JSON: {e}")

        elif uploaded.name.endswith(".csv"):
            try:
                df_up = pd.read_csv(io.StringIO(uploaded.read().decode("utf-8")), header=None)
                st.markdown(f'<p style="font-size:.8rem;color:var(--muted-2);font-family:var(--font)">Loaded <strong style="color:var(--text)">{len(df_up)}</strong> beats × {df_up.shape[1]} columns</p>', unsafe_allow_html=True)
                if df_up.shape[1] != 200:
                    st.warning(f"Expected 200 columns, got {df_up.shape[1]}. Padding/trimming automatically.")

                model, classes = load_model()
                if model is None:
                    st.error("No trained model found. Run baseline mode first.")
                else:
                    X_up = df_up.values.astype(np.float32)
                    if X_up.shape[1] < 200: X_up = np.pad(X_up, ((0,0),(0,200-X_up.shape[1])))
                    elif X_up.shape[1] > 200: X_up = X_up[:,:200]

                    with torch.no_grad():
                        probs    = torch.softmax(model(torch.tensor(X_up)), dim=1).numpy()
                        pred_idx = np.argmax(probs, axis=1)

                    pred_labels  = [classes[i] for i in pred_idx]
                    confidences  = probs.max(axis=1)
                    descriptions = [CLASS_INFO.get(l,(l,""))[0] for l in pred_labels]

                    summary = pd.Series(pred_labels).value_counts().reset_index()
                    summary.columns = ["Symbol","Count"]
                    summary["Type"] = summary["Symbol"].map(lambda s: CLASS_INFO.get(s,(s,""))[0])

                    ua, ub = st.columns(2)
                    with ua:
                        st.markdown('<div class="sh"><span class="sh-title">Prediction summary</span></div>', unsafe_allow_html=True)
                        st.dataframe(summary, use_container_width=True, hide_index=True)
                    with ub:
                        st.markdown('<div class="sh"><span class="sh-title">Class distribution</span></div>', unsafe_allow_html=True)
                        st.bar_chart(summary.set_index("Symbol")["Count"], color="#00c9a7")

                    results_df = pd.DataFrame({
                        "Beat #":range(1,len(pred_labels)+1),
                        "Symbol":pred_labels, "Arrhythmia type":descriptions,
                        "Confidence":[f"{c:.2%}" for c in confidences],
                    })
                    st.markdown('<div class="sh" style="margin-top:1rem"><span class="sh-title">Per-beat predictions</span></div>', unsafe_allow_html=True)
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    st.download_button("Download predictions CSV",
                                       results_df.to_csv(index=False).encode(),
                                       "ecg_predictions.csv","text/csv")
            except Exception as e:
                st.error(f"Error processing file: {e}")
    else:
        st.markdown("""
        <div class="gp empty" style="margin-top:.5rem">
          <div class="empty-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4e657a" stroke-width="1.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          </div>
          <div class="empty-title">Drop a file to analyze</div>
          <div class="empty-sub">CSV with 200-column ECG beats, or a JSON results file.</div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 6 : DATASET ──────────────────────────────────────────────────────────
with t_data:
    st.markdown(f"""
    <div class="imr" style="margin-bottom:1.4rem">
      <div class="imc"><div class="imc-lbl">Total beats</div><div class="imc-val">21,849</div></div>
      <div class="imc"><div class="imc-lbl">Records used</div><div class="imc-val">10 / 48</div></div>
      <div class="imc"><div class="imc-lbl">Beat window</div><div class="imc-val">200 smp</div></div>
      <div class="imc"><div class="imc-lbl">Train split</div><div class="imc-val">80%</div></div>
      <div class="imc"><div class="imc-lbl">Test split</div><div class="imc-val">20%</div></div>
    </div>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns([1.1, 1])

    with d1:
        st.markdown('<div class="sh"><span class="sh-title">MIT-BIH records</span><span class="sh-pill">10 used</span></div>', unsafe_allow_html=True)
        rec_descriptions = [
            "Normal sinus + PVCs","Normal + APBs","Normal sinus rhythm",
            "Normal + PVCs (occasional)","Normal + LBBB","AV block rhythms",
            "Normal + PVCs (frequent)","Left bundle branch block",
            "Normal sinus (clean)","Normal + PVCs",
        ]
        rec_df = pd.DataFrame({
            "Record": RECORDS_ALL,
            "Description": rec_descriptions,
            "Status": ["On disk" if (data_dir/f"{r}.hea").exists() else "Pending" for r in RECORDS_ALL],
        })
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

    with d2:
        st.markdown('<div class="sh"><span class="sh-title">Arrhythmia classes</span><span class="sh-pill">11 types</span></div>', unsafe_allow_html=True)
        report_support = {}
        if bm and "report" in bm:
            for k,v in bm["report"].items():
                if isinstance(v,dict): report_support[k] = int(v["support"])
        class_rows = [{"Symbol":sym,"Type":desc,"Test samples":report_support.get(sym,"—")}
                      for sym,(desc,_) in CLASS_INFO.items()]
        st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<div class="sh"><span class="sh-title">Sample ECG beat</span><span class="sh-pill">200 samples · 360 Hz</span></div>', unsafe_allow_html=True)
    ecg_img = RESULTS_DIR / "ecg_sample.png"
    if ecg_img.exists():
        st.image(str(ecg_img), use_container_width=True)
    st.markdown('<p style="font-size:.75rem;color:var(--muted);font-family:var(--font);margin-top:.5rem">Each beat is extracted as 100 samples before the R-peak + 100 after. The R-peak (ventricular depolarisation) sits at sample index 100.</p>', unsafe_allow_html=True)

st.divider()
st.markdown('<p style="font-size:.7rem;color:var(--muted);font-family:var(--font);text-align:center">ECG Federated Learning · PyTorch · SHAP · Streamlit · MIT-BIH PhysioNet</p>', unsafe_allow_html=True)
