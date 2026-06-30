# app.py — Customer Churn Predictor (Modern UI)
# Run: streamlit run app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import random

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Design System — Dark "ops console" theme
# Palette: Navy/Slate base, Indigo accent, Coral danger, Emerald safe
# Signature: live gauge chart sebagai focal point hasil prediksi
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.stApp { background: #0f1117 !important; }
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1a1d27; }
::-webkit-scrollbar-thumb { background: #3d4263; border-radius: 10px; }

/* HERO */
.hero {
    background: linear-gradient(135deg, #1a1d2e 0%, #16213e 40%, #0f3460 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 2.2rem 2.5rem 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace; font-size: 0.65rem; font-weight: 500;
    letter-spacing: 0.2em; color: #6366f1; text-transform: uppercase;
    margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.5rem;
}
.hero-eyebrow::before { content: '●'; color: #22d3ee; font-size: 0.5rem; }
.hero h1 { font-size: 2rem; font-weight: 700; color: #f8fafc; margin: 0 0 0.4rem; line-height: 1.2; }
.hero h1 span { color: #818cf8; }
.hero-sub { font-size: 0.9rem; color: #94a3b8; margin: 0; max-width: 520px; }
.hero-pills { display: flex; gap: 0.5rem; margin-top: 1.2rem; flex-wrap: wrap; }
.pill { background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3); color: #a5b4fc;
    font-size: 0.7rem; font-weight: 600; padding: 0.28rem 0.8rem; border-radius: 20px;
    font-family: 'DM Mono', monospace; letter-spacing: 0.05em; }
.pill.green { background: rgba(34,197,94,0.1); border-color: rgba(34,197,94,0.3); color: #86efac; }
.pill.cyan { background: rgba(34,211,238,0.1); border-color: rgba(34,211,238,0.3); color: #67e8f9; }

/* CARDS */
.glass-card { background: #1a1d2e; border: 1px solid #2d3152; border-radius: 16px;
    padding: 1.4rem 1.5rem; margin-bottom: 0.9rem; position: relative; }
.glass-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent); border-radius: 16px 16px 0 0; }
.card-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase;
    color: #475569; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
.card-label .dot { width: 5px; height: 5px; border-radius: 50%; background: #6366f1; display: inline-block; }

/* FORM INPUTS */
.stNumberInput > div > div > input {
    background: #0f1117 !important; border: 1.5px solid #2d3152 !important; border-radius: 10px !important;
    color: #e2e8f0 !important; font-size: 0.92rem !important; padding: 0.55rem 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stNumberInput > div > div > input:focus {
    border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important; outline: none !important;
}
.stNumberInput label { font-size: 0.78rem !important; font-weight: 600 !important; color: #94a3b8 !important; }
.stSelectbox > div > div { background: #0f1117 !important; border: 1.5px solid #2d3152 !important;
    border-radius: 10px !important; color: #e2e8f0 !important; }
.stSelectbox label { font-size: 0.78rem !important; font-weight: 600 !important; color: #94a3b8 !important; }

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important; color: #fff !important; border: none !important;
    border-radius: 12px !important; font-size: 0.95rem !important; font-weight: 700 !important;
    padding: 0.75rem 1.5rem !important; letter-spacing: 0.02em !important; transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(79,70,229,0.35) !important; width: 100% !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 28px rgba(79,70,229,0.5) !important; }
.sample-btn > button { background: #1a1d2e !important; border: 1px solid #2d3152 !important; color: #94a3b8 !important;
    font-size: 0.78rem !important; padding: 0.4rem 0.6rem !important; border-radius: 8px !important;
    box-shadow: none !important; font-weight: 500 !important; }
.sample-btn > button:hover { border-color: #6366f1 !important; color: #a5b4fc !important; transform: none !important; box-shadow: none !important; }

/* RESULT CARDS */
.result-churn { background: linear-gradient(135deg, #2d1419, #3d1a1a); border: 1px solid rgba(239,68,68,0.4);
    border-radius: 20px; padding: 2rem; text-align: center; position: relative; overflow: hidden; }
.result-churn::before { content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(239,68,68,0.12) 0%, transparent 60%); }
.result-churn .icon { font-size: 2.5rem; margin-bottom: 0.4rem; }
.result-churn .title { font-size: 1.6rem; font-weight: 800; color: #fca5a5; }
.result-churn .desc { font-size: 0.82rem; color: #f87171; opacity: 0.8; margin-top: 0.2rem; }
.result-ok { background: linear-gradient(135deg, #0d2419, #0f2d1e); border: 1px solid rgba(34,197,94,0.35);
    border-radius: 20px; padding: 2rem; text-align: center; position: relative; overflow: hidden; }
.result-ok::before { content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(34,197,94,0.1) 0%, transparent 60%); }
.result-ok .icon { font-size: 2.5rem; margin-bottom: 0.4rem; }
.result-ok .title { font-size: 1.6rem; font-weight: 800; color: #86efac; }
.result-ok .desc { font-size: 0.82rem; color: #4ade80; opacity: 0.8; margin-top: 0.2rem; }

/* STAT CHIPS */
.stat-chip { background: #1a1d2e; border: 1px solid #2d3152; border-radius: 12px; padding: 0.9rem 1rem; text-align: center; }
.stat-chip .s-label { font-size: 0.6rem; letter-spacing: 0.15em; text-transform: uppercase; color: #475569; font-weight: 600; }
.stat-chip .s-value { font-size: 1.5rem; font-weight: 700; margin-top: 0.1rem; font-family: 'DM Mono', monospace; }
.stat-chip .s-value.indigo { color: #818cf8; }
.stat-chip .s-value.coral { color: #f87171; }
.stat-chip .s-value.emerald { color: #4ade80; }
.stat-chip .s-value.amber { color: #fbbf24; }

/* RECOMMENDATIONS */
.rec-item { background: #1e2235; border-left: 3px solid #6366f1; border-radius: 0 10px 10px 0;
    padding: 0.7rem 1rem; margin-bottom: 0.5rem; font-size: 0.85rem; color: #cbd5e1;
    display: flex; align-items: flex-start; gap: 0.6rem; }
.rec-item .rec-icon { font-size: 1rem; flex-shrink: 0; }
.rec-item strong { color: #e2e8f0; }
.rec-item.warn { border-color: #f59e0b; }
.rec-item.danger { border-color: #ef4444; }
.rec-item.success { border-color: #22c55e; background: #0d2419; }
.rec-item.info { border-color: #38bdf8; }

/* SIDEBAR */
section[data-testid="stSidebar"] { background: #13151f !important; border-right: 1px solid #1e2235 !important; }
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }
.sidebar-logo { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #475569; letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 1.2rem; padding-bottom: 0.8rem; border-bottom: 1px solid #1e2235; }
.sidebar-logo span { color: #6366f1; font-size: 1rem; }
.score-ring { display: flex; flex-direction: column; align-items: center; padding: 1rem;
    background: #1a1d2e; border-radius: 14px; border: 1px solid #2d3152; margin-bottom: 1rem; }
.score-ring .score-num { font-size: 2.2rem; font-weight: 700; font-family: 'DM Mono', monospace; color: #818cf8; line-height: 1; }
.score-ring .score-label { font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.15em; color: #475569; margin-top: 0.2rem; font-weight: 600; }
.score-ring .score-sub { font-size: 0.72rem; color: #64748b; margin-top: 0.4rem; }
.feat-tag { display: inline-flex; align-items: center; gap: 0.3rem; background: #0f1117; border: 1px solid #2d3152;
    border-radius: 8px; padding: 0.25rem 0.6rem; font-size: 0.72rem; color: #94a3b8;
    font-family: 'DM Mono', monospace; margin: 0.18rem; }

/* TABS */
.stTabs [data-baseweb="tab-list"] { background: #1a1d2e !important; border-radius: 12px !important; padding: 4px !important;
    gap: 2px !important; border: 1px solid #2d3152 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border-radius: 9px !important; color: #64748b !important;
    font-weight: 600 !important; font-size: 0.82rem !important; padding: 0.5rem 1.2rem !important; transition: all 0.2s !important; }
.stTabs [aria-selected="true"] { background: #6366f1 !important; color: #fff !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

hr { border-color: #1e2235 !important; margin: 1.2rem 0 !important; }
.stAlert { border-radius: 10px !important; }
.stProgress > div > div > div { background: linear-gradient(90deg, #4f46e5, #7c3aed, #ec4899) !important; border-radius: 20px !important; }
.stProgress > div > div { background: #1e2235 !important; border-radius: 20px !important; height: 8px !important; }
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; }
.hint { font-size: 0.72rem; color: #475569; margin: -0.4rem 0 0.7rem; font-family: 'DM Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS (model, scaler, fitur, metadata) — semua harus konsisten
# ─────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")

@st.cache_resource
def load_artifacts():
    files = {
        'model'   : MODEL_DIR / 'best_model.pkl',
        'scaler'  : MODEL_DIR / 'scaler.pkl',
        'le'      : MODEL_DIR / 'label_encoders.pkl',
        'topfeat' : MODEL_DIR / 'top_features.pkl',
        'allfeat' : MODEL_DIR / 'all_features.pkl',
        'meta'    : MODEL_DIR / 'model_metadata.pkl',
    }
    arts, missing = {}, []
    for k, p in files.items():
        if p.exists():
            try:
                arts[k] = joblib.load(p)
            except Exception as e:
                arts[k] = None
        else:
            missing.append(p.name)
    arts['_missing'] = missing
    return arts

arts = load_artifacts()

if arts.get('model') is None or arts.get('scaler') is None or not arts.get('topfeat'):
    st.error("⚠️ File model belum lengkap / belum dibuat.")
    st.markdown("Jalankan terlebih dahulu:")
    st.code("python main.py", language="bash")
    if arts.get('_missing'):
        st.write("File yang hilang:", ", ".join(arts['_missing']))
    st.stop()

model     = arts['model']
scaler    = arts['scaler']
top_feat  = arts['topfeat']
le_map    = arts.get('le', {}) or {}
meta      = arts.get('meta', {}) or {}

# Validasi konsistensi jumlah fitur model vs scaler vs top_feat
n_model  = getattr(model, 'n_features_in_', len(top_feat))
n_scaler = getattr(scaler, 'n_features_in_', len(top_feat))
if len(top_feat) != n_scaler or len(top_feat) != n_model:
    st.error(
        f"⚠️ Artefak tidak sinkron — top_features memiliki {len(top_feat)} fitur, "
        f"namun scaler mengharapkan {n_scaler} dan model mengharapkan {n_model}. "
        "Jalankan ulang `python main.py` untuk membuat ulang artefak yang konsisten."
    )
    st.stop()

acc_val = meta.get('test_accuracy', 0.0)
f1_val  = meta.get('test_f1', 0.0)
model_name = meta.get('best_model_name', 'Model Terbaik')

# Konfigurasi tampilan tiap fitur (label, ikon, rentang wajar)
FIELD_META = {
    'satisfaction_score':         {'icon':'⭐','label':'Skor Kepuasan Pelanggan','min':1,'max':10,'default':7,'step':1,'help':'1 = sangat tidak puas, 10 = sangat puas','hint':'1 = sangat tidak puas · 10 = sangat puas'},
    'support_tickets':            {'icon':'🎫','label':'Jumlah Tiket Support','min':0,'max':20,'default':1,'step':1,'help':'Berapa kali menghubungi support','hint':'Semakin banyak → risiko lebih tinggi'},
    'avg_session_time':           {'icon':'⏱️','label':'Rata-rata Durasi Sesi (menit)','min':0.0,'max':60.0,'default':5.0,'step':0.5,'help':'Rata-rata waktu aktif per kunjungan','hint':'Durasi lebih lama = lebih engaged'},
    'last_3_month_purchase_freq': {'icon':'🛍️','label':'Frekuensi Beli 3 Bulan Terakhir','min':0,'max':30,'default':3,'step':1,'help':'Jumlah transaksi 3 bulan terakhir','hint':'0 = tidak pernah beli dalam 3 bulan'},
    'total_spent':                {'icon':'💰','label':'Total Pengeluaran ($)','min':0.0,'max':10000.0,'default':500.0,'step':50.0,'help':'Akumulasi total belanja','hint':'Total historis sejak bergabung'},
    'is_premium_user':            {'icon':'👑','label':'Status Premium (0/1)','min':0,'max':1,'default':0,'step':1,'help':'0 = reguler, 1 = premium','hint':'0 = reguler · 1 = premium'},
    'delivery_delay_days':        {'icon':'📦','label':'Rata-rata Keterlambatan Kirim (hari)','min':0,'max':30,'default':2,'step':1,'help':'Rata-rata hari keterlambatan','hint':'Tinggi = pengalaman buruk'},
    'total_visits':                {'icon':'👣','label':'Total Kunjungan','min':0,'max':500,'default':50,'step':1,'help':'Total kunjungan ke platform','hint':'Lebih banyak = lebih aktif'},
    'age':                         {'icon':'🎂','label':'Usia','min':18,'max':80,'default':35,'step':1,'help':'Usia pelanggan','hint':''},
    'nps_score':                   {'icon':'📈','label':'NPS Score','min':-100,'max':100,'default':0,'step':1,'help':'Net Promoter Score','hint':'-100 sangat negatif · 100 sangat positif'},
    'email_open_rate':             {'icon':'📧','label':'Email Open Rate','min':0.0,'max':1.0,'default':0.3,'step':0.01,'help':'Proporsi email yang dibuka','hint':'0 = tidak pernah · 1 = selalu dibuka'},
    'email_click_rate':            {'icon':'🖱️','label':'Email Click Rate','min':0.0,'max':1.0,'default':0.1,'step':0.01,'help':'Proporsi email yang diklik','hint':''},
    'discount_used':               {'icon':'🏷️','label':'Diskon Dipakai','min':0,'max':50,'default':3,'step':1,'help':'Jumlah diskon yang pernah digunakan','hint':''},
    'refund_requested':            {'icon':'↩️','label':'Permintaan Refund','min':0,'max':10,'default':0,'step':1,'help':'Jumlah permintaan refund','hint':''},
    'marketing_spend_per_user':    {'icon':'📣','label':'Marketing Spend per User ($)','min':0.0,'max':500.0,'default':30.0,'step':5.0,'help':'Biaya marketing per pelanggan','hint':''},
    'lifetime_value':              {'icon':'💎','label':'Lifetime Value ($)','min':0.0,'max':20000.0,'default':1500.0,'step':50.0,'help':'Estimasi nilai total pelanggan','hint':''},
    'avg_order_value':             {'icon':'🧾','label':'Rata-rata Nilai Order ($)','min':0.0,'max':2000.0,'default':100.0,'step':5.0,'help':'Nilai rata-rata per transaksi','hint':''},
    'pages_per_session':           {'icon':'📄','label':'Halaman per Sesi','min':1,'max':50,'default':5,'step':1,'help':'Jumlah halaman dilihat per sesi','hint':''},
}
CAT_OPTIONS = {
    'gender': ['Male','Female','Other'],
    'subscription_type': ['Basic','Standard','Premium'],
    'country': ['USA','UK','Germany','France','India'],
    'city': ['New York','London','Berlin','Paris','Mumbai'],
    'acquisition_channel': ['Organic','Paid','Referral','Social','Email'],
    'device_type': ['Mobile','Desktop','Tablet'],
    'payment_method': ['Credit Card','PayPal','Debit Card','Bank Transfer'],
}

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo"><span>📡</span> ChurnPredictor</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="score-ring">
        <div class="score-num">{acc_val:.0%}</div>
        <div class="score-label">Akurasi Model</div>
        <div class="score-sub">F1 {f1_val:.3f} · {model_name}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.65rem;color:#475569;letter-spacing:0.12em;text-transform:uppercase;font-weight:700;margin:1rem 0 0.5rem">Fitur Aktif</p>', unsafe_allow_html=True)
    feat_html = ''.join([f'<span class="feat-tag">◆ {f.replace("_"," ")}</span>' for f in top_feat])
    st.markdown(feat_html, unsafe_allow_html=True)

    st.divider()
    st.markdown('<p style="font-size:0.65rem;color:#475569;letter-spacing:0.12em;text-transform:uppercase;font-weight:700;margin:0 0 0.5rem">Panduan Cepat</p>', unsafe_allow_html=True)
    for step, txt in [("1","Isi form di bawah"), ("2","Klik Prediksi"), ("3","Lihat hasil & saran")]:
        st.markdown(f'<p style="font-size:0.8rem;color:#64748b;margin:0.2rem 0"><span style="color:#6366f1;font-family:DM Mono;font-weight:700">{step}.</span> {txt}</p>', unsafe_allow_html=True)

    if st.session_state.get('prediction_log'):
        log_df = pd.DataFrame(st.session_state.prediction_log)
        total, n_churn = len(log_df), (log_df['prediction']==1).sum()
        st.divider()
        st.markdown('<p style="font-size:0.65rem;color:#475569;letter-spacing:0.12em;text-transform:uppercase;font-weight:700;margin:0 0 0.5rem">Sesi Ini</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem">
            <div class="stat-chip"><div class="s-label">Total</div><div class="s-value indigo">{total}</div></div>
            <div class="stat-chip"><div class="s-label">⚠️ Churn</div><div class="s-value coral">{n_churn}</div></div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Machine Learning · Customer Intelligence</div>
    <h1>Prediksi <span>Customer Churn</span><br>Sebelum Terlambat</h1>
    <p class="hero-sub">Masukkan data pelanggan, dan model AI akan menghitung risiko churn beserta langkah retensi yang tepat.</p>
    <div class="hero-pills">
        <span class="pill">📡 {model_name}</span>
        <span class="pill green">✓ Akurasi {acc_val:.1%}</span>
        <span class="pill cyan">⚡ {len(top_feat)} fitur</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮  Prediksi Tunggal", "📂  Prediksi Batch", "📊  Riwayat & Analisis"])

def get_field_cfg(name):
    """Ambil konfigurasi tampilan untuk fitur, dengan fallback default jika tidak terdaftar."""
    if name in FIELD_META:
        return FIELD_META[name]
    return {'icon':'🔹','label':name.replace('_',' ').title(),'min':0.0,'max':1000.0,
            'default':0.0,'step':1.0,'help':'','hint':''}

# ═══════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI TUNGGAL
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p style="font-size:0.75rem;color:#475569;margin-bottom:0.4rem;font-weight:600">Isi cepat:</p>', unsafe_allow_html=True)
    cq1, cq2, cq3, cq4 = st.columns(4)

    with cq1:
        st.markdown('<div class="sample-btn">', unsafe_allow_html=True)
        if st.button("🎲 Acak", use_container_width=True):
            st.session_state['_samp'] = {f: round(random.uniform(get_field_cfg(f)['min'], get_field_cfg(f)['max']),2)
                                         if isinstance(get_field_cfg(f)['default'], float)
                                         else random.randint(int(get_field_cfg(f)['min']), int(get_field_cfg(f)['max']))
                                         for f in top_feat if f not in le_map}
        st.markdown('</div>', unsafe_allow_html=True)
    with cq2:
        st.markdown('<div class="sample-btn">', unsafe_allow_html=True)
        if st.button("⚠️ Contoh Churn", use_container_width=True):
            churn_vals = {'satisfaction_score':2,'support_tickets':8,'avg_session_time':1.2,
                          'last_3_month_purchase_freq':0,'total_spent':50.0,'is_premium_user':0,
                          'delivery_delay_days':12,'total_visits':5,'nps_score':-60,
                          'email_open_rate':0.05,'discount_used':0,'refund_requested':3}
            st.session_state['_samp'] = {f: churn_vals.get(f, get_field_cfg(f)['default']) for f in top_feat if f not in le_map}
        st.markdown('</div>', unsafe_allow_html=True)
    with cq3:
        st.markdown('<div class="sample-btn">', unsafe_allow_html=True)
        if st.button("✅ Contoh Loyal", use_container_width=True):
            loyal_vals = {'satisfaction_score':9,'support_tickets':0,'avg_session_time':12.5,
                          'last_3_month_purchase_freq':12,'total_spent':2500.0,'is_premium_user':1,
                          'delivery_delay_days':1,'total_visits':180,'nps_score':80,
                          'email_open_rate':0.85,'discount_used':10,'refund_requested':0}
            st.session_state['_samp'] = {f: loyal_vals.get(f, get_field_cfg(f)['default']) for f in top_feat if f not in le_map}
        st.markdown('</div>', unsafe_allow_html=True)
    with cq4:
        st.markdown('<div class="sample-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.pop('_samp', None); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    samp = st.session_state.get('_samp', {})
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # Pisahkan fitur numerik & kategorikal yang ada di top_feat
    num_feats = [f for f in top_feat if f not in le_map]
    cat_feats = [f for f in top_feat if f in le_map]
    half = max(1, (len(num_feats)+1)//2)

    left, right = st.columns(2, gap="medium")
    user_input = {}

    with left:
        st.markdown('<div class="glass-card"><div class="card-label"><span class="dot"></span>Data Utama</div>', unsafe_allow_html=True)
        for f in num_feats[:half]:
            cfg = get_field_cfg(f)
            is_float = isinstance(cfg['default'], float)
            val = samp.get(f, cfg['default'])
            if is_float:
                user_input[f] = st.number_input(f"{cfg['icon']} {cfg['label']}",
                    min_value=float(cfg['min']), max_value=float(cfg['max']),
                    value=float(val), step=float(cfg['step']), help=cfg['help'], key=f"in_{f}")
            else:
                user_input[f] = st.number_input(f"{cfg['icon']} {cfg['label']}",
                    min_value=int(cfg['min']), max_value=int(cfg['max']),
                    value=int(val), step=int(cfg['step']), help=cfg['help'], key=f"in_{f}")
            if cfg['hint']:
                st.markdown(f'<p class="hint">{cfg["hint"]}</p>', unsafe_allow_html=True)
        for f in cat_feats[:max(0, len(cat_feats)//2)]:
            opts = CAT_OPTIONS.get(f, list(getattr(le_map.get(f), 'classes_', ['-'])))
            user_input[f] = st.selectbox(f"🏷️ {f.replace('_',' ').title()}", opts, key=f"in_{f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card"><div class="card-label"><span class="dot"></span>Data Tambahan</div>', unsafe_allow_html=True)
        for f in num_feats[half:]:
            cfg = get_field_cfg(f)
            is_float = isinstance(cfg['default'], float)
            val = samp.get(f, cfg['default'])
            if is_float:
                user_input[f] = st.number_input(f"{cfg['icon']} {cfg['label']}",
                    min_value=float(cfg['min']), max_value=float(cfg['max']),
                    value=float(val), step=float(cfg['step']), help=cfg['help'], key=f"in_{f}")
            else:
                user_input[f] = st.number_input(f"{cfg['icon']} {cfg['label']}",
                    min_value=int(cfg['min']), max_value=int(cfg['max']),
                    value=int(val), step=int(cfg['step']), help=cfg['help'], key=f"in_{f}")
            if cfg['hint']:
                st.markdown(f'<p class="hint">{cfg["hint"]}</p>', unsafe_allow_html=True)
        for f in cat_feats[max(0, len(cat_feats)//2):]:
            opts = CAT_OPTIONS.get(f, list(getattr(le_map.get(f), 'classes_', ['-'])))
            user_input[f] = st.selectbox(f"🏷️ {f.replace('_',' ').title()}", opts, key=f"in_{f}")

        # quick risk preview (heuristik ringan, bukan model asli)
        s = user_input.get('satisfaction_score', 7); t = user_input.get('support_tickets', 0)
        fr = user_input.get('last_3_month_purchase_freq', 3); dl = user_input.get('delivery_delay_days', 0)
        pr = user_input.get('is_premium_user', 0)
        raw_risk = max(0, min(100, (10-s)*7 + t*4 - fr*3 + dl*2.5 - pr*8))
        risk_color = "#ef4444" if raw_risk > 60 else "#f59e0b" if raw_risk > 35 else "#22c55e"
        st.markdown(f"""
        <div style="background:#0f1117;border:1px solid #2d3152;border-radius:12px;padding:1rem;margin-top:0.5rem">
            <p style="font-size:0.6rem;color:#475569;letter-spacing:0.12em;text-transform:uppercase;font-weight:700;margin:0 0 0.4rem">Estimasi Risiko Cepat</p>
            <p style="font-size:1.6rem;font-weight:700;font-family:'DM Mono';color:{risk_color};margin:0;line-height:1">{int(raw_risk)}<span style="font-size:0.8rem;opacity:0.6">/100</span></p>
            <p style="font-size:0.72rem;color:#475569;margin:0.2rem 0 0">{"⚠️ Risiko Tinggi" if raw_risk>60 else "⚡ Risiko Sedang" if raw_risk>35 else "✅ Risiko Rendah"} (indikator awal)</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1,2,1])
    with btn_col:
        do_predict = st.button("📡  Analisis & Prediksi Churn", use_container_width=True)

    if do_predict:
        try:
            row = {}
            for f, v in user_input.items():
                if f in le_map:
                    try: row[f] = le_map[f].transform([str(v)])[0]
                    except Exception: row[f] = 0
                else:
                    row[f] = v
            ordered = [row.get(f, 0) for f in top_feat]
            X_in = np.array([ordered], dtype=float)
            X_in_s = scaler.transform(X_in)

            pred = model.predict(X_in_s)[0]
            proba = model.predict_proba(X_in_s)[0] if hasattr(model,'predict_proba') else [1-pred, pred]
            pc, pn = round(proba[1]*100,1), round(proba[0]*100,1)

            if 'prediction_log' not in st.session_state: st.session_state.prediction_log = []
            st.session_state.prediction_log.append({'Waktu': datetime.now().strftime('%H:%M:%S'),
                **user_input, 'prediction': int(pred), 'prob_churn': pc})

            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            st.divider()
            st.markdown("### Hasil Analisis")
            res_col, gauge_col = st.columns([1,1], gap="medium")

            with res_col:
                if pred == 1:
                    st.markdown('<div class="result-churn"><div class="icon">⚠️</div><div class="title">BERPOTENSI CHURN</div><div class="desc">Pelanggan ini kemungkinan akan meninggalkan layanan</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-ok"><div class="icon">✅</div><div class="title">TIDAK CHURN</div><div class="desc">Pelanggan kemungkinan akan tetap bertahan</div></div>', unsafe_allow_html=True)
                    if user_input.get('satisfaction_score', 0) >= 9: st.balloons()

                st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem">
                    <div class="stat-chip"><div class="s-label">🔴 Prob. Churn</div><div class="s-value coral">{pc}%</div></div>
                    <div class="stat-chip"><div class="s-label">🟢 Prob. Aman</div><div class="s-value emerald">{pn}%</div></div>
                </div>
                """, unsafe_allow_html=True)

            with gauge_col:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=pc,
                    number={'suffix':'%','font':{'size':36,'color':'#e2e8f0','family':'DM Mono'}},
                    title={'text':"Risiko Churn",'font':{'size':13,'color':'#94a3b8'}},
                    gauge={'axis':{'range':[0,100],'tickcolor':'#475569','tickfont':{'size':10,'color':'#475569'},'nticks':6},
                           'bar':{'color':'#ef4444' if pc>60 else '#f59e0b' if pc>35 else '#22c55e','thickness':0.28},
                           'bgcolor':'rgba(0,0,0,0)','bordercolor':'rgba(0,0,0,0)',
                           'steps':[{'range':[0,35],'color':'rgba(34,197,94,0.1)'},
                                    {'range':[35,65],'color':'rgba(245,158,11,0.1)'},
                                    {'range':[65,100],'color':'rgba(239,68,68,0.1)'}],
                           'threshold':{'line':{'color':'#818cf8','width':2},'thickness':0.75,'value':pc}}))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=30,b=10,l=20,r=20), height=220, font={'family':'DM Sans'})
                st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
                st.progress(int(pc), text=f"Level risiko: {int(pc)}%")

            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            st.markdown("### 💡 Rekomendasi Tindakan")
            recs = []
            sat = user_input.get('satisfaction_score', 7); tix = user_input.get('support_tickets', 0)
            freq = user_input.get('last_3_month_purchase_freq', 3); delay = user_input.get('delivery_delay_days', 0)
            prem = user_input.get('is_premium_user', 0)
            if pred == 1:
                if sat <= 4: recs.append(('danger','📞','<strong>Hubungi langsung</strong> — Skor kepuasan sangat rendah. Lakukan follow-up personal segera.'))
                elif sat <= 6: recs.append(('warn','💬','<strong>Survei kepuasan</strong> — Kirim email survei dan tawarkan solusi.'))
                if tix >= 5: recs.append(('danger','🔧',f'<strong>Selesaikan {tix} tiket</strong> — Banyak keluhan belum terselesaikan.'))
                if freq == 0: recs.append(('danger','🛍️','<strong>Reaktivasi pelanggan</strong> — Nol pembelian 3 bulan terakhir.'))
                elif freq < 3: recs.append(('warn','🎁','<strong>Program loyalitas</strong> — Frekuensi beli rendah.'))
                if delay >= 7: recs.append(('warn','🚚',f'<strong>Perbaiki logistik</strong> — Rata-rata {delay} hari keterlambatan.'))
                if prem == 0: recs.append(('info','👑','<strong>Tawarkan upgrade Premium</strong> — Cenderung mengurangi churn.'))
                if not recs: recs.append(('warn','🔄','<strong>Jalankan program retensi</strong> — Tingkatkan engagement.'))
            else:
                recs.append(('success','✅','<strong>Pertahankan kualitas</strong> — Pelanggan dalam kondisi sehat.'))
                if prem == 0: recs.append(('info','👑','<strong>Peluang upsell</strong> — Tawarkan upgrade ke Premium.'))
                if sat >= 9: recs.append(('success','⭐','<strong>Minta testimoni</strong> — Pelanggan sangat puas.'))
            for cls, ico, txt in recs:
                st.markdown(f'<div class="rec-item {cls}"><span class="rec-icon">{ico}</span><span>{txt}</span></div>', unsafe_allow_html=True)

            with st.expander("🔎 Lihat detail input"):
                disp = pd.DataFrame([{'Fitur': k.replace('_',' ').title(), 'Nilai': v} for k,v in user_input.items()])
                st.dataframe(disp, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Terjadi error: {e}")
            st.exception(e)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-label"><span class="dot"></span>Prediksi Banyak Pelanggan Sekaligus</div>
        <p style="font-size:0.85rem;color:#64748b;margin:0">
            Upload CSV berisi kolom: <code>{', '.join(top_feat)}</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

    template = pd.DataFrame([{f: get_field_cfg(f)['default'] for f in top_feat} for _ in range(3)])
    for i, f in enumerate(top_feat):
        cfg = get_field_cfg(f)
        lo, hi = cfg['min'], cfg['max']
        template[f] = [lo, hi, cfg['default']]

    dl_col, _ = st.columns([1,2])
    with dl_col:
        st.download_button("📥 Download Template CSV", data=template.to_csv(index=False),
            file_name="template_churn.csv", mime="text/csv", use_container_width=True)

    uploaded = st.file_uploader("Upload file CSV", type=['csv'], label_visibility='collapsed')
    if uploaded:
        try:
            bdf = pd.read_csv(uploaded)
            miss_c = [c for c in top_feat if c not in bdf.columns]
            if miss_c:
                st.error(f"❌ Kolom tidak lengkap: `{'`, `'.join(miss_c)}`")
                st.stop()
            st.success(f"✅ {len(bdf)} baris siap diproses")
            if st.button("🚀 Prediksi Semua Data", use_container_width=True):
                with st.spinner(f"Memproses {len(bdf)} pelanggan..."):
                    preds, probs = [], []
                    for _, row in bdf.iterrows():
                        Xi = scaler.transform(np.array([[row.get(f,0) for f in top_feat]], dtype=float))
                        preds.append(int(model.predict(Xi)[0]))
                        probs.append(round(model.predict_proba(Xi)[0][1]*100,1))
                    bdf['Prediksi'] = ['⚠️ CHURN' if p==1 else '✅ AMAN' for p in preds]
                    bdf['Prob Churn'] = probs

                n_churn = sum(preds)
                st.markdown(f"""
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.6rem;margin:1rem 0">
                    <div class="stat-chip"><div class="s-label">Total</div><div class="s-value indigo">{len(bdf)}</div></div>
                    <div class="stat-chip"><div class="s-label">⚠️ Churn</div><div class="s-value coral">{n_churn}</div></div>
                    <div class="stat-chip"><div class="s-label">✅ Aman</div><div class="s-value emerald">{len(bdf)-n_churn}</div></div>
                </div>
                """, unsafe_allow_html=True)

                cr1, cr2 = st.columns(2)
                with cr1:
                    fig_pie = px.pie(names=['⚠️ Churn','✅ Aman'], values=[n_churn, len(bdf)-n_churn],
                        color_discrete_sequence=['#ef4444','#22c55e'], hole=0.45)
                    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        legend_font_color='#94a3b8', font_color='#94a3b8', margin=dict(t=20,b=10,l=10,r=10), height=240)
                    st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
                with cr2:
                    fig_hist = px.histogram(bdf, x='Prob Churn', nbins=15, color_discrete_sequence=['#6366f1'])
                    fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        xaxis_color='#475569', yaxis_color='#475569', font_color='#94a3b8',
                        margin=dict(t=20,b=10,l=10,r=10), height=240,
                        xaxis_title='Probabilitas Churn (%)', yaxis_title='Jumlah')
                    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

                st.dataframe(bdf, use_container_width=True)
                st.download_button("📥 Download Hasil", data=bdf.to_csv(index=False),
                    file_name=f"hasil_batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"❌ {e}")

# ═══════════════════════════════════════════════════════════════
# TAB 3 — RIWAYAT & ANALISIS
# ═══════════════════════════════════════════════════════════════
with tab3:
    log = st.session_state.get('prediction_log', [])
    if not log:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:2.5rem">
            <div style="font-size:3rem;margin-bottom:0.5rem">📡</div>
            <p style="color:#475569;font-size:0.9rem">Belum ada riwayat prediksi.<br>Lakukan prediksi di tab <strong style="color:#818cf8">Prediksi Tunggal</strong> terlebih dahulu.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        log_df = pd.DataFrame(log)
        n, n_c = len(log_df), (log_df['prediction']==1).sum()
        avg_p = log_df['prob_churn'].mean()
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.6rem;margin-bottom:1rem">
            <div class="stat-chip"><div class="s-label">Total Prediksi</div><div class="s-value indigo">{n}</div></div>
            <div class="stat-chip"><div class="s-label">⚠️ Churn</div><div class="s-value coral">{n_c}</div></div>
            <div class="stat-chip"><div class="s-label">✅ Aman</div><div class="s-value emerald">{n-n_c}</div></div>
            <div class="stat-chip"><div class="s-label">Avg Risk</div><div class="s-value amber">{avg_p:.1f}%</div></div>
        </div>
        """, unsafe_allow_html=True)

        hc1, hc2 = st.columns(2)
        with hc1:
            fig_h1 = px.pie(names=['⚠️ Churn','✅ Aman'], values=[n_c, n-n_c],
                color_discrete_sequence=['#ef4444','#22c55e'], hole=0.5, title='Distribusi Hasil')
            fig_h1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8', title_font_color='#e2e8f0', margin=dict(t=40,b=10,l=10,r=10), height=260)
            st.plotly_chart(fig_h1, use_container_width=True, config={'displayModeBar': False})
        with hc2:
            fig_h2 = px.histogram(log_df, x='prob_churn', nbins=10, color_discrete_sequence=['#818cf8'], title='Distribusi Skor Risiko')
            fig_h2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8', title_font_color='#e2e8f0', xaxis_color='#475569', yaxis_color='#475569',
                margin=dict(t=40,b=10,l=10,r=10), height=260, xaxis_title='Prob. Churn (%)', yaxis_title='Jumlah')
            st.plotly_chart(fig_h2, use_container_width=True, config={'displayModeBar': False})

        with st.expander("📋 Lihat semua riwayat"):
            st.dataframe(log_df, use_container_width=True)
            st.download_button("📥 Export Riwayat", data=log_df.to_csv(index=False),
                file_name=f"riwayat_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True)

# FOOTER
st.markdown("""
<div style="text-align:center;padding:2rem 0 0.5rem;border-top:1px solid #1e2235;margin-top:2rem">
    <p style="font-size:0.65rem;color:#334155;font-family:'DM Mono';letter-spacing:0.1em;margin:0">
        CHURNPREDICTOR · UAS DATA SCIENCE · MACHINE LEARNING PIPELINE
    </p>
</div>
""", unsafe_allow_html=True)
