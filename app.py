import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Set page config for a premium wide layout
st.set_page_config(
    page_title="Gaming vs Academics: Predictor & Insights",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling (dark/neon aesthetic)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, .title-text {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
    }
    
    /* Custom Neon Gradient Title */
    .hero-title {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }
    
    .hero-subtitle {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-top: 0px;
        margin-bottom: 30px;
    }
    
    /* Card Container styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .metric-value-huge {
        font-size: 2.8rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        line-height: 1.1;
    }
    
    .accent-cyan {
        color: #00f2fe;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.3);
    }
    
    .accent-purple {
        color: #b5179e;
        text-shadow: 0 0 10px rgba(181, 23, 158, 0.3);
    }
    
    .accent-green {
        color: #4cc9f0;
        text-shadow: 0 0 10px rgba(76, 201, 240, 0.3);
    }
    
    /* Insight tags */
    .insight-tag {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .tag-low { background-color: rgba(76, 201, 240, 0.15); color: #4cc9f0; border: 1px solid rgba(76, 201, 240, 0.3); }
    .tag-medium { background-color: rgba(247, 127, 0, 0.15); color: #f77f00; border: 1px solid rgba(247, 127, 0, 0.3); }
    .tag-high { background-color: rgba(217, 4, 41, 0.15); color: #d90429; border: 1px solid rgba(217, 4, 41, 0.3); }
    
    /* Styled HR */
    hr {
        border-color: rgba(255, 255, 255, 0.08);
        margin: 25px 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load model and data
@st.cache_resource
def load_ml_model():
    if os.path.exists("academic_predictor_model.joblib"):
        return joblib.load("academic_predictor_model.joblib")
    return None

@st.cache_data
def load_stats_and_aggregates():
    stats = {}
    aggregates = {}
    
    if os.path.exists("dataset_stats.json"):
        with open("dataset_stats.json", "r") as f:
            stats = json.load(f)
            
    if os.path.exists("ui_aggregates.json"):
        with open("ui_aggregates.json", "r") as f:
            aggregates = json.load(f)
            
    return stats, aggregates

# Load resources
model = load_ml_model()
stats, aggregates = load_stats_and_aggregates()

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h2 style="font-family:'Outfit',sans-serif; margin-bottom:0; color:#00f2fe;">🎮 Parameter Input</h2>
    <p style="color:#a0aec0; font-size:0.9rem;">Sesuaikan profil & kebiasaan Anda</p>
</div>
""", unsafe_allow_html=True)

# A. Profil Demografis
st.sidebar.markdown("### 👤 Profil Demografis")
gender = st.sidebar.selectbox("Jenis Kelamin", options=["Male", "Female", "Other"])
age = st.sidebar.slider("Usia", min_value=16, max_value=24, value=20)

st.sidebar.markdown("---")

# B. Parameter Studi & Sekolah
st.sidebar.markdown("### 📚 Studi & Akademis")
study_hours = st.sidebar.slider(
    "Jam Belajar Harian",
    min_value=1.0,
    max_value=10.0,
    value=5.5,
    step=0.5,
    help="Rata-rata jam yang Anda habiskan untuk belajar per hari."
)
attendance = st.sidebar.slider(
    "Tingkat Kehadiran (%)",
    min_value=60.0,
    max_value=100.0,
    value=85.0,
    step=1.0,
    help="Persentase kehadiran Anda di kelas/sekolah."
)
stress_level = st.sidebar.selectbox(
    "Tingkat Stres",
    options=["Low", "Medium", "High"],
    index=1,
    help="Tingkat stres psikologis yang Anda rasakan sehari-hari."
)

st.sidebar.markdown("---")

# C. Parameter Gaming
st.sidebar.markdown("### 🕹️ Aktivitas Gaming")
gaming_hours = st.sidebar.slider(
    "Jam Gaming Harian",
    min_value=0.0,
    max_value=8.0,
    value=3.0,
    step=0.5,
    help="Rata-rata waktu bermain game Anda per hari."
)
gaming_genre = st.sidebar.selectbox(
    "Genre Game Terpopuler", 
    options=["FPS", "RPG", "Casual"],
    help="Jenis game yang paling sering Anda mainkan."
)
addiction_score = st.sidebar.slider(
    "Skor Adiksi Game",
    min_value=0.0,
    max_value=23.0,
    value=10.0,
    step=0.5,
    help="Skor tingkat ketergantungan/kecanduan game Anda (0 = Rendah, 23 = Ekstrem)."
)

st.sidebar.markdown("---")

# D. Kesehatan & Gaya Hidup
st.sidebar.markdown("### 💤 Gaya Hidup")
sleep_hours = st.sidebar.slider(
    "Jam Tidur per Malam",
    min_value=4.0,
    max_value=9.0,
    value=6.5,
    step=0.5,
    help="Rata-rata durasi tidur malam Anda."
)
device_usage = st.sidebar.slider(
    "Total Durasi Gadget (Jam)",
    min_value=float(max(gaming_hours + study_hours, 1.1)),
    max_value=14.0,
    value=float(max(gaming_hours + study_hours + 2.0, 7.5)),
    step=0.5,
    help="Total jam menatap layar gadget per hari."
)
social_activity = st.sidebar.slider(
    "Skor Aktivitas Sosial",
    min_value=0.0,
    max_value=5.0,
    value=2.5,
    step=0.1,
    help="Skor keterlibatan Anda dalam aktivitas sosial di luar belajar dan gaming."
)
reaction_time_ms = st.sidebar.slider(
    "Waktu Reaksi (Milidetik)",
    min_value=180,
    max_value=350,
    value=260,
    step=5,
    help="Kecepatan respons refleks motorik Anda (semakin rendah semakin cepat)."
)


# ----------------- MAIN AREA -----------------

# Header Section
st.markdown('<h1 class="hero-title">🎮 Gaming vs Academic Performance</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Predictive Analysis & Interactive Insights Dashboard (Model Akurasi Tinggi - R²: 92%)</p>', unsafe_allow_html=True)

# 1. Prediction computation
input_data = pd.DataFrame([{
    'age': age,
    'gaming_hours': gaming_hours,
    'study_hours': study_hours,
    'sleep_hours': sleep_hours,
    'attendance': attendance,
    'social_activity': social_activity,
    'device_usage': device_usage,
    'reaction_time_ms': reaction_time_ms,
    'addiction_score': addiction_score,
    'gender': gender,
    'gaming_genre': gaming_genre,
    'stress_level': stress_level
}])

# Reorder columns to match training order exactly
cols_order = [
    'age', 'gaming_hours', 'study_hours', 'sleep_hours', 
    'attendance', 'social_activity', 'device_usage', 
    'reaction_time_ms', 'addiction_score', 'gender', 'gaming_genre', 'stress_level'
]
input_data = input_data[cols_order]

# Predict
global_mean = stats.get('grades', {}).get('mean', 66.18)
if model is not None:
    prediction = model.predict(input_data)[0]
else:
    prediction = global_mean

diff_to_mean = prediction - global_mean

# Determine Impact & Risk Level
if prediction < 55.0 or (gaming_hours >= 5.5 and study_hours <= 3.5) or addiction_score >= 15.0:
    impact_level = "Tinggi (High Risk)"
    impact_class = "tag-high"
    impact_desc = "Model mendeteksi **risiko akademis yang tinggi**. Jam bermain game yang panjang, rendahnya waktu belajar, dan/atau adiksi game yang tinggi secara signifikan memangkas nilai akademis terprediksi Anda."
elif prediction < 75.0 or gaming_hours >= 3.5 or addiction_score >= 9.0:
    impact_level = "Sedang (Moderate)"
    impact_class = "tag-medium"
    impact_desc = "Kebiasaan bermain game Anda berada pada zona sedang. Performa akademis Anda cukup stabil, namun ada ruang besar untuk peningkatan jika Anda mengalokasikan lebih banyak jam belajar."
else:
    impact_level = "Rendah (Balanced / High Performance)"
    impact_class = "tag-low"
    impact_desc = "Luar biasa! Model memprediksi **performa akademik yang sangat baik**. Anda berhasil menyeimbangkan jam belajar yang tinggi dengan jam bermain game yang terkendali."

# Show positive alert highlighting high accuracy dataset
# st.success("""
# 🎉 **Model Machine Learning Akurat Aktif:** Kami telah memperbarui model menggunakan dataset terbaru **Gaming_Academic_Performance.csv**. 
# Model ini memiliki **R² Score sebesar 92.1%**, yang berarti prediksinya sangat sensitif dan logis terhadap kebiasaan gaming, jam belajar, tidur, dan adiksi Anda!
# """)

# KPI Cards row
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Prediksi Nilai Akademis</div>
        <div class="metric-value-huge accent-cyan">{prediction:.2f} <span style="font-size: 1.5rem; color:#a0aec0;">/100</span></div>
        <div style="color: {'#4cc9f0' if diff_to_mean >= 0 else '#d90429'}; font-size: 0.9rem; margin-top: 8px; font-weight:600;">
            {'+' if diff_to_mean >= 0 else ''}{diff_to_mean:.2f} poin dari rata-rata global
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Tingkat Risiko Akademis</div>
        <div class="metric-value-huge accent-purple" style="font-size: 2.2rem; padding-top: 8px; font-weight:700;">{impact_level.split(" ")[0]}</div>
        <div class="insight-tag {impact_class}">{impact_level}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rata-Rata Nilai Global</div>
        <div class="metric-value-huge" style="color: #cbd5e0;">{global_mean:.2f} <span style="font-size: 1.5rem; color:#a0aec0;">/100</span></div>
        <div style="color: #a0aec0; font-size: 0.9rem; margin-top: 8px;">
            Sampel Data: 8.000 Siswa
        </div>
    </div>
    """, unsafe_allow_html=True)

# Tabs Navigation
tab1, tab2, tab3 = st.tabs([
    "🎯 Prediksi & Rekomendasi Pintar", 
    "📊 Eksplorasi Hubungan Variabel", 
    "⚙️ Detail Data & Performa Model"
])

# ----------------- TAB 1: PREDIKSI & REKOMENDASI -----------------
with tab1:
    st.markdown("### 💡 Rekomendasi Berdasarkan Profil Anda")
    
    # Layout with sub-columns
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border-left: 4px solid #00f2fe; padding: 20px; border-radius: 0 12px 12px 0; margin-bottom: 25px;">
            <h4 style="margin: 0 0 10px 0; font-family:'Outfit',sans-serif;">Analisis Dampak Gaming & Belajar:</h4>
            <p style="margin: 0; color:#cbd5e0; line-height: 1.6;">{impact_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations checklist
        st.markdown("#### ✔️ Langkah Optimasi Gaya Hidup & Akademis:")
        
        recs = []
        
        # Gaming hours vs Study Hours ratio
        if gaming_hours > study_hours:
            recs.append("🔴 **Koreksi Rasio Waktu:** Anda menghabiskan lebih banyak waktu bermain game ({:.1f} jam) daripada belajar ({:.1f} jam). Hubungan negatif terkuat pada nilai Anda berasal dari tingginya waktu gaming harian. Disarankan untuk membalik rasio ini.".format(gaming_hours, study_hours))
        
        if gaming_hours >= 5.0:
            recs.append("🔴 **Kurangi Jam Bermain:** Bermain game harian selama {:.1f} jam secara statistik memotong nilai akademik rata-rata hingga di bawah 50 poin.".format(gaming_hours))
            
        if study_hours < 4.0:
            recs.append("🟡 **Tingkatkan Jam Belajar:** Waktu belajar Anda ({:.1f} jam) di bawah rata-rata. Menambah 1-2 jam belajar per hari akan menaikkan prediksi nilai Anda secara signifikan karena study_hours memiliki korelasi positif terbesar (+0.73).".format(study_hours))
            
        if sleep_hours < 6.0:
            recs.append("🔴 **Tingkatkan Waktu Tidur:** Tidur hanya {:.1f} jam menghambat pemulihan kognitif. Siswa dengan tidur 7-8.5 jam memiliki nilai rata-rata jauh lebih tinggi.".format(sleep_hours))
            
        if addiction_score >= 12.0:
            recs.append("🔴 **Waspadai Adiksi Game:** Skor adiksi Anda ({:.1f}/23) tinggi. Hal ini berkorelasi negatif kuat (-0.49) dengan penurunan nilai karena mengganggu disiplin belajar.".format(addiction_score))
            
        if attendance < 80.0:
            recs.append("🟡 **Tingkatkan Kehadiran:** Kehadiran Anda ({:.1f}%) di bawah rata-rata global (79.9%). Meningkatkan kehadiran di kelas secara langsung meningkatkan nilai akademik Anda.".format(attendance))
            
        if not recs:
            recs.append("✨ Profil Anda sangat luar biasa! Jam belajar Anda tinggi dan jam gaming harian terkontrol dengan baik.")
            
        for rec in recs:
            st.markdown(f"- {rec}")
            
    with right_col:
        st.markdown("#### ⚡ Profil Anda vs Rata-rata Dataset")
        
        # Prepare comparison table
        comparison_rows = []
        for feat in ['gaming_hours', 'study_hours', 'sleep_hours', 'attendance', 'addiction_score']:
            user_val = float(eval(feat))
            mean_val = stats.get(feat, {}).get('mean', 0.0)
            
            label_map = {
                'gaming_hours': 'Jam Game Harian (Jam)',
                'study_hours': 'Jam Belajar Harian (Jam)',
                'sleep_hours': 'Durasi Tidur (Jam)',
                'attendance': 'Kehadiran (%)',
                'addiction_score': 'Skor Adiksi Game'
            }
            
            comparison_rows.append({
                'Parameter': label_map[feat],
                'Nilai Anda': f"{user_val:.1f}",
                'Rata-rata Global': f"{mean_val:.1f}",
                'Status': "🔴 Kurang baik" if user_val > mean_val and feat in ['gaming_hours', 'addiction_score'] 
                          else "🟢 Lebih baik" if user_val > mean_val and feat in ['study_hours', 'sleep_hours', 'attendance']
                          else "🟡 Di bawah rata-rata" if user_val < mean_val and feat in ['study_hours', 'sleep_hours', 'attendance']
                          else "🟢 Lebih rendah/baik"
            })
            
        comparison_df = pd.DataFrame(comparison_rows)
        st.table(comparison_df.set_index('Parameter'))

# ----------------- TAB 2: EKSPLORASI HUBUNGAN VARIABEL -----------------
with tab2:
    st.markdown("### 📊 Pembuktian Hubungan Variabel dari 8.000 Data Responden")
    st.markdown("""
    Grafik di bawah ini diambil langsung dari hasil agregasi dataset nyata. Perhatikan bagaimana nilai akademik terbukti turun akibat gaming intens dan naik berkat jam belajar.
    """)
    
    if aggregates:
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "🕹️ Jam Gaming vs Nilai", 
            "📚 Jam Belajar vs Nilai",
            "💤 Durasi Tidur vs Nilai",
            "🧠 Stres vs Nilai"
        ])
        
        with subtab1:
            st.markdown("#### Hubungan Jam Gaming Harian dengan Nilai Akademis (Korelasi Negatif: -0.55)")
            gaming_data = pd.DataFrame(aggregates['gaming_hours'])
            st.bar_chart(data=gaming_data, x='gaming_bin', y='mean', use_container_width=True)
            st.info("""
            💡 **Wawasan Penting:** 
            Sangat jelas terlihat pola tangga menurun! Siswa yang bermain game **0-2 jam** sehari memiliki rata-rata nilai **87.2**, sedangkan mereka yang bermain secara intens **6-8 jam** sehari nilai akademisnya turun drastis hingga rata-rata **44.9**.
            """)
            
        with subtab2:
            st.markdown("#### Hubungan Jam Belajar Harian dengan Nilai Akademis (Korelasi Positif Kuat: +0.73)")
            study_data = pd.DataFrame(aggregates['study_hours'])
            st.bar_chart(data=study_data, x='study_bin', y='mean', use_container_width=True)
            st.info("""
            💡 **Wawasan Penting:**
            Grafik menunjukkan korelasi positif yang sangat kuat. Siswa yang belajar **8-10 jam** sehari mencatat rata-rata nilai menakjubkan sebesar **90.4**, dibandingkan mereka yang belajar hanya **1-3 jam** sehari dengan rata-rata nilai **36.9**.
            """)
            
        with subtab3:
            st.markdown("#### Hubungan Durasi Tidur Malam dengan Nilai Akademis (Korelasi Positif: +0.25)")
            sleep_data = pd.DataFrame(aggregates['sleep_hours'])
            st.bar_chart(data=sleep_data, x='sleep_bin', y='mean', use_container_width=True)
            st.info("""
            💡 **Wawasan Penting:**
            Tidur yang ideal (**7 hingga 8.5 jam**) mencatat nilai rata-rata tertinggi yaitu **71.8**. Sedangkan kurang tidur kronis (**4 hingga 5.5 jam**) menurunkan nilai rata-rata ke **59.3**.
            """)
            
        with subtab4:
            st.markdown("#### Rata-rata Nilai berdasarkan Tingkat Stres")
            stress_data = pd.DataFrame(aggregates['stress_level'])
            st.bar_chart(data=stress_data, x='stress_level', y='mean', use_container_width=True)
            st.info("""
            💡 **Wawasan Penting:**
            Siswa dengan tingkat stres tinggi memiliki rata-rata nilai lebih tinggi (**81.0**). Hal ini sering terjadi dalam konteks akademis karena siswa berprestasi cenderung membebani diri dengan jadwal belajar yang lebih padat dan merasakan stres yang lebih tinggi untuk mempertahankan performa mereka.
            """)
            
    else:
        st.warning("Data agregat grafik (`ui_aggregates.json`) tidak ditemukan.")

# ----------------- TAB 3: DETAIL DATA & PERFORMA MODEL -----------------
with tab3:
    st.markdown("### ⚙️ Detail Model Machine Learning & Dataset Baru")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        #### 🤖 Informasi Model Machine Learning
        * **Model:** Scikit-Learn Pipeline (`ColumnTransformer` + `RandomForestRegressor`)
        * **Jumlah Estimator (Trees):** 100
        * **Kedalaman Maksimum (Max Depth):** 12
        * **Akurasi Model (R² Score):** **92.12%**
        * **Mean Absolute Error (MAE):** **4.87 poin**
        
        > [!NOTE]
        > **Akurasi R² sebesar 92.12%** berarti model ini mampu memprediksi nilai akademis Anda secara sangat presisi berdasarkan parameter masukan Anda. Kesalahan rata-rata prediksi hanya berkisar ±4.87 poin.
        """)
        
    with col_b:
        st.markdown("""
        #### 📁 Informasi Dataset Baru
        * **Nama File:** `Gaming_Academic_Performance.csv`
        * **Ukuran Dataset:** 8.000 baris, 14 kolom
        * **Target Prediksi:** `grades` (Skor Nilai Akademis, skala 0-100)
        * **Korelasi Terkuat terhadap Nilai:**
          * Jam Belajar harian (`study_hours`): **+0.73** (Sangat Kuat)
          * Jam Gaming harian (`gaming_hours`): **-0.55** (Negatif Kuat)
          * Skor Adiksi Game (`addiction_score`): **-0.50** (Negatif Kuat)
          * Waktu Gadget harian (`device_usage`): **-0.47** (Negatif Kuat)
        """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a0aec0; font-size: 0.85rem; padding-bottom: 20px;">
    Dibuat untuk analisis praktikum Hubungan Gaming & Akademis. Dataset berukuran 8.000 Siswa.
</div>
""", unsafe_allow_html=True)
