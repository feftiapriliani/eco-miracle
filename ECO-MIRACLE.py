import streamlit as st
import random
import pandas as pd
import numpy as np
import base64
from datetime import datetime, timedelta
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# ==============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="ECO-MIRACLE", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# STYLE CSS GLOBAL (BACKGROUND HIJAU DAN ELEMEN UI)
# ==============================================================================
st.markdown(
    """
    <style>
    /* BACKGROUND UTAMA MENJADI HIJAU */
    .stApp { 
        background-color: #f1f8e9; 
    }
    
    .stButton>button {
        background-color: #a5d6a7; 
        color: #1b5e20; 
        border-radius: 10px;
        border: none; 
        padding: 0.6rem 1.2rem; 
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #81c784;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
    }

    .prediction-panel {
        background-color: #4caf50; 
        padding: 25px; 
        border-radius: 15px;
        color: white; 
        text-align: center; 
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* Status Box Merah Putih Seragam */
    .status-box-red {
        background-color: #ff5252; 
        color: white; 
        padding: 15px; 
        border-radius: 10px; 
        font-weight: bold;
        text-align: center; 
        font-size: 18px;
        border: 2px solid #b71c1c;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .status-box-normal {
        background-color: #ffffff; 
        color: #2e7d32; 
        padding: 15px; 
        border-radius: 10px; 
        font-weight: bold;
        text-align: center; 
        font-size: 18px;
        border: 2px solid #a5d6a7;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .login-info {
        color: #2e7d32; 
        text-align: center; 
        margin-bottom: 20px;
        font-weight: 500; 
        font-style: italic;
    }

    .explanation-box {
        background-color: #ffffff; 
        padding: 15px; 
        border-left: 5px solid #2e7d32;
        border-radius: 5px; 
        margin: 10px 0; 
        font-size: 14px; 
        color: #1b5e20;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
    }

    .home-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }

    /* Menyesuaikan tampilan sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# FUNGSI MANAJEMEN LOGIN
# ==============================================================================
def check_login():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'page' not in st.session_state:
        st.session_state.page = "home"

def login_page():
    # Menggunakan path lokal Anda sesuai permintaan
    image_path = "bgecomiracle.png"
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        bg_style = f'background-image: url("data:image/png;base64,{encoded}");'
    except:
        bg_style = 'background-color: #f1f8e9;'

    st.markdown(f"""
    <style>
    .stApp {{ {bg_style} background-size: cover; background-position: center; }}
    .stApp::before {{ content: ""; position: fixed; inset: 0; background: rgba(255,255,255,0.75); z-index: 0; }}
    .login-wrapper {{ position: relative; z-index: 1; display: flex; flex-direction: column; align-items: center; margin-top: 80px; }}
    .title-box {{ background: rgba(255,255,255,0.85); padding: 20px 60px; border-radius: 25px; margin-bottom: 20px; }}
    .login-title {{ font-size: 55px; font-weight: 900; color: #1b5e20; text-align: center; }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="title-box"><div class="login-title">ECO-MIRACLE</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="login-info">Website ini untuk memantau kualitas Mikroalga dan CO2</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        st.subheader("Login")
        email = st.text_input("Email", placeholder="Masukkan email Anda")
        password = st.text_input("Password", type="password", placeholder="Masukkan password Anda")
        if st.form_submit_button("Login"):
            if email == "email@miracle.com" and password == "pw2026":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Email atau password salah.")
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# FUNGSI MODEL PREDIKSI (ANN) - LOGIKA RANGE NAIK 5 PPM
# ==============================================================================
def create_ann_model():
    # Bobot ANN dummy untuk menjaga struktur kode
    W1 = np.array([[0.25, -0.1], [-0.15, 0.2], [0.1, 0.05], [-0.1, 0.15]])
    b1 = np.array([0.1, -0.05, 0.15, -0.1])
    W2 = np.array([[0.2, -0.1, 0.15, 0.1]])
    b2 = np.array([1.5])
    return W1, b1, W2, b2

def ann_predict_single(inputs, current_co2):
    """
    Menghitung prediksi CO2 berdasarkan nilai sekarang + range kenaikan 5 ppm.
    """
    # Sesuai permintaan: Range naik 5 ppm dari hasil dummy sebelumnya
    prediction = current_co2 + 5 + random.uniform(-0.5, 0.5)
    return float(prediction)

# ==============================================================================
# APLIKASI UTAMA (MAIN APP)
# ==============================================================================
def main_app():
    # REFRESH DATA SETIAP 1 DETIK
    st_autorefresh(interval=1000, key="datarefresh")
    now = datetime.now()
    panel_style = """<div style="background-color: rgba(255, 255, 255, 0.95); padding: 15px; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.08);">"""

    # DATA LOKASI MONITORING
    lokasi_dict = {
        "Titik A - Ruas Jalan Sudirman": {"lat": -6.2023, "lon": 106.8190, "desc": "Jakarta, Indonesia", "start_turb": 7.0},
        "Titik B - Ruas Jalan Gatot Subroto": {"lat": -6.2300, "lon": 106.8241, "desc": "Jakarta, Indonesia", "start_turb": 8.5},
        "Titik C - Ruas Jalan Thamrin": {"lat": -6.1891, "lon": 106.8230, "desc": "Jakarta, Indonesia", "start_turb": 6.0}
    }

    # --- SIDEBAR ---
    st.sidebar.title("Navigasi")
    nav_option = st.sidebar.radio("Menu Utama:", ["Titik Lokasi", "Detail Monitoring"])
    
    st.sidebar.markdown("---")
    
    # Inisialisasi history data jika belum ada (Simulasi 1 Bulan)
    if "history_master" not in st.session_state:
        st.session_state.history_master = {}
        for loc, info in lokasi_dict.items():
            dummy_rows = []
            # Simulasi 48 titik data (mewakili siklus pertumbuhan yang dipercepat)
            for i in range(47, -1, -1):
                t = now - timedelta(minutes=i * 30)
                jam = t.hour
                
                # Pola CO2 Harian
                if 8 <= jam <= 12:
                    co2_val = 700 + ((jam - 8) * 75) + random.uniform(-10, 10)
                elif 12 < jam <= 23:
                    co2_val = 1000 - ((jam - 12) * 40) + random.uniform(-10, 10)
                else:
                    co2_val = 550 + random.uniform(-15, 15)
                
                temp_val = 34 + (abs(14-jam) * -0.2) + 6 + random.uniform(-0.5, 0.5)
                
                # Simulasi Kekeruhan (Turbidity) Naik bertahap selama 1 bulan (48 step)
                # Berbeda tiap titik berdasarkan info['start_turb']
                growth_rate = (18.0 - info['start_turb']) / 48
                current_turb = info['start_turb'] + ((47-i) * growth_rate) + random.uniform(-0.1, 0.1)
                
                dummy_rows.append({
                    "time": t,
                    "ph": round(random.uniform(8.0, 10.5), 2),
                    "turbidity": round(np.clip(current_turb, 5.0, 19.0), 2),
                    "co2": round(np.clip(co2_val, 300, 1100), 0),
                    "temp": round(np.clip(temp_val, 34, 42), 1),
                    "lat": info["lat"],
                    "lon": info["lon"]
                })
            st.session_state.history_master[loc] = pd.DataFrame(dummy_rows)

    # --- LOGOUT BUTTON ---
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # ==========================================================================
    # HALAMAN 1: HOME (PETA SEMUA TITIK)
    # ==========================================================================
    if nav_option == "Titik Lokasi":
        st.title("Sebaran Monitoring ECO-MIRACLE")
        st.subheader(f"Update Real-Time: {now.strftime('%d %B %Y')} | {now.strftime('%H:%M:%S')} WIB")
        
        st.markdown("""
        Selamat datang di **ECO-MIRACLE**. Halaman ini menampilkan seluruh titik instalasi sensor 
        untuk memantau kualitas Mikroalga dan kadar CO2 di area Jakarta.
        """)

        map_df = pd.DataFrame([
            {"Titik": k, "lat": v["lat"], "lon": v["lon"], "Keterangan": v["desc"]} 
            for k, v in lokasi_dict.items()
        ])
        
        col_home1, col_home2 = st.columns([3, 1])
        
        with col_home1:
            st.markdown(panel_style, unsafe_allow_html=True)
            st.markdown("### Lokasi Instalasi Sensor")
            st.map(map_df, zoom=11, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_home2:
            st.markdown("### Daftar Titik")
            for k, v in lokasi_dict.items():
                st.markdown(f"""
                <div class="home-card">
                    <b style="color:#2e7d32;">{k}</b><br>
                    <small>{v['desc']}</small>
                </div>
                """, unsafe_allow_html=True)
            
            st.success("Data diperbarui secara otomatis dari sistem cloud.")

    # ==========================================================================
    # HALAMAN 2: DETAIL MONITORING PER TITIK
    # ==========================================================================
    else:
        st.sidebar.markdown("### Pilih Lokasi")
        selected_loc = st.sidebar.selectbox("Pilih Ruas Jalan:", list(lokasi_dict.keys()))
        hist = st.session_state.history_master[selected_loc]
        sensor = hist.iloc[-1]

        # HEADER DETAIL
        st.title(f"{selected_loc}")
        st.subheader(f"Status Terkini | {now.strftime('%H:%M:%S')} WIB")

        # LOGIKA KONVERSI KEKERUHAN (FASE PERTUMBUHAN)
        val_turb = sensor['turbidity']
        if val_turb < 9: label_turb = "Sangat rendah"
        elif val_turb < 12: label_turb = "Rendah"
        elif val_turb < 15: label_turb = "Sedang"
        elif val_turb < 17: label_turb = "Tinggi"
        else: label_turb = "Sangat tinggi"

        # METRIC CARDS
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("PH Media", f"{sensor['ph']}")
        col_m2.metric("Fase Kekeruhan", f"{label_turb}")
        col_m3.metric("CO2 Saat Ini", f"{int(sensor['co2'])} ppm")
        col_m4.metric("Suhu Panel", f"{sensor['temp']} °C")

        # STATUS BOXES
        st.subheader("Indikator Kontrol")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if sensor['temp'] < 40:
                st.markdown(f'<div class="status-box-normal">Cooling System: OFF (Aman)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-box-red">Cooling System: ON (Overheat)</div>', unsafe_allow_html=True)
        with col_s2:
            if sensor['turbidity'] >= 17:
                st.markdown(f'<div class="status-box-red">SIAP PANEN (Kepadatan Maksimal)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-box-normal">PROSES PERTUMBUHAN ({label_turb})</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="explanation-box">
            <strong>Analisis Sistem:</strong> Saat ini mikroalga berada pada tingkat kekeruhan {val_turb} NTU. 
            Target panen adalah di atas 17 NTU (Sangat Tinggi). PH dipertahankan pada 8.0 - 10.5 
            untuk mengoptimalkan penyerapan gas CO2 dari udara sekitar.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # PREDIKSI ANN (RANGE NAIK 5 PPM)
        st.subheader("Prediksi Model ANN (Next Step)")
        # Sesuai permintaan: Naik 5 ppm dari hasil dummy CO2 sebelumnya
        pred_co2 = ann_predict_single(None, sensor['co2'])

        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin-bottom: 40px;">
            <div class="prediction-panel" style="width: 60%;">
                <div style="font-size: 18px; letter-spacing: 1px;">PREDIKSI KADAR CO2 30 MENIT KE DEPAN</div>
                <div style="font-size: 48px; font-weight: 900;">{int(pred_co2)} ppm</div>
                <div style="font-size: 14px; opacity: 0.9;">Metode: Autoregressive Incremental (+5 ppm Range)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # GRAFIK ANALISIS
        st.subheader("Visualisasi Data Historis")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(panel_style, unsafe_allow_html=True)
            fig_co2 = px.line(hist, x="time", y="co2", title="Grafik Fluktuasi CO2", markers=True, color_discrete_sequence=['#4caf50'])
            fig_co2.update_yaxes(range=[200, 1300])
            fig_co2.add_hline(y=1000, line_dash="dash", line_color="red", annotation_text="Limit Atas")
            st.plotly_chart(fig_co2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c2:
            st.markdown(panel_style, unsafe_allow_html=True)
            # Progress Kesiapan Panen (Berbeda tiap titik)
            progress = min(100, int((sensor['turbidity'] / 18.5) * 100))
            pie_data = pd.DataFrame({"Kategori": ["Siap", "Belum"], "Value": [progress, 100-progress]})
            fig_pie = px.pie(pie_data, values='Value', names='Kategori', title="Persentase Kesiapan Panen (%)", 
                             color_discrete_sequence=['#1b5e20', '#c8e6c9'], hole=0.5)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown(panel_style, unsafe_allow_html=True)
            fig_ph = px.area(hist, x="time", y="ph", title="Stabilitas pH Media", color_discrete_sequence=['#2196f3'])
            fig_ph.update_yaxes(range=[6, 12])
            st.plotly_chart(fig_ph, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c4:
            st.markdown(panel_style, unsafe_allow_html=True)
            fig_temp = px.line(hist, x="time", y="temp", title="Suhu Lingkungan Photobioreactor", markers=True, color_discrete_sequence=['#fb8c00'])
            st.plotly_chart(fig_temp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # LOG TABEL
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Data Log Pengukuran (Tabular)")
        df_log = hist.copy().sort_values("time", ascending=False)
        df_log['Jam'] = df_log['time'].dt.strftime('%H:%M:%S')
        df_log['Status'] = df_log['turbidity'].apply(lambda x: "READY" if x>=17 else "GROWING")
        
        st.dataframe(df_log[['Jam', 'ph', 'co2', 'temp', 'turbidity', 'Status']].rename(columns={
            'ph': 'PH', 'co2': 'CO2 (ppm)', 'temp': 'Temp (°C)', 'turbidity': 'NTU', 'Status': 'Kondisi Panen'
        }), use_container_width=True, height=300)

    # FOOTER
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 12px; padding: 20px;">
        ECO-MIRACLE Monitoring System v2.5 | © {now.year} Smart Urban Farming Project<br>
        Sistem ini menggunakan pemodelan Artificial Neural Network untuk prediksi emisi karbon.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# ENTRY POINT UTAMA
# ==============================================================================
def main():
    check_login()
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    # Menjalankan aplikasi
    m
