import streamlit as st
import random
import pandas as pd
import numpy as np
import base64
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
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
# FUNGSI MODEL PREDIKSI (LSTM)
# ==============================================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def create_lstm_model():
    input_size = 1
    hidden_size = 4

    np.random.seed(42)

    Wf = np.random.randn(hidden_size, input_size + hidden_size)
    Wi = np.random.randn(hidden_size, input_size + hidden_size)
    Wc = np.random.randn(hidden_size, input_size + hidden_size)
    Wo = np.random.randn(hidden_size, input_size + hidden_size)

    bf = np.zeros(hidden_size)
    bi = np.zeros(hidden_size)
    bc = np.zeros(hidden_size)
    bo = np.zeros(hidden_size)

    return Wf, Wi, Wc, Wo, bf, bi, bc, bo


def lstm_predict(sequence, model):
    Wf, Wi, Wc, Wo, bf, bi, bc, bo = model

    hidden_size = Wf.shape[0]
    h = np.zeros(hidden_size)
    c = np.zeros(hidden_size)

    for x in sequence:
        x = np.array([x])
        combined = np.concatenate((h, x))

        f = sigmoid(np.dot(Wf, combined) + bf)
        i = sigmoid(np.dot(Wi, combined) + bi)
        c_hat = np.tanh(np.dot(Wc, combined) + bc)

        c = f * c + i * c_hat
        o = sigmoid(np.dot(Wo, combined) + bo)
        h = o * np.tanh(c)

    return h.mean()

# ==============================================================================
# APLIKASI UTAMA (MAIN APP)
# ==============================================================================
def main_app():
    # REFRESH DATA SETIAP 1 DETIK
    st_autorefresh(interval=1000, key="datarefresh")
    now = datetime.now(ZoneInfo("Asia/Jakarta"))
    panel_style = """<div style="background-color: rgba(255, 255, 255, 0.95); padding: 15px; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.08);">"""

    # DATA LOKASI MONITORING
    lokasi_dict = {
        "Titik A - Ruas Jalan Sudirman": {"lat": -6.2023, "lon": 106.8190, "desc": "Jakarta, Indonesia"},
        "Titik B - Ruas Jalan Gatot Subroto": {"lat": -6.2300, "lon": 106.8241, "desc": "Jakarta, Indonesia"},
        "Titik C - Ruas Jalan Thamrin": {"lat": -6.1891, "lon": 106.8230, "desc": "Jakarta, Indonesia"}
    }

    # --- SIDEBAR ---
    st.sidebar.title("Navigasi")
    nav_option = st.sidebar.radio("Menu Utama:", ["Titik Lokasi", "Detail Monitoring"])
    
    st.sidebar.markdown("---")
    
    # Inisialisasi history data jika belum ada
    if "history_master" not in st.session_state:
        st.session_state.history_master = {}
        for loc in lokasi_dict.keys():
            dummy_rows = []
            for i in range(47, -1, -1):
                t = now - timedelta(minutes=i * 30)
                jam = t.hour
                if 8 <= jam <= 12:
                    co2_val = 700 + ((jam - 8) * 75) + random.uniform(-10, 10)
                elif 12 < jam <= 23:
                    co2_val = 1000 - ((jam - 12) * 40) + random.uniform(-10, 10)
                else:
                    co2_val = 550 + random.uniform(-15, 15)
                temp_val = 34 + (abs(14-jam) * -0.2) + 6 + random.uniform(-0.5, 0.5)
                
                dummy_rows.append({
                    "time": t,
                    "ph": round(random.uniform(8.0, 10.5), 2),
                    "turbidity": round(7 + ((47-i) * 0.22), 2),
                    "co2": round(np.clip(co2_val, 700, 900), 0),
                    "temp": round(np.clip(temp_val, 34, 42), 1),
                    "color_sensor_pct": random.randint(20, 98), # Data random sensor warna
                    "lat": lokasi_dict[loc]["lat"],
                    "lon": lokasi_dict[loc]["lon"]
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
        untuk memantau kualitas Mikroalga dan kadar CO2.
        """)

        # Mengolah data untuk peta
        map_df = pd.DataFrame([
            {"Titik": k, "lat": v["lat"], "lon": v["lon"], "Keterangan": v["desc"]} 
            for k, v in lokasi_dict.items()
        ])
        
        col_home1, col_home2 = st.columns([3, 1])
        
        with col_home1:
            st.markdown(panel_style, unsafe_allow_html=True)
            st.markdown("### Lokasi Instalasi Sensor")
            st.map(map_df, zoom=12, use_container_width=True)
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
            
            st.success("Gunakan menu 'Detail Monitoring' untuk melihat grafik sensor spesifik.")

    # ==========================================================================
    # HALAMAN 2: DETAIL MONITORING PER TITIK
    # ==========================================================================
    else:
        st.sidebar.markdown("### Pilih Lokasi")
        selected_loc = st.sidebar.selectbox("Pilih Ruas Jalan:", list(lokasi_dict.keys()))
        coords = lokasi_dict[selected_loc]
        
        hist = st.session_state.history_master[selected_loc]
        sensor = hist.iloc[-1]

        # HEADER DETAIL
        st.title(f"{selected_loc}")
        st.subheader(f"Tanggal: {now.strftime('%d %B %Y')} | Waktu: {now.strftime('%H:%M:%S')} WIB")

       # Inisialisasi persen jika belum ada
        if "persen_turb" not in st.session_state:
            st.session_state.persen_turb = 10

        val_turb = sensor['turbidity']
        if val_turb < 9:
            label_turb = "Sangat rendah"
            min_val, max_val = 1, 20
        elif val_turb < 12:
            label_turb = "Rendah"
            min_val, max_val = 21, 40
        elif val_turb < 14:
            label_turb = "Sedang"
            min_val, max_val = 41, 60
        elif val_turb < 17:
            label_turb = "Tinggi"
            min_val, max_val = 61, 80
        else:
            label_turb = "Sangat tinggi"
            min_val, max_val = 81, 100

        # Gerak perlahan ke dalam range
        current = st.session_state.persen_turb

        if current < min_val:
            current += 1
        elif current > max_val:
            current -= 1
        else:
            current += 0.000035 # naik pelan

        st.session_state.persen_turb = float(np.clip(current, 1, 100))
        persen_turb = round(st.session_state.persen_turb, 1)
        # METRIC CARDS
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("PH", f"{sensor['ph']}")
        col_m2.metric("Kekeruhan", f"{label_turb}")
        col_m3.metric("CO2 (ppm)", f"{int(sensor['co2'])}")
        col_m4.metric("Suhu (°C)", f"{sensor['temp']}")

        # STATUS BOXES
        st.subheader("Status Operasional")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if sensor['temp'] < 40:
                st.markdown(f'<div class="status-box-normal">Cooling Mati (Suhu Aman: {sensor["temp"]}°C)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-box-red">Cooling Menyala (Suhu Tinggi: {sensor["temp"]}°C)</div>', unsafe_allow_html=True)
        with col_s2:
            if sensor['turbidity'] >= 17:
                st.markdown(f'<div class="status-box-red">Mikroalga Siap Panen (Kondisi: {label_turb.upper()})</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-box-normal">Mikroalga Normal ({label_turb})</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="explanation-box">
            <strong>Keterangan Pertumbuhan:</strong> Suhu optimal mikroalga berada pada 35-40°C. 
            Jika suhu > 40°C, cooling menyala. Kekeruhan sangat tinggi menunjukkan mikroalga dalam kondisi siap panen.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # PREDIKSI LSTM
        st.subheader("Prediksi 30 Menit ke Depan")

        model = create_lstm_model()

        # Ambil 6 data CO2 terakhir (3 jam terakhir)
        co2_series = hist["co2"].values[-6:]
        co2_norm = co2_series / 2000  # normalisasi

        prediction = lstm_predict(co2_norm, model)

        pred_co2 = float(np.clip(prediction * 1000 + 700, 650, 850))

        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin-bottom: 40px;">
            <div class="prediction-panel" style="width: 50%;">
                <div style="font-size: 18px;">Estimasi CO2 Berikutnya</div>
                <div style="font-size: 40px; font-weight: 900;">{int(pred_co2)} ppm</div>
                <div style="font-size: 14px; opacity: 0.8;">Berdasarkan Pemantauan Alat Pengukuran</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # GRAFIK ANALISIS
        st.subheader("Analisis Riwayat Data (24 Jam Terakhir)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(panel_style, unsafe_allow_html=True)
            fig_co2 = px.line(hist, x="time", y="co2", title="Tren CO2 (ppm)", markers=True, color_discrete_sequence=['#4caf50'])
            fig_co2.update_yaxes(range=[300, 1200])
            fig_co2.add_hline(y=800, line_dash="dash", line_color="blue", annotation_text="Baik")
            fig_co2.add_hline(y=1000, line_dash="dash", line_color="red", annotation_text="Tinggi")
            st.plotly_chart(fig_co2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c2:
            st.markdown(panel_style, unsafe_allow_html=True)
            pie_data = pd.DataFrame({"Kategori": ["Kekeruhan", "Sisa"], "Nilai": [persen_turb, 100 - persen_turb]})
            fig_pie = px.pie(pie_data, values='Nilai', names='Kategori', title=f"Tingkat kekeruhan: {persen_turb}%", color_discrete_sequence=['#2e7d32', '#e8f5e9'], hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown(panel_style, unsafe_allow_html=True)
            fig_ph = px.line(hist, x="time", y="ph", title="Tren pH (Tumbuh Baik: 8.5-10)", markers=True, color_discrete_sequence=['#2196f3'])
            fig_ph.update_yaxes(range=[7.5, 11])
            # AMBANG BATAS PH BARU (8.5 - 10)
            fig_ph.add_hrect(y0=8.5, y1=10.0, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Rentang Optimal")
            st.plotly_chart(fig_ph, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c4:
            st.markdown(panel_style, unsafe_allow_html=True)
            fig_temp = px.line(hist, x="time", y="temp", title="Tren Suhu (Rentang Optimal: 35-40°C)", markers=True, color_discrete_sequence=['#ff9800'])
            fig_temp.update_yaxes(range=[30, 45])
            # AMBANG BATAS SUHU BARU (35 - 40)
            fig_temp.add_hrect(y0=35, y1=40, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="Rentang Optimal")
            st.plotly_chart(fig_temp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # LOG TABEL
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Log Data Historis")
        df_tampil = hist.copy().sort_values("time", ascending=False)
        df_tampil['Waktu'] = df_tampil['time'].dt.strftime('%H:%M:%S')
        def get_label(x):
            if x < 9: return "sangat rendah"
            elif x < 12: return "rendah"
            elif x < 14: return "sedang"
            elif x < 17: return "tinggi"
            else: return "sangat tinggi"
        df_tampil['Kondisi'] = df_tampil['turbidity'].apply(get_label)
        st.dataframe(df_tampil[['Waktu', 'ph', 'Kondisi', 'co2', 'temp']].rename(columns={
            'ph': 'PH', 'Kondisi': 'Tingkat Kekeruhan', 'co2': 'CO2 (ppm)', 'temp': 'Suhu (°C)'
        }), use_container_width=True)

    # FOOTER
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: #757575;'>ECO-MIRACLE v2.0 - Sistem Monitoring Mikroalga Terintegrasi - {now.year}</p>", unsafe_allow_html=True)

# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    check_login()
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()







