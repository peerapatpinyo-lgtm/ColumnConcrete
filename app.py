import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import interp1d

class RCColumnProfessional:
    def __init__(self, fc, fy, b, h, db_mm, n_bars, cover_cm):
        self.fc, self.fy, self.b, self.h = fc, fy, b, h
        self.Es = 2.04e6
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))
        
        self.Ag = self.b * self.h
        self.fc_Ag_ton = (self.fc * self.Ag) / 1000  # แปลงเป็นตัน
        
        # คำนวณพื้นที่เหล็กเสริม
        as_single = (np.pi * (db_mm/10)**2) / 4
        d_prime = cover_cm + 0.9 + (db_mm/20)
        d = h - d_prime                     
        
        self.layers = [
            {'as': (n_bars/2) * as_single, 'd': d_prime},
            {'as': (n_bars/2) * as_single, 'd': d}
        ]
        
        self.total_as = sum(l['as'] for l in self.layers)
        self.rho = self.total_as / self.Ag
        self.dt = max(l['d'] for l in self.layers)

    def solve(self):
        results = []
        c_values = np.concatenate([
            np.linspace(0.001, self.dt, 200), 
            np.linspace(self.dt, self.h * 3, 200)
        ])
        
        for c in c_values:
            a = min(self.beta1 * c, self.h)
            
            Cc = 0.85 * self.fc * a * self.b
            Mc = Cc * (self.h/2 - a/2)
            
            Pn_s, Mn_s = 0, 0
            for layer in self.layers:
                eps_s = 0.003 * (c - layer['d']) / c
                fs = np.clip(eps_s * self.Es, -self.fy, self.fy)
                Fsi = layer['as'] * fs
                Pn_s += Fsi
                Mn_s += Fsi * (self.h/2 - layer['d'])

            pn = (Cc + Pn_s) / 1000 # ton
            mn = (Mc + Mn_s) / 100000 # ton-m
            
            # คำนวณค่า K และ R (Normalized parameters)
            kn = pn / self.fc_Ag_ton
            rn = mn / (self.fc_Ag_ton * (self.h / 100))
            
            et = 0.003 * (self.dt - c) / c
            ey = self.fy / self.Es
            
            if et >= 0.005:
                phi = 0.90
            elif et <= ey:
                phi = 0.65
            else:
                phi = 0.65 + 0.25 * (et - ey) / (0.005 - ey)
            
            results.append({
                'c': c, 'et': et, 'phi': phi,
                'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn,
                'Kn': kn, 'Rn': rn, 'phiKn': phi * kn, 'phiRn': phi * rn
            })

        # จุด Pure Tension
        tn = -self.total_as * self.fy / 1000
        results.append({
            'c': 0, 'et': 0.005, 'phi': 0.9,
            'Pn': tn, 'Mn': 0, 'phiPn': 0.9 * tn, 'phiMn': 0,
            'Kn': tn / self.fc_Ag_ton, 'Rn': 0, 'phiKn': 0.9 * (tn / self.fc_Ag_ton), 'phiRn': 0
        })

        # จุด Pure Compression
        po = (0.85 * self.fc * (self.Ag - self.total_as) + self.fy * self.total_as) / 1000
        phi_pn_max = 0.65 * 0.80 * po
        phi_kn_max = phi_pn_max / self.fc_Ag_ton

        df = pd.DataFrame(results).sort_values('Pn', ascending=True)
        return df, phi_pn_max, phi_kn_max, self.rho, self.fc_Ag_ton

# --- STREAMLIT UI ---
st.set_page_config(page_title="Advanced RC Column Design", layout="wide")
st.title("🏗️ Advanced RC Column Interaction Diagram")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("1. Section Properties")
    fc = st.number_input("f'c (ksc)", value=280)
    fy = st.number_input("fy (ksc)", value=4000)
    b = st.number_input("Width, b (cm)", value=40)
    h = st.number_input("Depth, h (cm)", value=60)
    
    st.subheader("2. Reinforcement")
    db = st.selectbox("Bar Size (mm)", [16, 20, 25, 28, 32], index=2)
    n_bars = st.number_input("Total Bars (Even number)", 4, 40, 8, step=2)
    cover = st.number_input("Covering (cm)", value=4.0)

    st.subheader("3. Applied Loads (Demand)")
    Pu = st.number_input("Factored Axial Load, Pu (ton)", value=150.0)
    Mu = st.number_input("Factored Moment, Mu (ton-m)", value=15.0)

engine = RCColumnProfessional(fc, fy, b, h, db, n_bars, cover)
df, phi_pn_max, phi_kn_max, rho, fc_Ag_ton = engine.solve()

# คำนวณ Demand Normalized
Ku = Pu / fc_Ag_ton
Ru = Mu / (fc_Ag_ton * (h / 100))

# เตรียมเส้นกราฟที่ตัดยอด (Capped)
df_design = df.copy()
df_design['phiPn'] = df_design['phiPn'].clip(upper=phi_pn_max)
df_design['phiKn'] = df_design['phiKn'].clip(upper=phi_kn_max)

# ตรวจสอบความปลอดภัย
try:
    interp_func = interp1d(df_design['phiPn'], df_design['phiMn'], kind='linear', fill_value=0, bounds_error=False)
    max_Mu_allowable = interp_func(Pu)
    is_safe = (Mu <= max_Mu_allowable) and (Pu <= phi_pn_max) and (Pu >= df_design['phiPn'].min())
except:
    is_safe = False

with col2:
    # --- Status Board ---
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    status_col1.metric("Reinforcement Ratio (ρ)", f"{rho*100:.2f}%", 
                       "OK (1%-8%)" if 0.01 <= rho <= 0.08 else "Warning (Out of bounds)", 
                       delta_color="normal" if 0.01 <= rho <= 0.08 else "inverse")
    status_col2.metric("Max Capacity (ΦPn,max)", f"{phi_pn_max:.1f} ton")
    status_col3.metric("Demand Ku", f"{Ku:.4f}")
    status_col4.metric("Demand Ru", f"{Ru:.4f}")

    if is_safe:
        st.success(f"✅ **SAFE:** The applied load is INSIDE the design envelope.")
    else:
        st.error(f"❌ **UNSAFE:** The applied load EXCEEDS the column capacity.")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 P-M Curve (Actual)", "📈 K-R Curve (Normalized)", "🗄️ Detailed Data"])

    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df['Mn'], y=df['Pn'], name="Nominal Capacity", line=dict(color='gray', dash='dash')))
        fig1.add_trace(go.Scatter(x=df_design['phiMn'], y=df_design['phiPn'], fill='tozeroy', name="Design Capacity (Φ)", line=dict(color='navy', width=3)))
        fig1.add_trace(go.Scatter(x=[Mu], y=[Pu], mode='markers', name="Demand (Mu, Pu)", marker=dict(color='red', size=12, symbol='x')))
        fig1.update_layout(xaxis_title="Moment, M (ton-m)", yaxis_title="Axial Load, P (ton)", plot_bgcolor='white', height=550)
        fig1.update_xaxes(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black')
        fig1.update_yaxes(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black')
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Rn'], y=df['Kn'], name="Nominal (Kn-Rn)", line=dict(color='gray', dash='dash')))
        fig2.add_trace(go.Scatter(x=df_design['phiRn'], y=df_design['phiKn'], fill='tozeroy', name="Design (ΦKn-ΦRn)", line=dict(color='teal', width=3)))
        fig2.add_trace(go.Scatter(x=[Ru], y=[Ku], mode='markers', name="Demand (Ru, Ku)", marker=dict(color='red', size=12, symbol='x')))
        fig2.update_layout(xaxis_title="Normalized Moment, R (M / f'c·Ag·h)", yaxis_title="Normalized Axial, K (P / f'c·Ag)", plot_bgcolor='white', height=550)
        fig2.update_xaxes(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black')
        fig2.update_yaxes(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black')
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### 🧮 Calculation Table")
        # แสดงผลตาราง เลือกเฉพาะคอลัมน์สำคัญมาแสดงให้ดูสะอาดตา
        display_df = df_design[['c', 'et', 'phi', 'Pn', 'Mn', 'phiPn', 'phiMn', 'Kn', 'Rn', 'phiKn', 'phiRn']].copy()
        display_df.columns = ['c (cm)', 'Strain (et)', 'Phi (Φ)', 'Pn (ton)', 'Mn (ton-m)', 'ΦPn', 'ΦMn', 'Kn', 'Rn', 'ΦKn', 'ΦRn']
        st.dataframe(display_df.style.format("{:.4f}"), use_container_width=True, height=550)
