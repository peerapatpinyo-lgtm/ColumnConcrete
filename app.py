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
        self.fc_Ag_ton = (self.fc * self.Ag) / 1000 
        
        # คำนวณพื้นที่เหล็กเสริม
        as_single = (np.pi * (db_mm/10)**2) / 4
        d_prime = cover_cm + 0.9 + (db_mm/20)
        d = h - d_prime                     
        
        self.layers = [
            {'as': (n_bars/2) * as_single, 'd': d_prime},
            {'as': (n_bars/2) * as_single, 'd': d}
        ]
        
        self.total_as = sum(l['as'] for l in self.layers)
        self.rho = self.total_as / self.Ag  # อัตราส่วนเหล็กเสริม
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

        tn = -self.total_as * self.fy / 1000
        results.append({
            'c': 0, 'et': 0.005, 'phi': 0.9,
            'Pn': tn, 'Mn': 0, 'phiPn': 0.9 * tn, 'phiMn': 0,
            'Kn': tn / self.fc_Ag_ton, 'Rn': 0, 'phiKn': 0.9 * (tn / self.fc_Ag_ton), 'phiRn': 0
        })

        po = (0.85 * self.fc * (self.Ag - self.total_as) + self.fy * self.total_as) / 1000
        phi_pn_max = 0.65 * 0.80 * po
        phi_kn_max = phi_pn_max / self.fc_Ag_ton

        df = pd.DataFrame(results).sort_values('Pn', ascending=True)
        return df, phi_pn_max, phi_kn_max, self.rho, self.fc_Ag_ton

# --- STREAMLIT UI ---
st.set_page_config(page_title="Professional RC Column Design", layout="wide")
st.title("🏗️ Professional RC Column Design & Checks")

col1, col2 = st.columns([1, 2.5])

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

    st.subheader("3. Slenderness (KL/r)")
    K_factor = st.number_input("Effective Length Factor (K)", value=1.0, step=0.1)
    Lu = st.number_input("Unsupported Length, Lu (m)", value=3.0, step=0.5)

    st.subheader("4. Applied Loads")
    Pu = st.number_input("Factored Axial Load, Pu (ton)", value=150.0)
    Mu = st.number_input("Factored Moment, Mu (ton-m)", value=15.0)

engine = RCColumnProfessional(fc, fy, b, h, db, n_bars, cover)
df, phi_pn_max, phi_kn_max, rho, fc_Ag_ton = engine.solve()

# --- Structural Checks ---
rho_pct = rho * 100
r_radius = 0.3 * h # รัศมีไจเรชันโดยประมาณตาม ACI 318
kl_r = (K_factor * Lu * 100) / r_radius # แปลง Lu เป็น cm

df_design = df.copy()
df_design['phiPn'] = df_design['phiPn'].clip(upper=phi_pn_max)
df_design['phiKn'] = df_design['phiKn'].clip(upper=phi_kn_max)

try:
    interp_func = interp1d(df_design['phiPn'], df_design['phiMn'], kind='linear', fill_value=0, bounds_error=False)
    max_Mu_allowable = interp_func(Pu)
    is_safe = (Mu <= max_Mu_allowable) and (Pu <= phi_pn_max) and (Pu >= df_design['phiPn'].min())
except:
    is_safe = False

with col2:
    # --- Status & Warning Dashboard ---
    st.markdown("### 📋 Design Summary & Checks")
    
    # แบ่ง 4 คอลัมน์สำหรับโชว์ตัวเลขสรุป
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Steel Ratio (ρ)", f"{rho_pct:.2f} %")
    metric2.metric("Slenderness (KL/r)", f"{kl_r:.1f}")
    metric3.metric("Max Capacity (ΦPn)", f"{phi_pn_max:.1f} ton")
    metric4.metric("Demand/Capacity", f"{Pu/phi_pn_max:.2f}" if phi_pn_max > 0 else "N/A")

    # แจ้งเตือนเรื่องเปอร์เซ็นต์เหล็ก
    if rho_pct < 1.0:
        st.warning(f"⚠️ **Reinforcement Warning:** อัตราส่วนเหล็กเสริม ({rho_pct:.2f}%) น้อยกว่าเกณฑ์ขั้นต่ำ 1% ตามมาตรฐาน ACI")
    elif rho_pct > 8.0:
        st.error(f"❌ **Reinforcement Error:** อัตราส่วนเหล็กเสริม ({rho_pct:.2f}%) มากกว่าเกณฑ์สูงสุด 8% ตามมาตรฐาน ACI (แนะนำให้ไม่เกิน 4% สำหรับทางปฏิบัติ)")
    else:
        st.success(f"✅ **Reinforcement:** อัตราส่วนเหล็กเสริม {rho_pct:.2f}% (ผ่านเกณฑ์ 1% - 8%)")

    # แจ้งเตือนเรื่องความชะลูด
    if kl_r > 22:
        st.warning(f"⚠️ **Slenderness Warning:** ค่า KL/r = {kl_r:.1f} (> 22) หน้าตัดนี้อาจเป็นเสายาว (Slender Column) ควรพิจารณา Moment Magnification (P-Delta Effect)")
    else:
        st.info(f"✅ **Slenderness:** ค่า KL/r = {kl_r:.1f} (อยู่ในเกณฑ์เสาสั้น)")

    # สรุปผลความปลอดภัยของ P-M
    if is_safe:
        st.success(f"✅ **CAPACITY CHECK:** จุดทำงาน (Pu={Pu} ton, Mu={Mu} ton-m) ปลอดภัย (Inside Envelope)")
    else:
        st.error(f"❌ **CAPACITY CHECK:** จุดทำงาน (Pu={Pu} ton, Mu={Mu} ton-m) ไม่ปลอดภัย (Outside Envelope)")

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 P-M Curve", "📈 K-R Curve", "🗄️ Detailed Data"])

    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df['Mn'], y=df['Pn'], name="Nominal Capacity", line=dict(color='gray', dash='dash')))
        fig1.add_trace(go.Scatter(x=df_design['phiMn'], y=df_design['phiPn'], fill='tozeroy', name="Design Capacity (Φ)", line=dict(color='navy', width=3)))
        fig1.add_trace(go.Scatter(x=[Mu], y=[Pu], mode='markers', name="Demand (Mu, Pu)", marker=dict(color='red', size=12, symbol='x')))
        fig1.update_layout(xaxis_title="Moment, M (ton-m)", yaxis_title="Axial Load, P (ton)", plot_bgcolor='white', height=500)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        Ku = Pu / fc_Ag_ton
        Ru = Mu / (fc_Ag_ton * (h / 100))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Rn'], y=df['Kn'], name="Nominal (Kn-Rn)", line=dict(color='gray', dash='dash')))
        fig2.add_trace(go.Scatter(x=df_design['phiRn'], y=df_design['phiKn'], fill='tozeroy', name="Design (ΦKn-ΦRn)", line=dict(color='teal', width=3)))
        fig2.add_trace(go.Scatter(x=[Ru], y=[Ku], mode='markers', name="Demand (Ru, Ku)", marker=dict(color='red', size=12, symbol='x')))
        fig2.update_layout(xaxis_title="Normalized Moment, R (M / f'c·Ag·h)", yaxis_title="Normalized Axial, K (P / f'c·Ag)", plot_bgcolor='white', height=500)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### 🧮 Calculation Table")
        display_df = df_design[['c', 'et', 'phi', 'Pn', 'Mn', 'phiPn', 'phiMn']].copy()
        display_df.columns = ['c (cm)', 'Strain (et)', 'Phi (Φ)', 'Pn (ton)', 'Mn (ton-m)', 'ΦPn', 'ΦMn']
        st.dataframe(display_df.style.format("{:.4f}"), use_container_width=True)
