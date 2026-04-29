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
        
        # คอนกรีต Modulus of Elasticity (kg/cm2)
        self.Ec = 15100 * np.sqrt(self.fc)
        self.Ig = (self.b * self.h**3) / 12  # Moment of Inertia (cm4)
        
        # จัดเลเยอร์เหล็ก (สมมติจัด 2 ฝั่ง)
        self.db_cm = db_mm / 10
        self.as_single = (np.pi * self.db_cm**2) / 4
        self.d_prime = cover_cm + 0.9 + (self.db_cm/2)
        self.d = h - self.d_prime                     
        
        self.n_bars = n_bars
        self.layers = [
            {'as': (n_bars/2) * self.as_single, 'd': self.d_prime},
            {'as': (n_bars/2) * self.as_single, 'd': self.d}
        ]
        
        self.total_as = sum(l['as'] for l in self.layers)
        self.rho = self.total_as / self.Ag 
        self.dt = max(l['d'] for l in self.layers)

    def solve_pm(self):
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

            pn = (Cc + Pn_s) / 1000 
            mn = (Mc + Mn_s) / 100000 
            
            et = 0.003 * (self.dt - c) / c
            ey = self.fy / self.Es
            
            if et >= 0.005: phi = 0.90
            elif et <= ey: phi = 0.65
            else: phi = 0.65 + 0.25 * (et - ey) / (0.005 - ey)
            
            results.append({'c': c, 'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn})

        tn = -self.total_as * self.fy / 1000
        results.append({'c': 0, 'Pn': tn, 'Mn': 0, 'phiPn': 0.9 * tn, 'phiMn': 0})

        po = (0.85 * self.fc * (self.Ag - self.total_as) + self.fy * self.total_as) / 1000
        phi_pn_max = 0.65 * 0.80 * po

        return pd.DataFrame(results).sort_values('Pn', ascending=True), phi_pn_max
        
    def calculate_slenderness(self, Pu, Mu, K_factor, Lu_m, Cm, beta_d):
        Lu_cm = Lu_m * 100
        r = 0.3 * self.h # รัศมีไจเรชัน ACI
        kl_r = (K_factor * Lu_cm) / r
        
        # 0.4 Ec Ig / (1 + beta_d)
        EI = (0.4 * self.Ec * self.Ig) / (1 + beta_d)
        
        # Euler Buckling Load (kg -> ton)
        Pc = (np.pi**2 * EI) / (K_factor * Lu_cm)**2 / 1000
        
        # คำนวณ Magnification Factor (Delta)
        phi_k = 0.75 # ACI Stiffness reduction
        if Pu >= (phi_k * Pc):
            delta = 999.9 # พังจากการโก่งเดาะ (Buckling Failure)
        else:
            delta = max(1.0, Cm / (1 - (Pu / (phi_k * Pc))))
            
        Mc = delta * Mu
        return kl_r, Pc, delta, Mc

# --- FUNCTION วาดรูปหน้าตัด ---
def plot_cross_section(engine):
    fig = go.Figure()
    # วาดกรอบคอนกรีต
    fig.add_trace(go.Scatter(x=[0, engine.b, engine.b, 0, 0], 
                             y=[0, 0, engine.h, engine.h, 0], 
                             mode='lines', name='Concrete Edge', line=dict(color='black', width=2)))
    
    # คำนวณพิกัดเหล็กเสริม
    x_coords = np.linspace(engine.d_prime, engine.b - engine.d_prime, int(engine.n_bars/2))
    
    # เหล็กชั้นล่าง (d)
    fig.add_trace(go.Scatter(x=x_coords, y=[engine.h - engine.d]*len(x_coords), 
                             mode='markers', name='Bottom Rebars', marker=dict(color='red', size=10)))
    # เหล็กชั้นบน (d_prime)
    fig.add_trace(go.Scatter(x=x_coords, y=[engine.h - engine.d_prime]*len(x_coords), 
                             mode='markers', name='Top Rebars', marker=dict(color='blue', size=10)))
    
    fig.update_layout(xaxis_title="Width, b (cm)", yaxis_title="Depth, h (cm)",
                      yaxis=dict(scaleanchor="x", scaleratio=1), # ล็อกสัดส่วนให้เป็นจริง
                      plot_bgcolor='whitesmoke', height=400, showlegend=False)
    return fig

# --- STREAMLIT UI ---
st.set_page_config(page_title="Ultimate RC Column", layout="wide")
st.title("🏗️ Ultimate RC Column (With Slenderness Effects)")

col1, col2 = st.columns([1, 2.5])

with col1:
    with st.expander("1. Section & Reinforcement", expanded=True):
        fc = st.number_input("f'c (ksc)", value=280)
        fy = st.number_input("fy (ksc)", value=4000)
        b = st.number_input("Width, b (cm)", value=40)
        h = st.number_input("Depth, h (cm)", value=60)
        db = st.selectbox("Bar Size (mm)", [16, 20, 25, 28, 32], index=2)
        n_bars = st.number_input("Total Bars (Even)", 4, 40, 8, step=2)
        cover = st.number_input("Covering (cm)", value=4.0)

    with st.expander("2. Loads & Slenderness", expanded=True):
        Pu = st.number_input("Factored Axial, Pu (ton)", value=150.0)
        Mu = st.number_input("Factored Moment, Mu (ton-m)", value=15.0)
        st.markdown("---")
        K_factor = st.number_input("Effective Length Factor (K)", value=1.0, step=0.1)
        Lu = st.number_input("Unsupported Length, Lu (m)", value=4.0, step=0.5)
        Cm = st.number_input("Cm factor (1.0 for single curvature)", value=1.0, step=0.1)
        beta_d = st.slider("Beta_d (Sustained Load Ratio)", 0.0, 1.0, 0.6)

engine = RCColumnProfessional(fc, fy, b, h, db, n_bars, cover)
df, phi_pn_max = engine.solve_pm()

# คำนวณเสายาว
kl_r, Pc, delta, Mc = engine.calculate_slenderness(Pu, Mu, K_factor, Lu, Cm, beta_d)

df_design = df.copy()
df_design['phiPn'] = df_design['phiPn'].clip(upper=phi_pn_max)

try:
    interp_func = interp1d(df_design['phiPn'], df_design['phiMn'], kind='linear', fill_value=0, bounds_error=False)
    max_Mu_allowable = interp_func(Pu)
    is_safe = (Mc <= max_Mu_allowable) and (Pu <= phi_pn_max) and (Pu >= df_design['phiPn'].min())
except:
    is_safe = False

with col2:
    # --- Status Board ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Steel Ratio (ρ)", f"{engine.rho*100:.2f} %")
    m2.metric("Slenderness (KL/r)", f"{kl_r:.1f}")
    m3.metric("Critical Load (Pc)", f"{Pc:.1f} ton")
    m4.metric("Magnifier (δ)", f"{delta:.3f}")

    if kl_r > 22:
        st.warning(f"⚠️ **Slender Column:** KL/r = {kl_r:.1f} > 22. จำเป็นต้องขยายโมเมนต์ (Moment Magnified).")
        if delta > 1.0:
            st.info(f"🔄 โมเมนต์ถูกขยายจาก **{Mu} ton-m** เป็น **{Mc:.2f} ton-m** (เพิ่มขึ้น {(delta-1)*100:.1f}%)")
    else:
        st.success(f"✅ **Short Column:** KL/r = {kl_r:.1f} ≤ 22 (ไม่ต้องขยายโมเมนต์)")

    if is_safe:
        st.success(f"✅ **SAFE:** จุดทำงาน (Pu={Pu} t, Mc={Mc:.1f} t-m) อยู่ในโค้งความสามารถ (Inside Envelope)")
    else:
        st.error(f"❌ **UNSAFE:** จุดทำงาน (Pu={Pu} t, Mc={Mc:.1f} t-m) เกินขีดจำกัด (Outside Envelope)")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 P-M Curve", "📐 Section", "📚 K-Factor", "📝 รายการคำนวณ (Report)"])

    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_design['phiMn'], y=df_design['phiPn'], fill='tozeroy', name="Design Capacity (Φ)", line=dict(color='navy', width=3)))
        fig1.add_trace(go.Scatter(x=[Mu], y=[Pu], mode='markers', name="Original (Mu, Pu)", marker=dict(color='orange', size=10, symbol='circle')))
        
        if delta > 1.0 and delta < 999:
            fig1.add_trace(go.Scatter(x=[Mc], y=[Pu], mode='markers', name="Magnified (Mc, Pu)", marker=dict(color='red', size=12, symbol='x')))
            fig1.add_annotation(x=Mc, y=Pu, ax=Mu, ay=Pu, xref="x", yref="y", axref="x", ayref="y", 
                                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="red")
            
        fig1.update_layout(xaxis_title="Moment, M (ton-m)", yaxis_title="Axial Load, P (ton)", plot_bgcolor='white', height=500)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.markdown("### รูปร่างหน้าตัดและการจัดเรียงเหล็ก (Cross-Section)")
        fig_sec = plot_cross_section(engine)
        st.plotly_chart(fig_sec, use_container_width=True)

    with tab3:
        st.markdown("### 📚 ตารางแนะนำค่า K (Effective Length Factor)")
        st.markdown("""
        | สภาพการยึดรั้งปลายเสา (End Conditions) | ค่า K ทางทฤษฎี | ค่า K แนะนำสำหรับออกแบบ |
        | :--- | :---: | :---: |
        | **ยึดแน่น-ยึดแน่น (Fixed-Fixed)** | 0.50 | **0.65** |
        | **ยึดแน่น-หมุนได้ (Fixed-Pinned)** | 0.70 | **0.80** |
        | **หมุนได้-หมุนได้ (Pinned-Pinned)** | 1.00 | **1.00** |
        | **ยึดแน่น-อิสระ (Fixed-Free)** | 2.00 | **2.10** |
        """)

    with tab4:
        st.markdown("## 📝 Detailed Calculation Report")
        st.markdown("This automated report summarizes the structural design, slenderness evaluation, and capacity checks for the reinforced concrete column according to ACI standards.")
        st.markdown("---")
        
        st.markdown("#### 1. Section & Material Properties")
        st.markdown(f"- **Dimensions:** Width $b = {b}$ cm, Depth $h = {h}$ cm")
        st.markdown(f"- **Gross Area ($A_g$):** `{engine.Ag:,.2f}` cm²")
        st.markdown(f"- **Concrete Compressive Strength ($f'_c$):** `{fc}` ksc")
        st.markdown(f"- **Steel Yield Strength ($f_y$):** `{fy}` ksc")
        
        st.markdown("#### 2. Reinforcement Details")
        st.markdown(f"- **Bar Arrangement:** `{n_bars}`-DB`{db}`")
        st.markdown(f"- **Concrete Covering:** `{cover}` cm")
        st.markdown(f"- **Total Steel Area ($A_{{st}}$):** `{engine.total_as:.2f}` cm²")
        st.markdown(f"- **Reinforcement Ratio ($\\rho$):** `{engine.rho*100:.2f}`% (ACI Limits: 1% - 8%)")
        
        st.markdown("#### 3. Slenderness Effect Evaluation")
        st.markdown(f"- **Unsupported Length ($L_u$):** `{Lu}` m")
        st.markdown(f"- **Effective Length Factor ($K$):** `{K_factor}`")
        st.markdown(f"- **Radius of Gyration ($r \\approx 0.3h$):** `{0.3*h:.2f}` cm")
        st.markdown(f"- **Slenderness Ratio ($KL/r$):** `{kl_r:.2f}`")
        
        if kl_r <= 22:
            st.success("✅ **Conclusion:** Since $KL/r \\le 22$, the column is classified as a **Short Column**. Moment magnification is NOT required.")
            st.markdown("#### 4. Capacity Check Summary")
        else:
            st.warning("⚠️ **Conclusion:** Since $KL/r > 22$, the column is classified as a **Slender Column**. Moment magnification is REQUIRED.")
            
            st.markdown("#### 4. Moment Magnification Method (Non-Sway Frame)")
            st.markdown(f"- **Concrete Modulus of Elasticity ($E_c = 15100\\sqrt{{f'_c}}$):** `{engine.Ec:,.0f}` ksc")
            st.markdown(f"- **Gross Moment of Inertia ($I_g = bh^3/12$):** `{engine.Ig:,.0f}` cm⁴")
            
            EI_val = (0.4 * engine.Ec * engine.Ig) / (1 + beta_d)
            st.markdown(f"- **Effective Flexural Stiffness ($EI$):** `{EI_val:,.0f}` kg-cm²")
            st.markdown(f"- **Euler Critical Buckling Load ($P_c = \\pi^2 EI / (KL_u)^2$):** `{Pc:.2f}` ton")
            st.markdown(f"- **Moment Magnification Factor ($\\delta$):** `{delta:.3f}`")
            st.info(f"🔄 **Magnified Design Moment ($M_c = \\delta M_u$):** `{Mc:.2f}` ton-m")
            
            st.markdown("#### 5. Capacity Check Summary")

        st.markdown(f"- **Applied Factored Axial Load ($P_u$):** `{Pu}` ton")
        st.markdown(f"- **Design Moment Demand ($M_c$):** `{Mc:.2f}` ton-m")
        st.markdown(f"- **Maximum Compressive Capacity ($\\phi P_{{n,max}}$):** `{phi_pn_max:.2f}` ton")
        
        if is_safe:
            st.success("🎯 **STATUS: SAFE** — The applied demand ($P_u, M_c$) is strictly within the interaction diagram envelope.")
        else:
            st.error("❌ **STATUS: UNSAFE** — The applied demand ($P_u, M_c$) exceeds the structural capacity of the section.")
            
        st.markdown("---")
        st.caption("Press `Ctrl + P` (or `Cmd + P` on Mac) to print or save this calculation report as a PDF.")
