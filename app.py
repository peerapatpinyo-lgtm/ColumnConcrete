import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Calculation Engine (ขยายจากฐานเดิมของคุณ) ---
class RCCapacityEngine:
    def __init__(self, fc, fy, b, h, db, n_bars, cover, L_cm=0, k=1.0):
        self.fc, self.fy, self.b, self.h = fc, fy, b, h
        self.L, self.k = L_cm, k
        self.es = 2040000  # ksc
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))
        
        # เหล็กเสริม
        self.as_total = n_bars * (np.pi * (db/10)**2 / 4)
        self.d_prime = cover + 0.9 + (db/20)
        self.d = h - self.d_prime
        self.x_pc = h / 2 

    def analyze_section(self):
        results = []
        po = (0.85 * self.fc * (self.b * self.h - self.as_total) + self.fy * self.as_total) / 1000
        phi_pn_max = 0.65 * 0.80 * po 
        
        c_list = [1e10] + list(np.geomspace(self.h * 2, 0.1, 100)) + [0]
        
        for c in c_list:
            if c >= 1e10:
                pn, mn, et = po, 0, 0
            elif c == 0:
                pn, mn, et = -self.as_total * self.fy / 1000, 0, 1.0
            else:
                a = min(self.beta1 * c, self.h)
                cc = 0.85 * self.fc * a * self.b
                eps_s1 = 0.003 * (c - self.d_prime) / c
                eps_t = 0.003 * (self.d - c) / c
                fs1 = np.clip(eps_s1 * self.es, -self.fy, self.fy)
                fs2 = np.clip(eps_t * self.es, -self.fy, self.fy)
                
                pn = (cc + (self.as_total/2 * fs1) + (self.as_total/2 * fs2)) / 1000
                mn = (cc * (self.x_pc - a/2) + (self.as_total/2 * fs1) * (self.x_pc - self.d_prime) - 
                      (self.as_total/2 * fs2) * (self.d - self.x_pc)) / 100000
                et = abs(eps_t) if eps_t < 0 else 0

            ey = self.fy / self.es
            phi = 0.65 if et <= ey else 0.90 if et >= 0.005 else 0.65 + 0.25*(et - ey)/(0.005 - ey)
            results.append({'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn, 'et': et})

        return pd.DataFrame(results), phi_pn_max

    def check_slenderness(self, Pu_ton, M2_tonm):
        """วิเคราะห์เสายาว (Moment Magnification)"""
        if self.L == 0: return M2_tonm
        r = 0.3 * self.h
        slenderness = (self.k * self.L) / r
        if slenderness <= 22: # กรณีเสาสั้น
            return M2_tonm
        
        # คำนวณ EI (Simplified)
        EI = (0.4 * self.fc * (self.b * self.h**3 / 12)) / (1 + 0.5) # 0.5 คือ beta_dns สมมติ
        pc = (np.pi**2 * EI) / (self.k * self.L)**2 / 1000 # Critical load (ton)
        delta_ns = max(1.0, 1.0 / (1 - (Pu_ton / (0.75 * pc))))
        return M2_tonm * delta_ns

def corbel_design(Vu, av, b, d, fc, fy):
    """คำนวณหูช้างเบื้องต้น (Shear Friction Theory)"""
    # เช็คขนาดหน้าตัดเบื้องต้น (Vn <= 0.2*fc*b*d)
    vn_max = 0.2 * fc * b * d / 1000
    phi_v = 0.75
    av_rebar = (Vu / (phi_v * fy * 1.0)) * 1000 # cm2 (Simplified)
    return vn_max, av_rebar

# --- UI Layout ---
st.set_page_config(page_title="Industrial RC Column Expert", layout="wide")
st.title("🏭 Industrial RC Column Designer (Pro)")

with st.sidebar:
    st.header("1. Material & Section")
    fc = st.number_input("f'c (ksc)", 210, 560, 280)
    fy = st.number_input("fy (ksc)", 3000, 5000, 4000)
    b = st.slider("Width b (cm)", 20, 100, 40)
    h = st.slider("Depth h (cm)", 20, 100, 50)
    
    st.header("2. Reinforcement")
    db = st.selectbox("DB Size (mm)", [16, 20, 25, 28, 32], index=2)
    n_bars = st.number_input("Number of Bars", 4, 32, 8, step=2)
    cover = 4.0
    
    st.header("3. Length & Boundary")
    L_m = st.number_input("Column Height (m)", 0.0, 15.0, 4.0)
    k_factor = st.selectbox("K Factor (Condition)", [0.7, 1.0, 1.2, 2.0], index=1)

# --- Main App Tabs ---
tab1, tab2 = st.tabs(["📊 P-M Analysis", "🏗️ Corbel & Detailing"])

engine = RCCapacityEngine(fc, fy, b, h, db, n_bars, cover, L_m*100, k_factor)
df_pm, p_cap = engine.analyze_section()

with tab1:
    st.subheader("Interactive P-M Diagram with Load Cases")
    
    # Input Load Cases
    st.write("ป้อนแรงที่กระทำ (Load Cases):")
    load_data = pd.DataFrame([{'Case': 'Crane Max', 'Pu (ton)': 120.0, 'Mu (ton-m)': 15.0}], index=[0])
    edited_loads = st.data_editor(load_data, num_rows="dynamic")
    
    # Process Magnified Moments
    processed_loads = []
    for _, row in edited_loads.iterrows():
        mu_mag = engine.check_slenderness(row['Pu (ton)'], row['Mu (ton-m)'])
        processed_loads.append({'Case': row['Case'], 'Pu': row['Pu (ton)'], 'Mu': mu_mag})
    df_loads = pd.DataFrame(processed_loads)

    # Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pm['phiMn'], y=df_pm['phiPn'].clip(upper=p_cap), 
                             fill='tozeroy', name="Design Capacity", line=dict(color='navy', width=3)))
    
    # Plot Load Points
    if not df_loads.empty:
        fig.add_trace(go.Scatter(x=df_loads['Mu'], y=df_loads['Pu'], mode='markers+text',
                                 text=df_loads['Case'], textposition="top right",
                                 marker=dict(color='red', size=10), name="Design Loads"))

    fig.update_layout(xaxis_title="Moment (ton-m)", yaxis_title="Axial Load (ton)", height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Corbel (หูช้าง) Design")
    c1, c2 = st.columns(2)
    with c1:
        vu_corbel = st.number_input("แรงเฉือนที่หูช้าง Vu (ton)", 0.0, 100.0, 20.0)
        av_dist = st.number_input("ระยะยื่น av (cm)", 5.0, 50.0, 20.0)
    
    v_max, a_rebar = corbel_design(vu_corbel, av_dist, b, h-cover, fc, fy)
    
    with c2:
        st.metric("Max Capacity (ton)", f"{v_max:.2f}")
        st.metric("Req. Steel (cm2)", f"{a_rebar:.2f}")
        if vu_corbel > v_max:
            st.error("วิบัติ! กรุณาเพิ่มขนาดหน้าตัดหูช้าง")
        else:
            st.success("ขนาดหน้าตัดหูช้างเพียงพอ")

st.divider()
st.caption("Developed for Professional Engineers | GitHub: your-repo-name")
