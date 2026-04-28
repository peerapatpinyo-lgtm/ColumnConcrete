import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ==========================================
# 1. ENGINEERING CALCULATION ENGINE (PRO)
# ==========================================

class ColumnEngine:
    def __init__(self, fc, fy, b, h, db, n_bars, cover):
        self.fc, self.fy, self.b, self.h = fc, fy, b, h
        self.es = 2.04e6  # ksc
        self.ec = 15100 * np.sqrt(fc) # ksc (ACI)
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))
        
        # Reinforcement Data
        self.as_bar = (np.pi * (db/10)**2) / 4 
        self.ast = n_bars * self.as_bar
        self.d_prime = cover + 0.9 + (db/20)
        self.d = h - self.d_prime
        self.ig = (b * h**3) / 12 # Gross Moment of Inertia
        
    def calculate_interaction(self):
        """สร้าง Interaction Diagram แบบ High-Resolution (100+ points)"""
        points = []
        # 1. Pure Compression (Capped)
        po = (0.85 * self.fc * (self.b * self.h - self.ast) + self.fy * self.ast) / 1000
        phi_pn_max = 0.65 * 0.80 * po
        
        # 2. Sweep Neutral Axis (c)
        c_steps = np.logspace(np.log10(0.1), np.log10(self.h * 10), 150)[::-1]
        for c in c_steps:
            a = min(self.beta1 * c, self.h)
            cc = 0.85 * self.fc * a * self.b
            
            # Strain & Stress in 2 layers
            eps_s1 = 0.003 * (c - self.d_prime) / c
            eps_t = 0.003 * (self.d - c) / c
            fs1 = max(-self.fy, min(self.fy, eps_s1 * self.es))
            fs2 = max(-self.fy, min(self.fy, eps_t * self.es))
            
            pn = (cc + (self.ast/2 * fs1) + (self.ast/2 * fs2)) / 1000
            mn = (cc * (self.h/2 - a/2) + (self.ast/2 * fs1) * (self.h/2 - self.d_prime) - (self.ast/2 * fs2) * (self.d - self.h/2)) / 100000
            
            # Phi Factor
            ey = self.fy / self.es
            eps_t_abs = abs(eps_t) if eps_t < 0 else 0
            phi = 0.65 if eps_t_abs <= ey else 0.90 if eps_t_abs >= 0.005 else 0.65 + 0.25*(eps_t_abs - ey)/(0.005 - ey)
            
            points.append({'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn})
            
        return pd.DataFrame(points), phi_pn_max

    def get_slenderness_magnifier(self, pu, mu, k, l_m, dns_ratio=0.5):
        """
        คำนวณการขยายโมเมนต์ (Moment Magnification) สำหรับเสาไม่ค้ำยัน (Non-sway)
        l_m: ความยาวเสา (เมตร)
        dns_ratio: Ratio of sustained load (beta_dns)
        """
        lu = l_m * 100 # cm
        r = 0.3 * self.h
        slenderness_ratio = (k * lu) / r
        
        # ACI Threshold for Short Column (Non-sway)
        # M1/M2 assume 1.0 (Worst case for simplicity)
        is_slender = slenderness_ratio > 22 
        
        if not is_slender:
            return 1.0, slenderness_ratio, mu, "Short Column"
        
        # Calculate EI (Simplified ACI: EI = 0.4 * Ec * Ig / (1 + beta_dns))
        ei = (0.4 * self.ec * self.ig) / (1 + dns_ratio)
        # Euler Buckling Load (Pc)
        pc = (np.pi**2 * ei) / (k * lu)**2 / 1000 # Tons
        
        # Magnifier (delta_ns)
        cm = 1.0 # Assume 1.0 for columns with transverse loads or end moments
        delta_ns = cm / (1 - (pu / (0.75 * pc)))
        delta_ns = max(delta_ns, 1.0)
        
        mc = delta_ns * mu
        return delta_ns, slenderness_ratio, mc, "Slender Column"

# ==========================================
# 2. STREAMLIT UI (PRO)
# ==========================================

st.set_page_config(page_title="Advanced RC Column Designer", layout="wide")
st.title("🛡️ Professional Industrial RC Column Designer")
st.markdown("### ACI 318-19 / WST SDM Standard")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("1. Materials")
    fc = st.number_input("Concrete f'c (ksc)", 210, 560, 280)
    fy = st.number_input("Steel fy (ksc)", 3000, 5000, 4000)
    
    st.header("2. Geometry & Reinforcement")
    b = st.slider("Width b (cm)", 20, 100, 40)
    h = st.slider("Depth h (cm)", 20, 100, 50)
    db = st.selectbox("Main Bar (mm)", [12, 16, 20, 25, 28, 32], index=3)
    n_bars = st.number_input("Number of Bars", 4, 32, 8, step=2)
    
    st.header("3. Loads & Slenderness")
    pu_req = st.number_input("Axial Load Pu (tons)", 0.0, 1000.0, 100.0)
    mu_req = st.number_input("Moment Mu (ton-m)", 0.0, 500.0, 15.0)
    l_clear = st.number_input("Clear Height L (m)", 1.0, 15.0, 5.0)
    k_factor = st.slider("Effective Length Factor (k)", 0.5, 2.1, 1.0)

# --- CALCULATION ---
engine = ColumnEngine(fc, fy, b, h, db, n_bars, 4.0)
df_pm, pn_cap = engine.calculate_interaction()
delta, kl_r, mc, col_type = engine.get_slenderness_magnifier(pu_req, mu_req, k_factor, l_clear)

# --- DISPLAY ---
col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    st.subheader("📊 Interaction Diagram")
    fig = go.Figure()
    # Nominal Curve
    fig.add_trace(go.Scatter(x=df_pm['Mn'], y=df_pm['Pn'], name="Nominal (Pn-Mn)", line=dict(color='gray', dash='dash')))
    # Design Curve (Capped)
    fig.add_trace(go.Scatter(x=df_res := df_pm['phiMn'], y=df_pm['phiPn'].clip(upper=pn_cap), 
                             fill='tozeroy', name="Design (ΦPn-ΦMn)", line=dict(color='blue', width=3)))
    # Design Point (Original & Magnified)
    fig.add_trace(go.Scatter(x=[mu_req], y=[pu_req], mode='markers', name="Original Load", marker=dict(color='orange', size=10)))
    fig.add_trace(go.Scatter(x=[mc], y=[pu_req], mode='markers+text', text=["Design Point"], textposition="top center",
                             name="Magnified Load (Mc)", marker=dict(color='red', size=14, symbol='diamond')))
    
    fig.update_layout(xaxis_title="Moment (ton-m)", yaxis_title="Axial Load (tons)", height=650, template="none")
    st.plotly_chart(fig, use_container_width=True)

with col_main2:
    st.subheader("📋 Engineering Report")
    
    # Validation Logic
    phi_pn_limit = np.interp(mc, df_pm['phiMn'], df_pm['phiPn'].clip(upper=pn_cap))
    is_safe = (pu_req <= phi_pn_limit) and (pu_req <= pn_cap)
    
    status_color = "green" if is_safe else "red"
    st.markdown(f"### Status: :{status_color}[{'PASS' if is_safe else 'FAIL'}]")
    
    st.write("---")
    st.write(f"**Column Type:** {col_type}")
    st.metric("Slenderness Ratio ($kL/r$)", f"{kl_r:.2f}", "Long Column (>22)" if kl_r > 22 else "Short Column")
    st.metric("Moment Magnifier ($\delta_{ns}$)", f"{delta:.2f}")
    st.metric("Design Moment ($M_c$)", f"{mc:.2f} ton-m")
    
    with st.expander("Section Details"):
        st.write(f"Steel Area ($A_{{st}}$): {engine.ast:.2f} $cm^2$")
        st.write(f"Steel Ratio ($\rho$): {(engine.ast/(b*h)*100):.2f}%")
        st.write(f"Plastic Centroid: {h/2} cm")

# --- CORBEL MINI-MODULE ---
st.markdown("---")
if st.checkbox("Show Corbel Design Module"):
    st.subheader("🏗️ Corbel Design (หูช้าง)")
    c1, c2 = st.columns(2)
    vu_c = c1.number_input("Corbel Vu (tons)", value=20.0)
    av_c = c2.number_input("Shear Span av (cm)", value=20.0)
    
    # Simple Check
    phi_v = 0.75
    d_c = h - 5.0
    vn_max = (0.2 * fc * b * d_c) / 1000
    if (vu_c/phi_v) > vn_max:
        st.error(f"Size Insufficient! Max Capacity: {phi_v * vn_max:.2f} tons")
    else:
        avf = (vu_c * 1000) / (phi_v * fy * 1.4)
        st.success(f"Size OK! Required Rebar $A_{{vf}}$: {avf:.2f} $cm^2$")
