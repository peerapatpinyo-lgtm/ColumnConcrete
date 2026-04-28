import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class ProfessionalRCDesign:
    def __init__(self, fc, fy, b, h, db, n_bars, cover):
        # Material Properties (Standard Units: ksc, cm)
        self.fc, self.fy, self.b, self.h = fc, fy, b, h
        self.es = 2040000 
        self.ec = 15100 * np.sqrt(fc)
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))
        
        # Section Properties
        self.as_bar = (np.pi * (db/10)**2) / 4 
        self.ast = n_bars * self.as_bar
        self.rho = self.ast / (b * h)
        self.d_prime = cover + 0.9 + (db/20) # Approx. to bar center
        self.d = h - self.d_prime
        self.ig = (b * h**3) / 12

    def calculate_interaction(self):
        """Standard Strain Compatibility Analysis"""
        pts = []
        # Pure Compression (Po)
        po = (0.85 * self.fc * (self.b * self.h - self.ast) + self.fy * self.ast) / 1000
        phi_pn_max = 0.65 * 0.80 * po
        
        c_vals = np.logspace(np.log10(0.1), np.log10(self.h * 10), 200)[::-1]
        for c in c_vals:
            a = min(self.beta1 * c, self.h)
            cc = 0.85 * self.fc * a * self.b
            
            # Strain Analysis
            eps_cu = 0.003
            eps_s1 = eps_cu * (c - self.d_prime) / c
            eps_t = eps_cu * (self.d - c) / c
            
            fs1 = np.clip(eps_s1 * self.es, -self.fy, self.fy)
            fs2 = np.clip(eps_t * self.es, -self.fy, self.fy)
            
            # Forces and Moments around plastic centroid (h/2)
            pn = (cc + (self.ast/2 * fs1) + (self.ast/2 * fs2)) / 1000
            mn = (cc*(self.h/2 - a/2) + (self.ast/2 * fs1)*(self.h/2 - self.d_prime) - (self.ast/2 * fs2)*(self.d - self.h/2)) / 100000
            
            # Phi Factor
            ey = self.fy / self.es
            et_abs = abs(eps_t) if eps_t < 0 else 0
            phi = 0.65 if et_abs <= ey else 0.90 if et_abs >= 0.005 else 0.65 + 0.25*(et_abs - ey)/(0.005 - ey)
            
            pts.append({'phiPn': phi * pn, 'phiMn': phi * mn, 'Pn': pn, 'Mn': mn})
            
        return pd.DataFrame(pts), phi_pn_max

    def slenderness_analysis(self, pu, mu, l_m, k, m1_m2_ratio=1.0):
        """ACI 318 Moment Magnification Method"""
        lu = l_m * 100 
        r = 0.3 * self.h
        slenderness = (k * lu) / r
        
        # Limit for non-sway frames
        limit = 34 - 12 * (m1_m2_ratio)
        is_slender = slenderness > limit
        
        if not is_slender:
            return 1.0, slenderness, mu, "SHORT"
        
        # EI calculation (Simplified)
        ei = (0.4 * self.ec * self.ig) / (1 + 0.5) # beta_dns = 0.5
        pc = (np.pi**2 * ei) / (k * lu)**2 / 1000
        delta_ns = max(1.0 / (1 - (pu / (0.75 * pc))), 1.0)
        
        return delta_ns, slenderness, delta_ns * mu, "SLENDER"

# --- UI Setup ---
st.set_page_config(page_title="PRO RC DESIGNER", layout="wide")
st.title("🏗️ Structural Verification: RC Column & Corbel")

with st.sidebar:
    st.header("1. Material & Section")
    fc = st.number_input("f'c (ksc)", 210, 560, 280)
    fy = st.number_input("fy (ksc)", 3000, 5000, 4000)
    b, h = st.slider("Width b (cm)", 20, 100, 40), st.slider("Depth h (cm)", 20, 100, 50)
    
    st.header("2. Reinforcement")
    db = st.selectbox("Main Bar (mm)", [16, 20, 25, 28, 32], index=1)
    n_bars = st.number_input("Count", 4, 32, 8, step=2)
    
    st.header("3. Loading & Length")
    pu, mu = st.number_input("Pu (tons)", 0.0, 500.0, 120.0), st.number_input("Mu (ton-m)", 0.0, 100.0, 15.0)
    l_m, k = st.number_input("L (m)", 1.0, 12.0, 6.0), st.slider("k factor", 0.7, 2.1, 1.0)

# Process
design = ProfessionalRCDesign(fc, fy, b, h, db, n_bars, 4.0)
df_pm, phi_pn_max = design.calculate_interaction()
delta, kl_r, mc, col_type = design.slenderness_analysis(pu, mu, l_m, k)

# Validation
max_phi_pn = np.interp(mc, df_pm['phiMn'], df_pm['phiPn'].clip(upper=phi_pn_max))
dcr = pu / max_phi_pn if max_phi_pn > 0 else 9.99

# --- Visuals ---
c1, c2 = st.columns([2, 1])

with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pm['phiMn'], y=df_pm['phiPn'].clip(upper=phi_pn_max), fill='tozeroy', name='Capacity Envelope', line=dict(color='navy')))
    fig.add_trace(go.Scatter(x=[mu, mc], y=[pu, pu], mode='lines+markers', name='Moment Magnification', line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=[mc], y=[pu], marker=dict(size=12, color='red', symbol='diamond'), name='Design Point (Mc)'))
    fig.update_layout(xaxis_title="Moment (ton-m)", yaxis_title="Axial Load (tons)", height=600)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("📋 Engineering Verification")
    st.metric("DCR (Demand/Capacity)", f"{dcr:.2f}", delta=f"{1-dcr:.2%}", delta_color="normal" if dcr < 1 else "inverse")
    
    if dcr <= 1.0 and 0.01 <= design.rho <= 0.08:
        st.success("✅ DESIGN PASS")
    else:
        st.error("❌ DESIGN REJECTED")

    st.write(f"**Classification:** {col_type} ($kL/r = {kl_r:.1f}$)")
    st.write(f"**Magnification Factor ($\delta_{{ns}}$):** {delta:.2f}")
    st.write(f"**Steel Ratio ($\rho$):** {design.rho:.2%}")
    
    with st.expander("Detailed Checks"):
        st.write(f"Min Steel (1%): {'OK' if design.rho >= 0.01 else 'FAIL'}")
        st.write(f"Max Steel (8%): {'OK' if design.rho <= 0.08 else 'FAIL'}")
        st.write(f"Pn max limit: {phi_pn_max:.2f} tons")

st.markdown("---")
st.caption("Disclaimer: ผลลัพธ์นี้ใช้เพื่อการตรวจสอบเบื้องต้นเท่านั้น โปรดใช้ความระมัดระวังในการออกแบบโครงสร้างจริง")
