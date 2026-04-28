import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. ADVANCED CALCULATION ENGINE
# ==========================================

class RCProDesigner:
    def __init__(self, fc, fy, b, h, db, n_bars, cover):
        self.fc, self.fy, self.b, self.h = fc, fy, b, h
        self.es = 2040000  # ksc
        self.ec = 15100 * np.sqrt(fc) # ksc
        self.beta1 = max(0.65, min(0.85, 0.85 - (0.05 * (fc - 280) / 70)))
        
        # Reinforcement Layers (Top & Bottom)
        self.as_total = n_bars * (np.pi * (db/10)**2 / 4)
        self.d_prime = cover + 0.9 + (db/20) # cm
        self.d = h - self.d_prime
        self.ig = (b * h**3) / 12 # Gross Moment of Inertia (cm4)

    def get_interaction_data(self):
        """คำนวณพิกัด P-M สำหรับสร้าง Interaction Diagram"""
        data = []
        # 1. Pure Compression Limit (ACI 318)
        po = (0.85 * self.fc * (self.b * self.h - self.as_total) + self.fy * self.as_total) / 1000
        phi_pn_max = 0.65 * 0.80 * po # Tied Column
        
        # 2. Variable Neutral Axis (c) sweep
        c_list = np.logspace(np.log10(0.1), np.log10(self.h * 10), 150)[::-1]
        for c in c_list:
            a = min(self.beta1 * c, self.h)
            cc = 0.85 * self.fc * a * self.b
            
            # Strain Compatibility
            eps_s1 = 0.003 * (c - self.d_prime) / c
            eps_t = 0.003 * (self.d - c) / c
            
            fs1 = max(-self.fy, min(self.fy, eps_s1 * self.es))
            fs2 = max(-self.fy, min(self.fy, eps_t * self.es))
            
            pn = (cc + (self.as_total/2 * fs1) + (self.as_total/2 * fs2)) / 1000
            mn = (cc * (self.h/2 - a/2) + (self.as_total/2 * fs1) * (self.h/2 - self.d_prime) - 
                  (self.as_total/2 * fs2) * (self.d - self.h/2)) / 100000
            
            # Strength Reduction Factor (Phi)
            ey = self.fy / self.es
            eps_t_abs = abs(eps_t) if eps_t < 0 else 0
            phi = 0.65 if eps_t_abs <= ey else 0.90 if eps_t_abs >= 0.005 else 0.65 + 0.25*(eps_t_abs - ey)/(0.005 - ey)
            
            data.append({'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn})
            
        return pd.DataFrame(data), phi_pn_max

    def check_slenderness(self, pu, mu, k, l_m):
        """วิเคราะห์เสาสั้น/เสายาว และขยายโมเมนต์ (Moment Magnification)"""
        lu = l_m * 100 # cm
        r = 0.3 * self.h # Radius of gyration for rectangular
        kl_r = (k * lu) / r
        
        # ACI limit for non-sway frames: kl/r < 22 (conservative)
        is_slender = kl_r > 22
        
        if not is_slender:
            return 1.0, kl_r, mu, "Short Column"
        
        # Slender Column - Moment Magnification
        # EI = 0.4 * Ec * Ig / (1 + beta_dns); beta_dns assume 0.5 (sustained load ratio)
        ei = (0.4 * self.ec * self.ig) / (1 + 0.5)
        pc = (np.pi**2 * ei) / (k * lu)**2 / 1000 # Euler Buckling Load (Tons)
        
        # Magnification factor (delta_ns)
        cm = 1.0 # Standard for columns with transverse loads or unknown M1/M2
        delta_ns = cm / (1 - (pu / (0.75 * pc)))
        delta_ns = max(delta_ns, 1.0)
        
        return delta_ns, kl_r, delta_ns * mu, "Slender Column"

# ==========================================
# 2. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Engineer RC Pro", layout="wide")
st.title("🏗️ Professional RC Column Design & Slenderness Analysis")
st.caption("Standard: ACI 318-19 Strength Design Method")

# Sidebar Inputs
with st.sidebar:
    st.header("🛠️ Parameters")
    fc = st.number_input("f'c (ksc)", 210, 560, 280)
    fy = st.number_input("fy (ksc)", 3000, 5000, 4000)
    b = st.slider("Width b (cm)", 20, 100, 40)
    h = st.slider("Depth h (cm)", 20, 100, 50)
    db = st.selectbox("Bar Size (mm)", [12, 16, 20, 25, 28, 32], index=3)
    n_bars = st.number_input("Bars (Must be even)", 4, 32, 8, step=2)
    st.markdown("---")
    pu_req = st.number_input("Pu (tons)", 0.0, 1000.0, 100.0)
    mu_req = st.number_input("Mu (ton-m)", 0.0, 500.0, 15.0)
    l_m = st.number_input("Clear Height L (m)", 1.0, 15.0, 5.0)
    k_val = st.slider("Effective Length (k)", 0.5, 2.1, 1.0)

# Calculation
designer = RCProDesigner(fc, fy, b, h, db, n_bars, 4.0)
df_pm, phi_pn_max = designer.get_interaction_data()
delta, klr, mc, col_type = designer.check_slenderness(pu_req, mu_req, k_val, l_m)

# Main Display
c1, c2 = st.columns([2, 1])

with c1:
    fig = go.Figure()
    # Design Curve
    fig.add_trace(go.Scatter(x=df_pm['phiMn'], y=df_pm['phiPn'].clip(upper=phi_pn_max), 
                             fill='tozeroy', name="Design (ΦPn-ΦMn)", line=dict(color='blue', width=3)))
    # Nominal Curve
    fig.add_trace(go.Scatter(x=df_pm['Mn'], y=df_pm['Pn'], name="Nominal (Pn-Mn)", line=dict(color='gray', dash='dash')))
    # Design Points
    fig.add_trace(go.Scatter(x=[mu_req], y=[pu_req], mode='markers', name="Input Load", marker=dict(color='orange', size=10)))
    fig.add_trace(go.Scatter(x=[mc], y=[pu_req], mode='markers+text', text=["Design Point"], textposition="top center",
                             name="Magnified Load (Mc)", marker=dict(color='red', size=14, symbol='diamond')))
    
    fig.update_layout(xaxis_title="Moment (ton-m)", yaxis_title="Axial Load (tons)", height=650)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("📊 Results Summary")
    # Safety Check
    phi_pn_at_mc = np.interp(mc, df_pm['phiMn'], df_pm['phiPn'].clip(upper=phi_pn_max))
    is_safe = (pu_req <= phi_pn_at_mc) and (pu_req <= phi_pn_max)
    
    res_color = "green" if is_safe else "red"
    st.markdown(f"### Status: :{res_color}[{'PASS' if is_safe else 'FAIL'}]")
    
    st.write(f"**Classification:** {col_type}")
    st.metric("Slenderness Ratio (kL/r)", f"{klr:.2f}")
    st.metric("Magnifier (δ_ns)", f"{delta:.2f}")
    st.metric("Design Moment (Mc)", f"{mc:.2f} ton-m")
    
    with st.expander("Section Specs"):
        st.write(f"Steel Area: {designer.as_total:.2f} cm²")
        st.write(f"Steel Ratio: {(designer.as_total/(b*h)*100):.2f}%")
