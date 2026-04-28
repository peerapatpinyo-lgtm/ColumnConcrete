import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. RIGOROUS ENGINEERING ENGINE
# ==========================================

class RCCalculatorV3:
    def __init__(self, fc, fy, b, h, db, n_bars, cover):
        self.fc = fc
        self.fy = fy
        self.b = b
        self.h = h
        self.es = 2040000  # ksc (Modulus of Elasticity)
        self.ey = fy / self.es # Yield Strain
        
        # Geometry & Reinforcement
        self.as_bar = (np.pi * (db/10)**2) / 4 # cm2
        self.ast = n_bars * self.as_bar
        
        # จัดเหล็ก 2 ชั้น (Top/Bottom) สำหรับเสาสี่เหลี่ยมมาตรฐาน
        self.d_prime = cover + 0.9 + (db/20) # cm
        self.d = h - self.d_prime             # cm
        self.as_each_side = self.ast / 2

    def get_beta1(self):
        if self.fc <= 280: return 0.85
        if self.fc >= 560: return 0.65
        return 0.85 - (0.05 * (self.fc - 280) / 70)

    def solve_section(self, c):
        """คำนวณ Pn และ Mn สำหรับระยะ Neutral Axis (c) ใดๆ"""
        beta1 = self.get_beta1()
        a = beta1 * c
        
        # 1. Concrete Force (Cc)
        if c <= 0:
            cc = 0
            ma_c = 0
        else:
            a_eff = min(a, self.h)
            cc = 0.85 * self.fc * a_eff * self.b
            ma_c = cc * (self.h/2 - a_eff/2) # Moment around center
        
        # 2. Steel Forces (Fs) - Strain Compatibility
        # Layer 1 (Compression side)
        eps_s1 = 0.003 * (c - self.d_prime) / c if c != 0 else -0.003
        fs1 = max(-self.fy, min(self.fy, eps_s1 * self.es))
        f_s1_eff = fs1 - 0.85 * self.fc if eps_s1 > 0 and a > self.d_prime else fs1
        force_s1 = self.as_each_side * f_s1_eff
        ma_s1 = force_s1 * (self.h/2 - self.d_prime)

        # Layer 2 (Tension side)
        eps_s2 = 0.003 * (c - self.d) / c if c != 0 else -0.003
        fs2 = max(-self.fy, min(self.fy, eps_s2 * self.es))
        force_s2 = self.as_each_side * fs2
        ma_s2 = force_s2 * (self.h/2 - self.d) # d > h/2, will result in negative moment contribution correctly

        # 3. Summation
        pn = (cc + force_s1 + force_s2) / 1000 # Metric Tons
        mn = (ma_c + ma_s1 + ma_s2) / 100000 # Ton-m
        
        # 4. Phi Factor (ACI 318)
        eps_t = abs(eps_s2) if eps_s2 < 0 else 0
        if eps_t <= self.ey: phi = 0.65
        elif eps_t >= 0.005: phi = 0.90
        else: phi = 0.65 + (0.90 - 0.65) * (eps_t - self.ey) / (0.005 - self.ey)
        
        return pn, mn, phi

    def generate_curve(self):
        points = []
        
        # จุดที่ 1: Pure Compression (Po)
        po = (0.85 * self.fc * (self.b * self.h - self.ast) + self.fy * self.ast) / 1000
        phi_pn_max = 0.65 * 0.80 * po
        points.append({'Pn': po, 'Mn': 0, 'phiPn': phi_pn_max, 'phiMn': 0, 'label': 'Pure Comp'})

        # วนลูปหาจุดบนกราฟ (Neutral Axis sweep)
        c_steps = np.logspace(np.log10(0.1), np.log10(self.h * 5), 200)[::-1]
        for c in c_steps:
            pn, mn, phi = self.solve_section(c)
            if pn < -self.ast * self.fy / 1000: break # Stop at Pure Tension
            points.append({'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn, 'label': ''})
            
        # จุดสุดท้าย: Pure Tension
        pt = -self.ast * self.fy / 1000
        points.append({'Pn': pt, 'Mn': 0, 'phiPn': 0.9 * pt, 'phiMn': 0, 'label': 'Pure Tension'})
        
        return pd.DataFrame(points), phi_pn_max

# ==========================================
# 2. STREAMLIT INTERFACE
# ==========================================

st.set_page_config(page_title="Engineer RC Pro", layout="wide")
st.title("📊 Standard RC Column Interaction Diagram")
st.info("Calculation Reference: ACI 318-19 / WST SDM (Strength Design Method)")

with st.sidebar:
    st.header("Parameters")
    fc = st.number_input("f'c (ksc)", 210, 560, 280)
    fy = st.number_input("fy (ksc)", 3000, 5000, 4000)
    b = st.number_input("b (cm)", 20, 100, 40)
    h = st.number_input("h (cm)", 20, 100, 50)
    db = st.selectbox("DB (mm)", [12, 16, 20, 25, 28, 32], index=3)
    n_bars = st.number_input("Number of Bars", 4, 32, 8, step=2)
    st.markdown("---")
    pu = st.number_input("Required Pu (tons)", value=100.0)
    mu = st.number_input("Required Mu (ton-m)", value=15.0)

# Calculation
calc = RCCalculatorV3(fc, fy, b, h, db, n_bars, 4.0)
df, pn_cap = calc.generate_curve()

# Plotting
fig = go.Figure()
# Nominal Curve
fig.add_trace(go.Scatter(x=df['Mn'], y=df['Pn'], name="Nominal Pn-Mn", line=dict(color='gray', dash='dash')))
# Design Curve (Phi Pn - Phi Mn)
fig.add_trace(go.Scatter(x=df['phiMn'], y=df['phiPn'].clip(upper=pn_cap), 
                         name="Design Strength (Φ)", fill='tozeroy', line=dict(color='blue', width=3)))
# Design Point
fig.add_trace(go.Scatter(x=[mu], y=[pu], mode='markers', name="Load Point", marker=dict(color='red', size=12, symbol='diamond')))

fig.update_layout(xaxis_title="Moment (ton-m)", yaxis_title="Axial Load (tons)", height=700)
st.plotly_chart(fig, use_container_width=True)

# Verification Logic
interp_p = np.interp(mu, df['phiMn'], df['phiPn'].clip(upper=pn_cap))
is_safe = pu <= interp_p and pu >= df['phiPn'].min()

if is_safe:
    st.success(f"✅ PASS: Point ({mu}, {pu}) is within the capacity.")
else:
    st.error(f"❌ FAIL: Point ({mu}, {pu}) exceeds section capacity.")
