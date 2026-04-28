import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ==========================================
# 1. ADVANCED CALCULATION ENGINE
# ==========================================

class RCCalculator:
    @staticmethod
    def get_beta1(fc):
        """คำนวณค่า beta1 ตามมาตรฐาน ACI 318"""
        if fc <= 280: return 0.85
        elif fc >= 560: return 0.65
        else: return 0.85 - (0.05 * (fc - 280) / 70)

    @staticmethod
    def get_phi(epsilon_t, fy):
        """คำนวณค่า Phi (Strength Reduction Factor) ตามค่า Strain"""
        epsilon_ty = fy / 2.04e6 # Es = 2.04e6 ksc
        if epsilon_t <= epsilon_ty: return 0.65 # Compression Controlled
        if epsilon_t >= 0.005: return 0.90      # Tension Controlled
        # Transition Zone
        return 0.65 + (0.90 - 0.65) * (epsilon_t - epsilon_ty) / (0.005 - epsilon_ty)

    @classmethod
    def calculate_pm_points(cls, fc, fy, b, h, d_prime, as_total):
        d = h - d_prime
        as_half = as_total / 2
        beta1 = cls.get_beta1(fc)
        es = 2.04e6
        
        points = []

        # 1. Point A: Pure Compression (Maximum Axial)
        po = (0.85 * fc * (b * h - as_total) + fy * as_total) / 1000
        phi_pn_max = 0.65 * 0.80 * po
        points.append({'m': 0, 'p': phi_pn_max, 'label': 'Max Axial'})

        # 2. Point B: Zero Tension (c = h)
        c = h
        a = beta1 * c
        cc = 0.85 * fc * a * b / 1000
        fs_prime = min(6120 * (c - d_prime) / c, fy)
        fs = 6120 * (c - d) / c # จะได้ค่าเป็นบวก (แรงอัด)
        pn = cc + (as_half * fs_prime / 1000) + (as_half * fs / 1000)
        mn = (cc * (h/2 - a/2) + (as_half * fs_prime / 1000) * (h/2 - d_prime) - (as_half * fs / 1000) * (d - h/2)) / 100
        points.append({'m': 0.65 * mn, 'p': 0.65 * pn, 'label': 'Zero Tension'})

        # 3. Point C: Balanced Point (epsilon_s = epsilon_y)
        ey = fy / es
        cb = 0.003 * d / (0.003 + ey)
        ab = beta1 * cb
        cc = 0.85 * fc * ab * b / 1000
        fs_prime = min(6120 * (cb - d_prime) / cb, fy)
        pn_b = cc + (as_half * fs_prime / 1000) - (as_half * fy / 1000)
        mn_b = (cc * (h/2 - ab/2) + (as_half * fs_prime / 1000) * (h/2 - d_prime) + (as_half * fy / 1000) * (d - h/2)) / 100
        points.append({'m': 0.65 * mn_b, 'p': 0.65 * pn_b, 'label': 'Balanced'})

        # 4. Point D: Tension Controlled (epsilon_t = 0.005)
        c005 = 0.003 * d / (0.003 + 0.005)
        a005 = beta1 * c005
        cc = 0.85 * fc * a005 * b / 1000
        fs_prime = min(6120 * (c005 - d_prime) / c005, fy)
        pn_005 = cc + (as_half * fs_prime / 1000) - (as_half * fy / 1000)
        mn_005 = (cc * (h/2 - a005/2) + (as_half * fs_prime / 1000) * (h/2 - d_prime) + (as_half * fy / 1000) * (d - h/2)) / 100
        points.append({'m': 0.90 * mn_005, 'p': 0.90 * pn_005, 'label': 'Tension Controlled'})

        # 5. Point E: Pure Flexure (Pn = 0)
        # Simplified: As*fy = 0.85*fc*a*b
        a_pure = (as_half * fy) / (0.85 * fc * b)
        mn_pure = (as_half * fy * (d - a_pure/2)) / 100000
        points.append({'m': 0.90 * mn_pure, 'p': 0, 'label': 'Pure Moment'})

        return pd.DataFrame(points)

# ==========================================
# 2. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Pro RC Design", layout="wide")
st.title("🏗️ Professional RC Column Design (ACI 318-SDM)")

with st.sidebar:
    st.header("1. Material & Section")
    fc = st.number_input("f'c (ksc)", value=280)
    fy = st.number_input("fy (ksc)", value=4000)
    b = st.slider("Width b (cm)", 20, 100, 40)
    h = st.slider("Depth h (cm)", 20, 100, 50)
    
    st.header("2. Reinforcement")
    db = st.selectbox("DB Size (mm)", [12, 16, 20, 25, 28, 32])
    n_bars = st.number_input("Number of bars", value=8, step=2)
    cover = st.number_input("Clear Cover (cm)", value=4.0)
    as_total = (np.pi * (db/20)**2) * n_bars
    
    st.header("3. Factored Loads")
    pu = st.number_input("Pu (tons)", value=80.0)
    mu = st.number_input("Mu (ton-m)", value=15.0)

# Calculations
calc = RCCalculator()
df_pm = calc.calculate_pm_points(fc, fy, b, h, cover + db/20, as_total)

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interaction Diagram")
    fig = go.Figure()
    # Draw Curve
    fig.add_trace(go.Scatter(x=df_pm['m'], y=df_pm['p'], mode='lines+markers', 
                             name='Capacity Envelope', line=dict(color='blue', shape='spline')))
    # Design Point
    is_safe = False
    # Simple check if point is inside polygon (Approximation)
    fig.add_trace(go.Scatter(x=[mu], y=[pu], mode='markers', 
                             name='Design Point', marker=dict(color='red', size=15, symbol='diamond')))
    
    fig.update_layout(xaxis_title="Phi Mn (ton-m)", yaxis_title="Phi Pn (tons)", height=600)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Results Summary")
    # ตรวจสอบเบื้องต้น
    max_p = df_pm['p'].max()
    max_m = df_pm['m'].max()
    
    st.write(f"**Total Rebar Area:** {as_total:.2f} $cm^2$")
    st.write(f"**Ratio ($\rho$):** {as_total/(b*h)*100:.2d} %")
    
    if (pu <= max_p) and (mu <= max_m):
        st.success("STATUS: PASS (Preliminary)")
    else:
        st.error("STATUS: FAIL (Out of Bound)")

    with st.expander("Show Calculation Points"):
        st.dataframe(df_pm)

# --- CORBEL DESIGN MODULE ---
st.markdown("---")
st.subheader("🏗️ Corbel Design (หูช้างรับเครน)")
cc1, cc2 = st.columns(2)
with cc1:
    vu = st.number_input("Vu (tons)", value=20.0)
    av = st.number_input("Shear Span av (cm)", value=15.0)
with cc2:
    # ACI 318 Corbel Check
    d_corbel = h - cover
    if av/d_corbel > 1.0:
        st.warning("Warning: av/d > 1.0. This is not a Corbel (use Beam theory).")
    
    # Shear Friction
    phi_v = 0.75
    avf = (vu * 1000) / (phi_v * fy * 1.4) # mu = 1.4 for normal weight
    st.info(f"Required Shear Friction Rebar ($A_{{vf}}$): {avf:.2f} $cm^2$")
