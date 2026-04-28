import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ==========================================
# 1. ENGINEERING CALCULATION LOGIC
# ==========================================

def calculate_column_strength(fc, fy, b, h, d_prime, as_total):
    """
    คำนวณกำลังของเสา RC แบบ Simplified Interaction Diagram (ACI 318)
    """
    phi_comp = 0.65  # สำหรับเสาปลอกเดี่ยว (Tied Column)
    ag = b * h
    d = h - d_prime
    
    # 1. Pure Compression (Point A)
    # Po = 0.85 * fc' * (Ag - Ast) + fy * Ast
    # Pn_max = 0.80 * Po (สำหรับเสาปลอกเดี่ยว)
    po = (0.85 * fc * (ag - as_total) + fy * as_total) / 1000 # Convert to metric tons
    phi_pn_max = phi_comp * 0.80 * po

    # 2. Balanced Point (Point B)
    # cb = 6117 / (6117 + fy) * d (Simplified Strain Compatibility)
    cb = (600 / (600 + fy)) * d
    ab = 0.85 * cb # beta1 is approx 0.85 for fc' <= 280 ksc
    
    # Force components at balanced
    c_c = 0.85 * fc * ab * b / 1000
    fs_prime = 600 * (cb - d_prime) / cb
    fs_prime = min(fs_prime, fy)
    s_force = (as_total / 2) * (fy - 0.85 * fc) / 1000 # Approximation
    
    pn_b = c_c # Simplified balanced axial
    mn_b = (c_c * (h/2 - ab/2) + s_force * (h/2 - d_prime)) / 100000 # Ton-m
    
    phi_pn_b = phi_comp * pn_b
    phi_mn_b = phi_comp * mn_b

    # 3. Pure Flexure (Point C)
    # Mn = As * fy * (d - a/2)
    a_flex = (as_total/2 * fy) / (0.85 * fc * b)
    mn_pure = (as_total/2 * fy * (d - a_flex/2)) / 100000
    phi_mn_pure = 0.90 * mn_pure # Tension controlled

    return {
        "phiPn_max": phi_pn_max,
        "phiPn_b": phi_pn_b,
        "phiMn_b": phi_mn_b,
        "phiMn_pure": phi_mn_pure
    }

def check_slenderness(k, l, b, h):
    """เช็คเสาสั้น/เสายาว ตามมาตรฐาน ACI (r = 0.3h สำหรับหน้าตัดสี่เหลี่ยม)"""
    r = 0.3 * h
    slenderness_ratio = (k * l) / r
    is_short = slenderness_ratio < 34 # Simplified check for non-sway
    return slenderness_ratio, is_short

def design_corbel(vu, fc, fy, b, d, av):
    """
    คำนวณ Corbel (หูช้าง) เบื้องต้น (Shear Friction Theory)
    """
    phi_shear = 0.75
    # Check depth requirement
    # Vn must be <= 0.2 * fc * b * d
    vn_max = (0.2 * fc * b * d) / 1000
    is_size_ok = (vu / phi_shear) <= vn_max
    
    # Required reinforcement (Simplified Shear Friction)
    mu = 1.4 # Coefficient of friction for normal concrete
    avf = (vu * 1000) / (phi_shear * fy * mu) # cm2
    return is_size_ok, avf

# ==========================================
# 2. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Industrial RC Designer", layout="wide")
st.title("🏗️ Industrial RC Column & Corbel Designer")
st.markdown("---")

# --- SIDEBAR INPUTS ---
st.sidebar.header("🛠️ Input Parameters")

# Material Properties
fc = st.sidebar.number_input("Concrete Strength, $f'_c$ (ksc)", value=280)
fy = st.sidebar.number_input("Steel Strength, $f_y$ (ksc)", value=4000)

# Section Properties
st.sidebar.subheader("Section Dimensions (cm)")
col_b = st.sidebar.slider("Width (b)", 20, 100, 40)
col_h = st.sidebar.slider("Depth (h)", 20, 100, 40)
cover = st.sidebar.number_input("Covering (cm)", value=4.0)

# Reinforcement
st.sidebar.subheader("Reinforcement")
rebar_dia = st.sidebar.selectbox("Bar Diameter (mm)", [12, 16, 20, 25, 28])
rebar_count = st.sidebar.number_input("Total Number of Bars", value=8, step=2)
as_total = (np.pi * (rebar_dia/20)**2) * rebar_count

# Loads & Slenderness
st.sidebar.subheader("Design Loads & Length")
pu_load = st.sidebar.number_input("Axial Load, $P_u$ (tons)", value=50.0)
mu_load = st.sidebar.number_input("Moment, $M_u$ (ton-m)", value=10.0)
col_l = st.sidebar.number_input("Clear Height, $L$ (m)", value=4.0) * 100
k_factor = st.sidebar.selectbox("Effective Length Factor (k)", [0.7, 1.0, 1.2, 2.0], index=1)

# Corbel Input
st.sidebar.subheader("Corbel Design (Optional)")
vu_corbel = st.sidebar.number_input("Corbel Shear, $V_u$ (tons)", value=15.0)
av_dist = st.sidebar.number_input("Shear Span, $a_v$ (cm)", value=20.0)

# --- CALCULATIONS ---
results = calculate_column_strength(fc, fy, col_b, col_h, cover + (rebar_dia/20), as_total)
slend_ratio, is_short = check_slenderness(k_factor, col_l, col_b, col_h)
corbel_ok, avf_req = design_corbel(vu_corbel, fc, fy, col_b, col_h-cover, av_dist)

# --- DISPLAY MAIN CONTENT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Interaction Diagram")
    
    # Prepare P-M Curve Points
    m_points = [0, results['phiMn_b'], results['phiMn_pure']]
    p_points = [results['phiPn_max'], results['phiPn_b'], 0]
    
    fig = go.Figure()
    # P-M Curve
    fig.add_trace(go.Scatter(x=m_points, y=p_points, mode='lines+markers', name='Capacity Envelope', line=dict(color='royalblue', width=3)))
    # Design Point
    fig.add_trace(go.Scatter(x=[mu_load], y=[pu_load], mode='markers', name='Design Point (Pu, Mu)', marker=dict(color='red', size=12, symbol='x')))
    
    fig.update_layout(xaxis_title="Moment (ton-m)", yaxis_title="Axial Load (tons)", height=500)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("断面 Cross-section Visualization")
    fig_rc, ax = plt.subplots(figsize=(4, 4))
    rect = plt.Rectangle((0, 0), col_b, col_h, linewidth=2, edgecolor='black', facecolor='lightgrey')
    ax.add_patch(rect)
    # Plot bars (simple representation)
    ax.scatter([cover, col_b-cover]*int(rebar_count/2), 
               [cover]*2 + [col_h-cover]*2 + [col_h/2]*(rebar_count-4), 
               color='red', s=100)
    ax.set_xlim(-5, col_b+5)
    ax.set_ylim(-5, col_h+5)
    ax.set_aspect('equal')
    st.pyplot(fig_rc)

# --- SUMMARY TABLE ---
st.subheader("📋 Design Summary")
status = "✅ PASS" if (pu_load <= results['phiPn_max'] and mu_load <= results['phiMn_b']) else "❌ FAIL"
color = "green" if status == "✅ PASS" else "red"

col_res1, col_res2, col_res3 = st.columns(3)
col_res1.metric("Slenderness Ratio", f"{slend_ratio:.2f}", "Short Column" if is_short else "Long Column")
col_res2.metric("Max Axial Capacity", f"{results['phiPn_max']:.2f} tons")
col_res3.markdown(f"### Status: :{color}[{status}]")

# --- CORBEL RESULT ---
with st.expander("🏗️ Corbel Design Result"):
    if corbel_ok:
        st.success(f"Corbel Size: OK | Required $A_{{vf}}$ = {avf_req:.2f} cm²")
    else:
        st.error("Corbel size too small for shear friction! Increase depth or width.")

# Engineering Notes
st.info(f"Note: Interaction diagram is based on simplified SDM. Φ factor used: {0.65} for compression.")
