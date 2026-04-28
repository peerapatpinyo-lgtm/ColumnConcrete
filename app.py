import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --- 1. SET PAGE CONFIG & CUSTOM CSS ---
st.set_page_config(page_title="RC Pro Designer", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .status-pass { color: #28a745; font-weight: bold; }
    .status-fail { color: #dc3545; font-weight: bold; }
    div[data-testid="stExpander"] { border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CALCULATION ENGINE ---
class RCColumnEngine:
    def __init__(self, fc, fy, b, h, db_mm, n_bars, cover_cm):
        self.fc, self.fy, self.b, self.h = fc, fy, b, h
        self.es = 2.04e6 
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))
        self.db_mm = db_mm
        self.n_bars = n_bars
        self.cover = cover_cm
        
        as_single = np.pi * (db_mm/20)**2 / 4
        self.d_prime = cover_cm + 0.9 + (db_mm/20)
        self.d = h - self.d_prime
        self.ast = n_bars * as_single
        
        # จัดเลเยอร์เหล็ก (สมมติเสาสมมาตร 2 ฝั่ง)
        self.layers = [{'as': (n_bars/2) * as_single, 'd': self.d_prime},
                       {'as': (n_bars/2) * as_single, 'd': self.d}]

    def solve_interaction(self):
        points = []
        c_values = np.concatenate([np.linspace(self.h * 3, self.h, 40), np.linspace(self.h, 0.1, 160)])
        
        balanced_point = None
        ey = self.fy / self.es
        
        for c in c_values:
            a = min(self.beta1 * c, self.h)
            force_c = 0.85 * self.fc * a * self.b
            mom_c = force_c * (self.h/2 - a/2)
            f_s_total, m_s_total, et = 0, 0, 0
            
            for layer in self.layers:
                eps = 0.003 * (c - layer['d']) / c
                fs = np.clip(eps * self.es, -self.fy, self.fy)
                f_layer = layer['as'] * fs
                f_s_total += f_layer
                m_s_total += f_layer * (self.h/2 - layer['d'])
                if layer['d'] == self.d: et = abs(0.003 * (self.d - c) / c) if c < self.d else 0
            
            pn, mn = (force_c + f_s_total) / 1000, (mom_c + m_s_total) / 100000
            phi = 0.65 if et <= ey else 0.90 if et >= 0.005 else 0.65 + 0.25*(et - ey)/(0.005 - ey)
            
            # เก็บจุด Balanced Point
            if balanced_point is None and et >= ey:
                balanced_point = (phi * mn, phi * pn)
                
            points.append({'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn, 'et': et})
        
        po = (0.85 * self.fc * (self.b * self.h - self.ast) + self.fy * self.ast) / 1000
        return pd.DataFrame(points).sort_values('Pn', ascending=False), 0.65 * 0.80 * po, balanced_point

    def design_shear(self, Vu_ton, stirrup_db=9):
        phi = 0.75
        vc = 0.53 * np.sqrt(self.fc) * self.b * self.d / 1000
        vs_req = (Vu_ton / phi) - vc
        av = 2 * (np.pi * (stirrup_db/20)**2 / 4)
        
        if vs_req <= 0:
            s_req = min(self.d/2, 60.0, 16*(self.db_mm/10))
            status = "PASS (Concrete only)"
        else:
            s_calc = (av * self.fy * self.d / (vs_req * 1000))
            s_req = min(s_calc, self.d/2, 60.0)
            status = "PASS (With Stirrups)" if vs_req < (2.1 * np.sqrt(self.fc) * self.b * self.d / 1000) else "FAIL (Section too small)"
            
        return s_req, vc * phi, status

# --- 3. UI SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4342/4342728.png", width=80)
    st.title("Settings")
    with st.expander("Concrete & Steel", expanded=True):
        fc = st.number_input("f'c (ksc)", 210, 560, 280)
        fy = st.number_input("fy (ksc)", 3000, 5000, 4000)
    with st.expander("Geometry", expanded=True):
        b = st.slider("Width b (cm)", 20, 100, 40)
        h = st.slider("Depth h (cm)", 20, 100, 60)
    with st.expander("Reinforcement", expanded=True):
        db = st.selectbox("Main Bar (mm)", [16, 20, 25, 28, 32], index=2)
        n_bars = st.number_input("Total Bars", 4, 32, 8, step=2)
        stirrup_db = st.selectbox("Stirrup (mm)", [6, 9, 12], index=1)

# --- 4. CALCULATION ---
engine = RCColumnEngine(fc, fy, b, h, db, n_bars, 4.0)
df_pm, p_cap, b_pt = engine.solve_interaction()

# --- 5. MAIN CONTENT ---
col_main, col_side = st.columns([2.5, 1])

with col_main:
    # 5.1 Interaction Diagram
    st.subheader("P-M Interaction Diagram")
    fig = go.Figure()

    # Nominal Curve
    fig.add_trace(go.Scatter(x=df_pm['Mn'], y=df_pm['Pn'], name="Nominal Capacity",
                             line=dict(color='rgba(150,150,150,0.5)', dash='dash', width=1.5)))
    
    # Design Curve (The "Nose")
    fig.add_trace(go.Scatter(x=df_pm['phiMn'], y=df_pm['phiPn'].clip(upper=p_cap),
                             fill='tozeroy', fillcolor='rgba(25, 55, 109, 0.1)',
                             name="Design Capacity (Φ)", line=dict(color='#19376D', width=4)))

    # Balanced Point Annotation
    if b_pt:
        fig.add_trace(go.Scatter(x=[b_pt[0]], y=[b_pt[1]], mode='markers+text',
                                 text=["Balanced Point"], textposition="top right",
                                 marker=dict(color='#FF6000', size=10, symbol='diamond')))

    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(title='Moment (ton-m)', gridcolor='#f0f0f0', zerolinecolor='black'),
        yaxis=dict(title='Axial Load (ton)', gridcolor='#f0f0f0', zerolinecolor='black'),
        margin=dict(l=20, r=20, t=20, b=20),
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with col_side:
    # 5.2 Section Sketch
    st.subheader("Section Preview")
    fig_sec, ax = plt.subplots(figsize=(4, 5))
    ax.add_patch(plt.Rectangle((0, 0), b, h, facecolor='#f8f9fa', edgecolor='#333', lw=2))
    # Ties
    ax.add_patch(plt.Rectangle((4, 4), b-8, h-8, fill=False, edgecolor='#dc3545', ls='--', lw=1))
    # Bars
    x_bars = np.linspace(5, b-5, int(n_bars/2))
    for x in x_bars:
        ax.add_patch(plt.Circle((x, 5), db/20, color='#19376D'))
        ax.add_patch(plt.Circle((x, h-5), db/20, color='#19376D'))
    ax.set_aspect('equal')
    ax.axis('off')
    st.pyplot(fig_sec)

    # 5.3 Shear Design Card
    st.subheader("Shear Design")
    vu = st.number_input("Design Shear Vu (ton)", 0.0, 200.0, 15.0)
    s_space, phi_vc, status = engine.design_shear(vu, stirrup_db)
    
    st.metric("Concrete Strength (ΦVc)", f"{phi_vc:.2f} ton")
    status_class = "status-pass" if "PASS" in status else "status-fail"
    st.markdown(f"Status: <span class='{status_class}'>{status}</span>", unsafe_allow_html=True)
    if "PASS" in status:
        st.success(f"Use Ties: RB{stirrup_db} @ {s_space:.0f} cm")

# --- 6. FOOTER METRICS ---
st.write("---")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Max Axial (ΦPn,max)", f"{p_cap:.1f} ton")
m2.metric("Steel Ratio (ρ)", f"{(engine.ast/(b*h))*100:.2f} %")
m3.metric("Concrete Area", f"{b*h} cm²")
m4.metric("Steel Area", f"{engine.ast:.1f} cm²")
