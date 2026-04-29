import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import interp1d

class RCColumnProfessional:
    def __init__(self, shape, layout, b, h, fc, fy, db_mm, n_bars, nx, ny, cover_cm):
        self.shape = shape
        self.layout = layout
        self.b = b
        self.h = h
        self.fc, self.fy = fc, fy
        self.Es = 2.04e6
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))
        
        # 1. Properties by Shape
        if self.shape == "Rectangular":
            self.Ag = self.b * self.h
            self.Ig = (self.b * self.h**3) / 12  
        else: # Circular (b and h represent Diameter D)
            self.D = self.h
            self.Ag = (np.pi * self.D**2) / 4
            self.Ig = (np.pi * self.D**4) / 64
            
        self.Ec = 15100 * np.sqrt(self.fc)
        
        # 2. Rebar Generation (x, y coordinates from center)
        self.db_cm = db_mm / 10
        self.as_single = (np.pi * self.db_cm**2) / 4
        self.d_prime = cover_cm + 0.9 + (self.db_cm/2) # 0.9 cm for ties
        
        self.bars = []
        if self.shape == "Rectangular":
            x_min, x_max = -self.b/2 + self.d_prime, self.b/2 - self.d_prime
            y_min, y_max = -self.h/2 + self.d_prime, self.h/2 - self.d_prime
            
            if self.layout == "2-Faces (Top/Bottom)":
                n_face = n_bars // 2
                x_coords = np.linspace(x_min, x_max, n_face)
                for x in x_coords:
                    self.bars.append({'x': x, 'y': y_max}) # Top
                    self.bars.append({'x': x, 'y': y_min}) # Bottom
                    
            elif self.layout == "4-Faces (Uniform)":
                # Top and Bottom
                x_coords = np.linspace(x_min, x_max, nx)
                for x in x_coords:
                    self.bars.append({'x': x, 'y': y_max}) 
                    self.bars.append({'x': x, 'y': y_min})
                # Left and Right (exclude corners)
                if ny > 2:
                    y_coords = np.linspace(y_min, y_max, ny)[1:-1]
                    for y in y_coords:
                        self.bars.append({'x': x_min, 'y': y})
                        self.bars.append({'x': x_max, 'y': y})
                        
        elif self.shape == "Circular":
            Rs = self.D/2 - self.d_prime
            for i in range(n_bars):
                theta = i * 2 * np.pi / n_bars
                self.bars.append({'x': Rs * np.sin(theta), 'y': Rs * np.cos(theta)})
                
        self.total_as = len(self.bars) * self.as_single
        self.rho = self.total_as / self.Ag 
        
        # Extreme tension steel depth (dt)
        self.dt = self.h/2 - min(bar['y'] for bar in self.bars)

    def solve_pm(self):
        results = []
        c_values = np.concatenate([
            np.linspace(0.001, self.dt, 200), 
            np.linspace(self.dt, self.h * 3, 200)
        ])
        
        for c in c_values:
            a = min(self.beta1 * c, self.h)
            
            # --- Concrete Compression Block Area & Centroid ---
            if self.shape == "Rectangular":
                Cc = 0.85 * self.fc * a * self.b
                Mc = Cc * (self.h/2 - a/2)
            else: # Circular Math
                R = self.D / 2
                if a >= self.D:
                    Ac, y_bar = np.pi * R**2, 0
                else:
                    # Circular segment mechanics
                    theta = 2 * np.arccos((R - a) / R)
                    Ac = (R**2 / 2) * (theta - np.sin(theta))
                    y_bar = (4 * R * np.sin(theta/2)**3) / (3 * (theta - np.sin(theta))) if Ac > 0 else R
                Cc = 0.85 * self.fc * Ac
                Mc = Cc * y_bar # y_bar is distance from centroid
            
            # --- Steel Contribution ---
            Pn_s, Mn_s = 0, 0
            for bar in self.bars:
                d_i = self.h/2 - bar['y'] # depth from extreme compression fiber
                eps_s = 0.003 * (c - d_i) / c
                fs = np.clip(eps_s * self.Es, -self.fy, self.fy)
                
                Fsi = self.as_single * fs
                Pn_s += Fsi
                Mn_s += Fsi * bar['y'] # Moment arm from center

            pn = (Cc + Pn_s) / 1000 
            mn = (Mc + Mn_s) / 100000 
            
            # Phi factor calculation
            et = 0.003 * (self.dt - c) / c
            ey = self.fy / self.Es
            
            if self.shape == "Circular": # ACI rules for Spiral/Ties
                phi_comp = 0.75 if self.layout == "Circular" else 0.65 
            else:
                phi_comp = 0.65
                
            if et >= 0.005: phi = 0.90
            elif et <= ey: phi = phi_comp
            else: phi = phi_comp + (0.90 - phi_comp) * (et - ey) / (0.005 - ey)
            
            results.append({'c': c, 'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn})

        # Pure Tension
        tn = -self.total_as * self.fy / 1000
        results.append({'c': 0, 'Pn': tn, 'Mn': 0, 'phiPn': 0.9 * tn, 'phiMn': 0})

        # Maximum Axial Load (Po)
        po = (0.85 * self.fc * (self.Ag - self.total_as) + self.fy * self.total_as) / 1000
        phi_max_factor = 0.85 if self.shape == "Circular" else 0.80
        phi_comp = 0.75 if self.shape == "Circular" else 0.65
        phi_pn_max = phi_comp * phi_max_factor * po

        return pd.DataFrame(results).sort_values('Pn', ascending=True), phi_pn_max
        
    def calculate_slenderness(self, Pu, Mu, K_factor, Lu_m, Cm, beta_d):
        Lu_cm = Lu_m * 100
        r = 0.25 * self.h if self.shape == "Circular" else 0.3 * self.h
        kl_r = (K_factor * Lu_cm) / r
        
        EI = (0.4 * self.Ec * self.Ig) / (1 + beta_d)
        Pc = (np.pi**2 * EI) / (K_factor * Lu_cm)**2 / 1000
        
        phi_k = 0.75
        if Pu >= (phi_k * Pc):
            delta = 999.9 
        else:
            delta = max(1.0, Cm / (1 - (Pu / (phi_k * Pc))))
            
        Mc = delta * Mu
        return kl_r, Pc, delta, Mc

# --- FUNCTION: PLOT CROSS-SECTION ---
def plot_cross_section(engine):
    fig = go.Figure()
    
    if engine.shape == "Rectangular":
        fig.add_trace(go.Scatter(x=[-engine.b/2, engine.b/2, engine.b/2, -engine.b/2, -engine.b/2], 
                                 y=[-engine.h/2, -engine.h/2, engine.h/2, engine.h/2, -engine.h/2], 
                                 mode='lines', name='Concrete Edge', line=dict(color='black', width=2)))
    else:
        # Draw Circle
        theta = np.linspace(0, 2*np.pi, 100)
        x_circ = (engine.D/2) * np.cos(theta)
        y_circ = (engine.D/2) * np.sin(theta)
        fig.add_trace(go.Scatter(x=x_circ, y=y_circ, mode='lines', name='Concrete Edge', line=dict(color='black', width=2)))
    
    # Draw Rebars
    x_bars = [bar['x'] for bar in engine.bars]
    y_bars = [bar['y'] for bar in engine.bars]
    fig.add_trace(go.Scatter(x=x_bars, y=y_bars, mode='markers', name='Rebars', 
                             marker=dict(color='red', size=10, line=dict(color='darkred', width=1))))
    
    axis_limit = max(engine.b, engine.h) * 0.6 if engine.shape == "Rectangular" else engine.D * 0.6
    fig.update_layout(xaxis_title="Width / X (cm)", yaxis_title="Depth / Y (cm)",
                      xaxis=dict(range=[-axis_limit, axis_limit]),
                      yaxis=dict(range=[-axis_limit, axis_limit], scaleanchor="x", scaleratio=1), 
                      plot_bgcolor='whitesmoke', height=500, showlegend=False)
    return fig

# --- STREAMLIT UI ---
st.set_page_config(page_title="Ultimate RC Column", layout="wide")
st.title("🏗️ Ultimate RC Column (Pro Edition)")

col1, col2 = st.columns([1, 2.5])

with col1:
    with st.expander("1. Section & Reinforcement", expanded=True):
        shape = st.radio("Section Shape", ["Rectangular", "Circular"], horizontal=True)
        st.markdown("---")
        
        fc = st.number_input("f'c (ksc)", value=280)
        fy = st.number_input("fy (ksc)", value=4000)
        
        if shape == "Rectangular":
            c1, c2 = st.columns(2)
            b = c1.number_input("Width, b (cm)", value=40)
            h = c2.number_input("Depth, h (cm)", value=60)
            layout = st.selectbox("Rebar Layout", ["2-Faces (Top/Bottom)", "4-Faces (Uniform)"])
            
            if layout == "2-Faces (Top/Bottom)":
                n_bars = st.number_input("Total Bars (Even)", 4, 40, 8, step=2)
                nx, ny = 0, 0
            else:
                c3, c4 = st.columns(2)
                nx = c3.number_input("Bars in X (Width)", 2, 20, 3)
                ny = c4.number_input("Bars in Y (Depth)", 2, 20, 4)
                n_bars = (2 * nx) + (2 * ny) - 4
                st.info(f"Total Bars Computed: {n_bars}")
        else:
            b = h = st.number_input("Diameter, D (cm)", value=50)
            layout = "Circular"
            n_bars = st.number_input("Total Bars (min 6)", 6, 60, 8, step=1)
            nx, ny = 0, 0
            
        db = st.selectbox("Bar Size (mm)", [16, 20, 25, 28, 32], index=2)
        cover = st.number_input("Covering (cm)", value=4.0)

    with st.expander("2. Loads & Slenderness", expanded=True):
        Pu = st.number_input("Factored Axial, Pu (ton)", value=150.0)
        Mu = st.number_input("Factored Moment, Mu (ton-m)", value=15.0)
        st.markdown("---")
        K_factor = st.number_input("Effective Length Factor (K)", value=1.0, step=0.1)
        Lu = st.number_input("Unsupported Length, Lu (m)", value=4.0, step=0.5)
        Cm = st.number_input("Cm factor (1.0 for single curvature)", value=1.0, step=0.1)
        beta_d = st.slider("Beta_d (Sustained Load Ratio)", 0.0, 1.0, 0.6)

engine = RCColumnProfessional(shape, layout, b, h, fc, fy, db, n_bars, nx, ny, cover)
df, phi_pn_max = engine.solve_pm()

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
    st.markdown("### 📋 Executive Summary")
    m1, m2, m3, m4 = st.columns(4)
    
    rho_pct = engine.rho * 100
    if rho_pct < 1.0:
        rho_status, rho_color = "⚠️ Below 1%", "inverse"
    elif rho_pct > 8.0:
        rho_status, rho_color = "❌ Exceeds 8%", "inverse"
    else:
        rho_status, rho_color = "✅ Normal (OK)", "normal"
        
    m1.metric("Steel Ratio (ρ)", f"{rho_pct:.2f} %", rho_status, delta_color=rho_color)
    m2.metric("Slenderness (KL/r)", f"{kl_r:.1f}", "Slender" if kl_r > 22 else "Short", delta_color="off")
    m3.metric("Critical Load (Pc)", f"{Pc:,.1f} ton")
    m4.metric("Magnifier (δ)", f"{delta:.3f}")

    st.markdown("---")
    st.markdown("#### 🔍 Design Diagnostics")
    
    e_min_m = 0.015 + 0.03 * (h / 100) 
    M_min = Pu * e_min_m
    Actual_Mu = max(Mu, M_min) 
    
    if Mu < M_min:
        st.info(f"💡 **Min. Eccentricity:** The inputted moment is too low. Using minimum **Mu,min = {M_min:.2f} t-m**")
    
    if kl_r > 22:
        st.warning(f"⚠️ **Slender Column Effect:** (KL/r = {kl_r:.1f} > 22)")
        if delta > 1.0:
            Mc_display = Actual_Mu * delta
            st.markdown(f"> 🔄 Design moment magnified: **{Actual_Mu:.2f} t-m** ➔ **<span style='color:red; font-size:1.1em;'>{Mc_display:.2f} t-m</span>**", unsafe_allow_html=True)
        else:
            Mc_display = Actual_Mu
    else:
        st.success(f"✅ **Short Column:** (KL/r = {kl_r:.1f} ≤ 22). No magnification required.")
        Mc_display = Actual_Mu
        
    Mc = Mc_display 

    st.markdown("<br>", unsafe_allow_html=True) 
    if is_safe:
        st.success(f"### ✅ **STATUS: SAFE**\nThe applied demand **(Pu = {Pu} ton, Mc = {Mc:.2f} ton-m)** is within capacity.")
    else:
        st.error(f"### ❌ **STATUS: UNSAFE**\nThe applied demand **(Pu = {Pu} ton, Mc = {Mc:.2f} ton-m)** exceeds capacity!")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 P-M Curve", "📐 Section Details", "📝 Calculation Report"])

    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_design['phiMn'], y=df_design['phiPn'], fill='tozeroy', name="Design Capacity (Φ)", line=dict(color='navy', width=3)))
        fig1.add_trace(go.Scatter(x=[Mu], y=[Pu], mode='markers', name="Original (Mu, Pu)", marker=dict(color='orange', size=10, symbol='circle')))
        
        if delta > 1.0 and delta < 999:
            fig1.add_trace(go.Scatter(x=[Mc], y=[Pu], mode='markers', name="Magnified (Mc, Pu)", marker=dict(color='red', size=12, symbol='x')))
            fig1.add_annotation(x=Mc, y=Pu, ax=Mu, ay=Pu, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor="red")
            
        fig1.update_layout(xaxis_title="Moment, M (ton-m)", yaxis_title="Axial Load, P (ton)", plot_bgcolor='white', height=500)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.markdown("### Cross-Section & Rebar Arrangement")
        fig_sec = plot_cross_section(engine)
        st.plotly_chart(fig_sec, use_container_width=True)

    with tab3:
        st.markdown("## 📝 Detailed Calculation Report")
        st.markdown("---")
        st.markdown("#### 1. Section Properties")
        if shape == "Rectangular":
            st.latex(rf"A_g = {b} \times {h} = {engine.Ag:,.2f} \text{{ cm}}^2")
            st.latex(rf"I_g = \frac{{{b} \times {h}^3}}{{12}} = {engine.Ig:,.2f} \text{{ cm}}^4")
        else:
            st.latex(rf"A_g = \frac{{\pi \times {b}^2}}{{4}} = {engine.Ag:,.2f} \text{{ cm}}^2")
            st.latex(rf"I_g = \frac{{\pi \times {b}^4}}{{64}} = {engine.Ig:,.2f} \text{{ cm}}^4")
            
        st.markdown("#### 2. Reinforcement Ratio")
        st.latex(rf"\rho = \frac{{{engine.total_as:.2f}}}{{{engine.Ag:,.2f}}} = {engine.rho*100:.2f}\%")
