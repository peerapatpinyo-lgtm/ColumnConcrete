import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. CORE CALCULATION ENGINE (ACI 318-19)
# ==========================================

class RCColumnDesigner:
    def __init__(self, fc, fy, b, h, db, n_bars, cover):
        self.fc = fc
        self.fy = fy
        self.b = b
        self.h = h
        self.es = 2.04e6  # Modulus of Elasticity (ksc)
        
        # Section Geometry
        self.cover_all = cover + 0.9 + (db/20) # Total dist to bar center (approx with stirrup)
        self.ast = n_bars * (np.pi * (db/20)**2) # Total steel area (cm2)
        
        # Reinforcement Layers (Simplified to 2 layers for standard square columns)
        self.d_layers = [self.cover_all, h - self.cover_all]
        self.as_layers = [self.ast/2, self.ast/2]
        
    def get_params(self):
        # Beta 1 according to ACI
        if self.fc <= 280: b1 = 0.85
        elif self.fc >= 560: b1 = 0.65
        else: b1 = 0.85 - (0.05 * (self.fc - 280) / 70)
        return b1

    def solve_capacity(self, c):
        """Calculates Pn and Mn for a given Neutral Axis depth 'c'"""
        if c <= 0: return self.pure_tension()
        
        b1 = self.get_params()
        a = min(b1 * c, self.h)
        
        # 1. Concrete Force (Cc)
        cc = 0.85 * self.fc * a * self.b
        mn_c = cc * (self.h/2 - a/2) # Moment around geometric center
        
        # 2. Steel Forces (Fs)
        pn_s = 0
        mn_s = 0
        et = 0 # Net tensile strain for phi calculation
        
        for d_i, as_i in zip(self.d_layers, self.as_layers):
            eps_i = 0.003 * (c - d_i) / c
            fs_i = max(-self.fy, min(self.fy, eps_i * self.es))
            
            # If bar is in compression zone, subtract concrete area displacement
            f_eff = fs_i - 0.85 * self.fc if eps_i > 0 and d_i < a else fs_i
            
            force = as_i * f_eff
            pn_s += force
            mn_s += force * (self.h/2 - d_i)
            et = eps_i # Assuming last layer is the tension face
            
        pn = (cc + pn_s) / 1000 # Tons
        mn = (mn_c + mn_s) / 100000 # Ton-m
        
        # 3. Phi Factor Calculation
        ey = self.fy / self.es
        eps_t_abs = abs(et) if et < 0 else 0
        if eps_t_abs <= ey: phi = 0.65
        elif eps_t_abs >= 0.005: phi = 0.90
        else: phi = 0.65 + (0.90 - 0.65) * (eps_t_abs - ey) / (0.005 - ey)
        
        return pn, mn, phi

    def pure_tension(self):
        pn = -self.ast * self.fy / 1000
        return pn, 0, 0.90

    def generate_diagram(self):
        data = []
        # Pure Compression (Point A)
        po = (0.85 * self.fc * (self.b * self.h - self.ast) + self.fy * self.ast) / 1000
        phi_pn_max = 0.65 * 0.80 * po
        data.append({'Pn': po, 'Mn': 0, 'phiPn': phi_pn_max, 'phiMn': 0})
        
        # Sweep Neutral Axis from Infinity to 0
        c_range = np.logspace(np.log10(0.1), np.log10(self.h * 10), 200)[::-1]
        for c in c_range:
            pn, mn, phi = self.solve_capacity(c)
            data.append({'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn})
            
        # Pure Tension
        pn_t, mn_t, phi_t = self.pure_tension()
        data.append({'Pn': pn_t, 'Mn': mn_t, 'phiPn': phi_t * pn_t, 'phiMn': 0})
        
        return pd.DataFrame(data), phi_pn_max

# ==========================================
# 2. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Industrial RC Designer", layout="wide")
st.title("🧱 Industrial RC Column Interaction Diagram")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Design Parameters")
    fc = st.number_input("f'c (Concrete Strength - ksc)", 210, 560, 280)
    fy = st.number_input("fy (Steel Strength - ksc)", 3000, 5000, 4000)
    b = st.number_input("Column Width b (cm)", 20, 100, 40)
    h = st.number_input("Column Depth h (cm)", 20, 100, 50)
    db = st.selectbox("Main Bar Diameter (mm)", [12, 16, 20, 25, 28, 32], index=3)
    n_bars = st.number_input("Total Number of Bars", 4, 24, 8, step=2)
    st.markdown("---")
    pu = st.number_input("Applied Load Pu (tons)", value=120.0)
    mu = st.number_input("Applied Moment Mu (ton-m)", value=20.0)

# Calculation
designer = RCColumnDesigner(fc, fy, b, h, db, n_bars, 4.0)
df_res, phi_pn_max = designer.generate_diagram()

# Visualization
col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure()
    # Nominal Curve
    fig.add_trace(go.Scatter(x=df_res['Mn'], y=df_res['Pn'], name="Nominal Capacity (Pn-Mn)", line=dict(color='gray', dash='dash')))
    # Design Curve
    fig.add_trace(go.Scatter(x=df_res['phiMn'], y=df_res['phiPn'].clip(upper=phi_pn_max), 
                             fill='tozeroy', name="Design Capacity (ΦPn-ΦMn)", line=dict(color='blue', width=3)))
    # Design Point
    fig.add_trace(go.Scatter(x=[mu], y=[pu], mode='markers+text', text=["Design Point"], textposition="top center",
                             marker=dict(color='red', size=15, symbol='x'), name="Factored Load"))
    
    fig.update_layout(xaxis_title="Moment (ton-m)", yaxis_title="Axial Load (tons)", height=700, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("✅ Safety Verification")
    
    # Check if point is inside the curve
    max_phi_pn_at_mu = np.interp(mu, df_res['phiMn'], df_res['phiPn'].clip(upper=phi_pn_max))
    
    is_safe = (pu <= max_phi_pn_at_mu) and (pu >= df_res['phiPn'].min())
    
    if is_safe:
        st.success("### STATUS: PASS")
    else:
        st.error("### STATUS: FAIL")
        
    st.write(f"**Max ΦPn at given Mu:** {max_phi_pn_at_mu:.2f} tons")
    st.write(f"**Total Steel Area:** {designer.ast:.2f} cm²")
    st.write(f"**Reinforcement Ratio:** {(designer.ast/(b*h)*100):.2f} %")
    st.caption("Min ratio ACI: 1.0%, Max ratio: 8.0%")

st.info("💡 Note: กราฟนี้คำนวณโดยใช้พฤติกรรมจริงของหน้าตัด (Strain Compatibility) โดยพิจารณาค่า Phi ที่แปรผันตามแรงดึงในเหล็กเสริม")
