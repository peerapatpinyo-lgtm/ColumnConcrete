import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ==========================================
# 1. CORE ENGINEERING ENGINE (ACI 318-19 SDM)
# ==========================================

class RCCalculator:
    def __init__(self, fc, fy, b, h, db, n_bars, cover):
        self.fc = fc
        self.fy = fy
        self.b = b
        self.h = h
        self.d = h - (cover + db/20 + 0.9) # d calculation (approx. stirrup 9mm)
        self.d_prime = cover + db/20 + 0.9
        self.as_total = (np.pi * (db/20)**2) * n_bars
        self.as_side = self.as_total / 2 # Assume equal reinforcement on two faces
        self.es = 2.04e6 # Elastic modulus of steel (ksc)
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))

    def get_capacity(self, c):
        """Calculate Pn and Mn for a given neutral axis depth 'c'"""
        a = self.beta1 * c
        # Concrete compression force
        cc = 0.85 * self.fc * min(a, self.h) * self.b
        
        # Strain in steel layers
        eps_s_prime = 0.003 * (c - self.d_prime) / c
        eps_t = 0.003 * (self.d - c) / c
        
        fs_prime = max(-self.fy, min(self.fy, eps_s_prime * self.es))
        fs = max(-self.fy, min(self.fy, eps_t * self.es))
        
        # Nominal Strengths
        pn = (cc + self.as_side * fs_prime + self.as_side * fs) / 1000 # Metric Tons
        mn = (cc * (self.h/2 - min(a, self.h)/2) + 
              self.as_side * fs_prime * (self.h/2 - self.d_prime) - 
              self.as_side * fs * (self.d - self.h/2)) / 100000 # Ton-m
        
        # Strength reduction factor (Phi)
        eps_ty = self.fy / self.es
        if eps_t <= eps_ty:
            phi = 0.65
        elif eps_t >= 0.005:
            phi = 0.90
        else:
            phi = 0.65 + (0.90 - 0.65) * (eps_t - eps_ty) / (0.005 - eps_ty)
            
        return pn, mn, phi, eps_t

    def generate_diagram(self):
        """Generates points for the Interaction Diagram"""
        results = []
        # Pure Compression point
        po = (0.85 * self.fc * (self.b * self.h - self.as_total) + self.fy * self.as_total) / 1000
        phi_pn_max = 0.65 * 0.80 * po
        
        # Sweep neutral axis from very large to very small
        c_values = np.linspace(self.h * 2, self.d * 0.1, 150)
        for c in c_values:
            pn, mn, phi, _ = self.get_capacity(c)
            results.append({'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn})
        
        # Add Pure Flexure
        results.append({'Pn': 0, 'Mn': (self.as_side * self.fy * (self.d - self.d_prime)) / 100000, 
                        'phiPn': 0, 'phiMn': 0.9 * (self.as_side * self.fy * (self.d - self.d_prime)) / 100000})
        
        df = pd.DataFrame(results)
        # Cap Pn at Pn_max
        df['phiPn_capped'] = df['phiPn'].clip(upper=phi_pn_max)
        return df, phi_pn_max

# ==========================================
# 2. STREAMLIT UI SETUP
# ==========================================

st.set_page_config(page_title="Advanced RC Column Designer", layout="wide")
st.markdown("""<style> .main { background-color: #f5f7f9; } </style>""", unsafe_allow_html=True)

st.title("🏗️ Professional RC Column & Corbel Designer")
st.caption("Standard: ACI 318-19 / WST SDM Method")

# --- SIDEBAR: INPUT ---
with st.sidebar:
    st.header("🔍 Input Parameters")
    with st.expander("Materials", expanded=True):
        fc = st.number_input("f'c (Concrete - ksc)", 210, 600, 280)
        fy = st.number_input("fy (Steel - ksc)", 3000, 5000, 4000)
    
    with st.expander("Section Geometry", expanded=True):
        b = st.number_input("Width b (cm)", 20, 200, 40)
        h = st.number_input("Depth h (cm)", 20, 200, 50)
        cover = st.number_input("Clear Cover (cm)", 2.0, 7.5, 4.0)
    
    with st.expander("Reinforcement", expanded=True):
        db = st.selectbox("DB Main Bar (mm)", [12, 16, 20, 25, 28, 32], index=3)
        n_bars = st.number_input("Total Bars (Even number)", 4, 40, 8, step=2)
    
    with st.expander("Design Loads", expanded=True):
        pu_req = st.number_input("Factored Pu (tons)", 0.0, 1000.0, 100.0)
        mu_req = st.number_input("Factored Mu (ton-m)", 0.0, 500.0, 20.0)
        l_column = st.number_input("Length L (m)", 1.0, 15.0, 4.0)
        k_factor = st.number_input("Effective Length k", 0.5, 2.1, 1.0)

# --- CALCULATION PROCESS ---
calc = RCCalculator(fc, fy, b, h, db, n_bars, cover)
df_diag, pn_max = calc.generate_diagram()

# Slenderness Check
r = 0.3 * h
slenderness = (k_factor * l_column * 100) / r
is_slender = slenderness > 22 # ACI threshold for non-sway

# --- MAIN DISPLAY ---
tab1, tab2 = st.tabs(["📊 Column Design", "🏗️ Corbel Design"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Interaction Diagram")
        fig = go.Figure()
        # Nominal Curve
        fig.add_trace(go.Scatter(x=df_diag['Mn'], y=df_diag['Pn'], name='Nominal (Pn-Mn)', 
                                 line=dict(color='gray', dash='dash')))
        # Design Curve (Capped)
        fig.add_trace(go.Scatter(x=df_diag['phiMn'], y=df_diag['phiPn_capped'], name='Design (phiPn-phiMn)', 
                                 line=dict(color='blue', width=3), fill='tozeroy'))
        # Design Point
        fig.add_trace(go.Scatter(x=[mu_req], y=[pu_req], mode='markers', name='Required Load',
                                 marker=dict(color='red', size=15, symbol='diamond-wide')))
        
        fig.update_layout(xaxis_title="Moment (ton-m)", yaxis_title="Axial (tons)", height=600, 
                          hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Verification")
        # Check if inside curve (simplified check)
        # We check if req_Pu is less than max allowed Pn for the given Mu
        interp_p = np.interp(mu_req, df_diag['phiMn'], df_diag['phiPn_capped'])
        is_safe = (pu_req <= interp_p) and (pu_req <= pn_max)
        
        if is_safe:
            st.success("### STATUS: PASS")
        else:
            st.error("### STATUS: FAIL")
        
        st.metric("Slenderness Ratio (kL/r)", f"{slenderness:.2f}", 
                  "Long Column" if is_slender else "Short Column", delta_color="inverse")
        
        st.write("---")
        st.write(f"**Section:** {b}x{h} cm")
        st.write(f"**Steel Ratio:** {(calc.as_total/(b*h)*100):.2f}%")
        st.caption("Min: 1.0%, Max: 8.0% (ACI)")

# --- CORBEL (BRACKET) MODULE ---
with tab2:
    st.subheader("Corbel Reinforcement Design (ACI 318)")
    c_col1, c_col2 = st.columns(2)
    
    with c_col1:
        vu_c = st.number_input("Factored Shear Vu (tons)", 0.0, 200.0, 30.0)
        av_c = st.number_input("Distance av (cm)", 5.0, 100.0, 20.0)
        n_c = st.number_input("Tension Force Nuc (tons)", 0.0, 100.0, 6.0) # 0.2*Vu typical
        
    with c_col2:
        d_c = h - cover # Effective depth of corbel
        if av_c / d_c > 1.0:
            st.error("Invalid Geometry: av/d > 1.0 (Not a Corbel, use Beam Theory)")
        else:
            # Shear Friction
            phi_v = 0.75
            mu_friction = 1.4 # Normal concrete
            avf = (vu_c * 1000) / (phi_v * fy * mu_friction)
            
            # Flexure
            af = (vu_c * 1000 * av_c + n_c * 1000 * (h-d_c)) / (phi_v * fy * d_c)
            
            # Tension
            an = (n_c * 1000) / (phi_v * fy)
            
            # Area of Primary Tension Steel (As)
            as_primary = max(af + an, (2*avf/3 + an))
            # Area of Closed Stirrups (Ah)
            ah = 0.5 * (as_primary - an)
            
            st.info(f"**Primary Steel (As):** {as_primary:.2f} cm²")
            st.info(f"**Horizontal Ties (Ah):** {ah:.2f} cm²")
            st.write("---")
            st.caption("Check Vn max: " + str(round(phi_v * 0.2 * fc * b * d_c / 1000, 2)) + " tons")

st.markdown("---")
st.markdown("💡 **Tip:** สำหรับเสาโรงงานสูงเกิน 6 เมตร โปรดตรวจสอบค่า Moment Magnification ($\delta_{ns}$) หาก Slenderness > 22")
