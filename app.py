import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# REFINED ENGINEERING ENGINE
# ==========================================

class RCColumnPro:
    def __init__(self, fc, fy, b, h, db, n_bars, cover):
        self.fc, self.fy, self.b, self.h = fc, fy, b, h
        self.es = 2.04e6  # ksc
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))
        
        # การจัดเหล็ก: สมมติจัดสองฝั่ง (สมมาตร) เพื่อความแม่นยำของกราฟ
        self.d_prime = cover + 0.9 + db/20  # ระยะถึงจุดศก.เหล็กอัด
        self.d = h - self.d_prime          # ระยะถึงจุดศก.เหล็กดึง
        self.as_total = (np.pi * (db/20)**2) * n_bars
        self.as_side = self.as_total / 2

    def calculate_points(self):
        points = []
        # วนค่า Neutral Axis (c) ตั้งแต่หน้าตัดเต็มจนถึงเกือบศูนย์
        # เพิ่มจุด c = infinity สำหรับ Pure Compression
        c_list = [1e10, self.h * 1.5, self.h] + list(np.linspace(self.h, 1, 100)) + [0.1]
        
        for c in c_list:
            # 1. Concrete Force
            a = min(self.beta1 * c, self.h)
            cc = 0.85 * self.fc * a * self.b
            
            # 2. Steel Forces (Strain Compatibility)
            eps_cu = 0.003
            eps_s_prime = eps_cu * (c - self.d_prime) / c
            eps_t = eps_cu * (self.d - c) / c
            
            fs_prime = max(-self.fy, min(self.fy, eps_s_prime * self.es))
            fs = max(-self.fy, min(self.fy, eps_t * self.es))
            
            # 3. Nominal Strength (Pn, Mn)
            pn = (cc + self.as_side * fs_prime + self.as_side * fs) / 1000
            # Moment รอบ Centerline ของหน้าตัด
            mn = (cc * (self.h/2 - a/2) + self.as_side * fs_prime * (self.h/2 - self.d_prime) - 
                  self.as_side * fs * (self.d - self.h/2)) / 100000
            
            # 4. Phi Factor (ACI 318-19)
            eps_ty = self.fy / self.es
            if eps_t <= eps_ty: phi = 0.65
            elif eps_t >= 0.005: phi = 0.90
            else: phi = 0.65 + (0.90 - 0.65) * (eps_t - eps_ty) / (0.005 - eps_ty)
            
            points.append({'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn, 'type': 'calc'})

        df = pd.DataFrame(points)
        
        # 5. Pure Compression Cap (ACI requirement)
        po = (0.85 * self.fc * (self.b * self.h - self.as_total) + self.fy * self.as_total) / 1000
        phi_pn_max = 0.65 * 0.80 * po  # Tied column limit
        
        # ตัดส่วนเกินของกราฟที่เกิน Pn_max
        df['phiPn_capped'] = df['phiPn'].clip(upper=phi_pn_max)
        
        return df, phi_pn_max

# ==========================================
# UI & VISUALIZATION
# ==========================================

st.set_page_config(page_title="Corrected RC Design", layout="wide")
st.title("🏗️ Accurate RC Column Interaction Diagram")

# Sidebar
with st.sidebar:
    st.header("Inputs")
    fc = st.number_input("f'c (ksc)", value=280)
    fy = st.number_input("fy (ksc)", value=4000)
    b = st.slider("Width (cm)", 20, 100, 40)
    h = st.slider("Depth (cm)", 20, 100, 50)
    db = st.selectbox("Bar Size (mm)", [12, 16, 20, 25, 28, 32], index=3)
    n_bars = st.number_input("Total Bars", value=8, step=2)
    pu = st.number_input("Pu Load (tons)", value=120.0)
    mu = st.number_input("Mu Moment (ton-m)", value=15.0)

# Run Calculation
engine = RCColumnPro(fc, fy, b, h, db, n_bars, 4.0)
df, pn_limit = engine.calculate_points()

# Plotting
fig = go.Figure()

# เส้นแรงต้านที่ยอมรับ (Design Curve) - เส้นที่เราใช้เช็คความปลอดภัย
fig.add_trace(go.Scatter(x=df['phiMn'], y=df['phiPn_capped'], name='Design Curve (ΦPn-ΦMn)',
                         line=dict(color='blue', width=4), fill='tozeroy'))

# เส้นกำลังพิกัด (Nominal Curve) - เส้นดิบก่อนคูณ Φ
fig.add_trace(go.Scatter(x=df['Mn'], y=df['Pn'], name='Nominal Curve (Pn-Mn)',
                         line=dict(color='rgba(150,150,150,0.5)', dash='dash')))

# จุดที่แรงกระทำจริง
fig.add_trace(go.Scatter(x=[mu], y=[pu], mode='markers+text', name='Design Load',
                         text=["(Mu, Pu)"], textposition="top right",
                         marker=dict(color='red', size=12, symbol='x')))

fig.update_layout(xaxis_title="Moment Mn (ton-m)", yaxis_title="Axial Pn (tons)", 
                  height=700, template="plotly_white")

st.plotly_chart(fig, use_container_width=True)

# Status Check
safe_p = np.interp(mu, df['phiMn'], df['phiPn_capped'])
if pu <= safe_p and pu <= pn_limit:
    st.success(f"✅ PASS: Your load is inside the capacity envelope.")
else:
    st.error(f"❌ FAIL: Load exceeds section capacity.")
