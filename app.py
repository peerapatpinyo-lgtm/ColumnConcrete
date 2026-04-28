import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class RCCapacityEngine:
    def __init__(self, fc, fy, b, h, db, n_bars, cover):
        self.fc, self.fy, self.b, self.h = fc, fy, b, h
        self.es = 2040000 # ksc
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))
        
        # จัดการเหล็กเสริม (สมมติจัด 2 แถวเพื่อความแม่นยำ)
        self.as_total = n_bars * (np.pi * (db/10)**2 / 4)
        self.d_prime = cover + 0.9 + (db/20) # ระยะถึงจุด ศก. เหล็ก
        self.d = h - self.d_prime
        
        # 1. คำนวณ Plastic Centroid (Xpc) - สำคัญที่สุดเพื่อให้กราฟไม่เบี้ยว
        # สำหรับหน้าตัดสมมาตร Xpc = h/2
        self.x_pc = h / 2 

    def analyze_section(self):
        results = []
        
        # จุดที่ 1: Pure Compression (Po)
        po = (0.85 * self.fc * (self.b * self.h - self.as_total) + self.fy * self.as_total) / 1000
        phi_pn_max = 0.65 * 0.80 * po # Capped for Tied Column
        
        # วนลูปค่า c (Neutral Axis) ให้ละเอียดขึ้น
        # จาก c มหาศาล (อัดเต็มหน้าตัด) ถึง c เข้าใกล้ 0 (ดึงเต็มหน้าตัด)
        c_list = [1e10] + list(np.geomspace(self.h * 2, 0.1, 200)) + [0]
        
        for c in c_list:
            if c >= 1e10: # Pure Compression
                pn, mn, et = po, 0, 0
            elif c == 0: # Pure Tension
                pn, mn, et = -self.as_total * self.fy / 1000, 0, 1.0
            else:
                a = min(self.beta1 * c, self.h)
                cc = 0.85 * self.fc * a * self.b
                
                # Strain Compatibility
                eps_s1 = 0.003 * (c - self.d_prime) / c
                eps_t = 0.003 * (self.d - c) / c
                
                fs1 = np.clip(eps_s1 * self.es, -self.fy, self.fy)
                fs2 = np.clip(eps_t * self.es, -self.fy, self.fy)
                
                # กำลังภายใน (Pn, Mn รอบ Plastic Centroid)
                pn = (cc + (self.as_total/2 * fs1) + (self.as_total/2 * fs2)) / 1000
                mn = (cc * (self.x_pc - a/2) + (self.as_total/2 * fs1) * (self.x_pc - self.d_prime) - 
                      (self.as_total/2 * fs2) * (self.d - self.x_pc)) / 100000
                et = abs(eps_t) if eps_t < 0 else 0

            # Phi Factor (ACI 318-19)
            ey = self.fy / self.es
            phi = 0.65 if et <= ey else 0.90 if et >= 0.005 else 0.65 + 0.25*(et - ey)/(0.005 - ey)
            
            results.append({'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn, 'et': et})

        df = pd.DataFrame(results)
        return df, phi_pn_max

# --- UI Setup ---
st.set_page_config(page_title="Professional RC Analysis", layout="wide")
st.title("🏗️ Rigorous RC Sectional Analysis")

with st.sidebar:
    st.header("Section Parameters")
    fc = st.number_input("f'c (ksc)", 210, 560, 280)
    fy = st.number_input("fy (ksc)", 3000, 5000, 4000)
    b, h = st.slider("Width (cm)", 20, 100, 40), st.slider("Depth (cm)", 20, 100, 50)
    db = st.selectbox("DB (mm)", [16, 20, 25, 28, 32], index=2)
    n_bars = st.number_input("Total Bars", 4, 32, 8, step=2)

engine = RCCapacityEngine(fc, fy, b, h, db, n_bars, 4.0)
df_pm, p_cap = engine.analyze_section()

# --- Plotly Graph ---
fig = go.Figure()

# เส้น Nominal (Pn-Mn) - เส้นประสีเทา
fig.add_trace(go.Scatter(x=df_pm['Mn'], y=df_pm['Pn'], name="Nominal Capacity", 
                         line=dict(color='silver', dash='dash')))

# เส้น Design (phiPn-phiMn) - เส้นทึบสีน้ำเงิน พร้อมจุด Cap
fig.add_trace(go.Scatter(x=df_pm['phiMn'], y=df_pm['phiPn'].clip(upper=p_cap), 
                         fill='tozeroy', name="Design Capacity (Φ)", line=dict(color='navy', width=4)))

fig.update_layout(xaxis_title="Moment (ton-m)", yaxis_title="Axial Load (tons)", height=700,
                  hovermode="x unified", template="plotly_white")

st.plotly_chart(fig, use_container_width=True)

st.info(f"**Engineering Note:** กราฟนี้ใช้การวิเคราะห์แบบ Incremental Strain โดยพิจารณาพฤติกรรมจริงของคอนกรีตและเหล็กเสริมทุกสถานะ ตั้งแต่ Pure Compression จนถึง Pure Tension")
