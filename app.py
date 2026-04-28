import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class RCColumnProfessional:
    def __init__(self, fc, fy, b, h, db_mm, n_bars, cover_cm):
        self.fc, self.fy, self.b, self.h = fc, fy, b, h
        self.Es = 2.04e6
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))
        
        # จัดเลเยอร์เหล็ก (สมมติจัดสมมาตร 2 ฝั่งเพื่อความเสถียรของกราฟ)
        as_single = (np.pi * (db_mm/10)**2) / 4
        d_prime = cover_cm + 0.9 + (db_mm/20) # ระยะถึงเหล็กบน
        d = h - d_prime                     # ระยะถึงเหล็กล่าง
        
        self.layers = [
            {'as': (n_bars/2) * as_single, 'd': d_prime},
            {'as': (n_bars/2) * as_single, 'd': d}
        ]

    def solve(self):
        results = []
        # วนลูปค่า c อย่างเป็นระบบ เพื่อให้ได้เส้นกราฟที่ลากต่อเนื่องกัน (ไม่ตัดไปมา)
        # ตั้งแต่ c น้อยมาก (เหล็กครากด้วยแรงดึง) จนถึง c มาก (คอนกรีตเต็มหน้าตัด)
        c_values = np.linspace(0.01, self.h * 2, 500)
        
        for c in c_values:
            a = min(self.beta1 * c, self.h)
            
            # 1. แรงและโมเมนต์จากคอนกรีต
            Cc = 0.85 * self.fc * a * self.b
            Mc = Cc * (self.h/2 - a/2)
            
            # 2. แรงและโมเมนต์จากเหล็กเสริม
            Pn_s = 0
            Mn_s = 0
            et = 0 # Strain เหล็กชั้นล่าง
            
            for layer in self.layers:
                eps_s = 0.003 * (c - layer['d']) / c
                fs = np.clip(eps_s * self.Es, -self.fy, self.fy)
                Fsi = layer['as'] * fs
                Pn_s += Fsi
                Mn_s += Fsi * (self.h/2 - layer['d'])
                
                # หา et เพื่อคำนวณ Phi
                if layer['d'] == max(l['d'] for l in self.layers):
                    et = abs(0.003 * (layer['d'] - c) / c) if c < layer['d'] else 0

            # 3. รวมผล (Nominal)
            pn = (Cc + Pn_s) / 1000 # ton
            mn = (Mc + Mn_s) / 100000 # ton-m
            
            # 4. คำนวณ Phi Factor (ACI 318-19)
            ey = self.fy / self.Es
            phi = 0.65 if et <= ey else 0.90 if et >= 0.005 else 0.65 + 0.25*(et - ey)/(0.005 - ey)
            
            results.append({'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn})

        # เพิ่มจุด Pure Tension (สำคัญ: เพื่อให้กราฟมาจบที่แกน Y ด้านล่าง)
        total_as = sum(l['as'] for l in self.layers)
        tn = -total_as * self.fy / 1000
        results.append({'Pn': tn, 'Mn': 0, 'phiPn': 0.9 * tn, 'phiMn': 0})

        # คำนวณจุด Pure Compression Cap (Pn_max)
        po = (0.85 * self.fc * (self.b * self.h - total_as) + self.fy * total_as) / 1000
        phi_pn_max = 0.65 * 0.80 * po
        
        df = pd.DataFrame(results).sort_values('Pn', ascending=True) # เรียงเพื่อให้ลากเส้นสวยงาม
        return df, phi_pn_max

# --- STREAMLIT UI ---
st.set_page_config(page_title="Refined Column Design")
st.title("🏗️ Textbook-Correct Interaction Diagram")

with st.sidebar:
    fc = st.number_input("f'c (ksc)", value=280)
    fy = st.number_input("fy (ksc)", value=4000)
    b = st.slider("b (cm)", 20, 100, 40)
    h = st.slider("h (cm)", 20, 100, 60)
    db = st.selectbox("Bar Size (mm)", [16, 20, 25, 28, 32], index=2)
    n_bars = st.number_input("Bars", 4, 32, 8, step=2)

engine = RCColumnProfessional(fc, fy, b, h, db, n_bars, 4.0)
df, phi_pn_max = engine.solve()

# --- Plotly Graph ---
fig = go.Figure()

# 1. เส้น Nominal (Pn-Mn)
fig.add_trace(go.Scatter(x=df['Mn'], y=df['Pn'], name="Nominal (Pn-Mn)",
                         line=dict(color='gray', dash='dash')))

# 2. เส้น Design (Phi Pn - Phi Mn) พร้อมตัดยอด Pn_max
df_design = df.copy()
df_design['phiPn'] = df_design['phiPn'].clip(upper=phi_pn_max)

fig.add_trace(go.Scatter(x=df_design['phiMn'], y=df_design['phiPn'], 
                         fill='tozeroy', name="Design (ΦPn-ΦMn)",
                         line=dict(color='navy', width=3)))

fig.update_layout(xaxis_title="Moment (ton-m)", yaxis_title="Axial Load (ton)",
                  plot_bgcolor='white', height=700)
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')

st.plotly_chart(fig, use_container_width=True)

st.info(f"**Calculated Po:** {phi_pn_max/0.65/0.8:.1f} ton | **ΦPn,max:** {phi_pn_max:.1f} ton")
