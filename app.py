import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class RCColumnEngine:
    def __init__(self, fc, fy, b, h, db_mm, n_bars, cover_cm, L_m=0, k=1.0):
        self.fc, self.fy, self.b, self.h = fc, fy, b, h
        self.L, self.k = L_m * 100, k
        self.es = 2.04e6 
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))
        self.db_mm = db_mm
        self.n_bars = n_bars
        self.cover = cover_cm
        
        # จัดเลเยอร์เหล็ก (Top-Bottom)
        as_single = np.pi * (db_mm/20)**2 / 4
        self.d_prime = cover_cm + 0.9 + (db_mm/20)
        self.d = h - self.d_prime
        self.ast = n_bars * as_single
        
        self.layers = [
            {'as': (n_bars/2) * as_single, 'd': self.d_prime},
            {'as': (n_bars/2) * as_single, 'd': self.d}
        ]

    def solve_interaction(self):
        # ... (เหมือนเดิมจากโค้ดชุดก่อน) ...
        points = []
        c_values = np.concatenate([np.linspace(self.h * 2, self.h, 50), np.linspace(self.h, 0.1, 200)])
        for c in c_values:
            a = min(self.beta1 * c, self.h)
            force_c = 0.85 * self.fc * a * self.b
            mom_c = force_c * (self.h/2 - a/2)
            f_s_total, m_s_total, et = 0, 0, 0
            for layer in self.layers:
                eps = 0.003 * (c - layer['d']) / c
                fs = np.clip(eps * self.es, -self.fy, self.fy)
                f_total = layer['as'] * fs
                f_s_total += f_total
                m_s_total += f_total * (self.h/2 - layer['d'])
                if layer['d'] == self.d: et = abs(0.003 * (self.d - c) / c) if c < self.d else 0
            pn, mn = (force_c + f_s_total) / 1000, (mom_c + m_s_total) / 100000
            ey = self.fy / self.es
            phi = 0.65 if et <= ey else 0.90 if et >= 0.005 else 0.65 + 0.25*(et - ey)/(0.005 - ey)
            points.append({'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn})
        
        po = (0.85 * self.fc * (self.b * self.h - self.ast) + self.fy * self.ast) / 1000
        return pd.DataFrame(points).sort_values('Pn', ascending=False), 0.65 * 0.80 * po

    def design_shear(self, Vu_ton, stirrup_db=9):
        """ คำนวณเหล็กปลอกตามแรงเฉือน """
        phi = 0.75
        d = self.d
        # 1. กำลังของคอนกรีต Vc = 0.53 * sqrt(f'c) * b * d
        vc = 0.53 * np.sqrt(self.fc) * self.b * d / 1000 # ton
        
        if Vu_ton <= (phi * vc / 2):
            return "ไม่ต้องการเหล็กปลอกตามการคำนวณ (ใช้ตามมาตรฐานขั้นต่ำ)", 0, vc
        
        vs_required = (Vu_ton / phi) - vc
        av = 2 * (np.pi * (stirrup_db/20)**2 / 4) # เหล็กปลอก 2 ขา
        
        if vs_required <= 0:
            s_req = min(d/2, 60.0, 16*(self.db_mm/10), 48*(stirrup_db/10), self.b)
        else:
            s_calc = (av * self.fy * d / (vs_required * 1000))
            s_req = min(s_calc, d/2, 60.0)
            
        return f"ระยะห่างเหล็กปลอกแนะนำ: {s_req:.1f} cm", s_req, vc

    def draw_section(self):
        fig, ax = plt.subplots(figsize=(4, 4))
        # วาดคอนกรีต
        ax.add_patch(plt.Rectangle((0, 0), self.b, self.h, color='#E0E0E0', ec='black', lw=3))
        # วาดเหล็กปลอก (Tie)
        tie_offset = self.cover
        ax.add_patch(plt.Rectangle((tie_offset, tie_offset), self.b-2*tie_offset, self.h-2*tie_offset, 
                                   fill=False, ec='red', lw=1.5, ls='--'))
        # วาดเหล็กยืน (Top & Bottom)
        x_pos = np.linspace(self.d_prime, self.b - self.d_prime, int(self.n_bars/2))
        for x in x_pos:
            ax.add_patch(plt.Circle((x, self.d_prime), self.db_mm/20, color='black'))
            ax.add_patch(plt.Circle((x, self.h - self.d_prime), self.db_mm/20, color='black'))
            
        ax.set_xlim(-5, self.b+5); ax.set_ylim(-5, self.h+5)
        ax.set_aspect('equal'); ax.axis('off')
        return fig

# --- Streamlit UI ---
st.set_page_config(page_title="Industrial Column Pro", layout="wide")
st.title("🏗️ RC Column Expert System (Level 2)")

with st.sidebar:
    st.header("1. Geometry & Materials")
    fc = st.number_input("f'c (ksc)", value=280)
    fy = st.number_input("fy (ksc)", value=4000)
    b, h = st.slider("Width b (cm)", 20, 100, 40), st.slider("Depth h (cm)", 20, 100, 60)
    db = st.selectbox("Main Bar (mm)", [16, 20, 25, 28, 32], index=2)
    n_bars = st.number_input("Total Bars", 4, 32, 8, step=2)
    
    st.header("2. Shear Reinforcement")
    stirrup_db = st.selectbox("Stirrup Size (mm)", [6, 9, 12], index=1)
    vu_input = st.number_input("Design Shear Vu (ton)", value=15.0)

engine = RCColumnEngine(fc, fy, b, h, db, n_bars, 4.0)
df_pm, phi_pn_max = engine.solve_interaction()

col1, col2 = st.columns([2, 1])

with col1:
    tab_pm, tab_shear = st.tabs(["📉 Interaction Diagram", "🛡️ Shear Design"])
    with tab_pm:
        fig_pm = go.Figure()
        fig_pm.add_trace(go.Scatter(x=df_pm['phiMn'], y=df_pm['phiPn'].clip(upper=phi_pn_max), 
                                     fill='tozeroy', name='Design Capacity', line=dict(color='navy')))
        fig_pm.update_layout(xaxis_title="Moment (ton-m)", yaxis_title="Axial Load (ton)")
        st.plotly_chart(fig_pm, use_container_width=True)
        
    with tab_shear:
        st.subheader("การคำนวณแรงเฉือน")
        msg, s_spacing, vc_cap = engine.design_shear(vu_input, stirrup_db)
        st.info(f"กำลังรับแรงเฉือนของคอนกรีต (ΦVc): {0.75 * vc_cap:.2f} ton")
        st.success(msg)
        
with col2:
    st.subheader("Section Drawing")
    st.pyplot(engine.draw_section())
    st.write(f"**Section:** {b}x{h} cm")
    st.write(f"**Reinforcement:** {n_bars}-DB{db}")
    st.write(f"**Ties:** RB{stirrup_db} @ {s_spacing if s_spacing > 0 else '-':.0f} cm")
