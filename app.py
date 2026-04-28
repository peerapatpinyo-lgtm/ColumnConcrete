import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class SeniorRCEngine:
    def __init__(self, fc, fy, b, h, db, n_bars, cover):
        self.fc, self.fy, self.b, self.h = fc, fy, b, h
        self.es = 2.04e6
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))
        
        # 1. Discrete Bar Layout (กระจายเหล็ก 4 ด้าน)
        self.bars = []
        as_bar = np.pi * (db/20)**2 / 4
        d_prime = cover + 0.9 + (db/20)
        
        # วางเหล็ก 4 มุมก่อน
        self.bars.extend([{'as': as_bar, 'd': d_prime}, {'as': as_bar, 'd': h - d_prime}] * 2)
        # กระจายเหล็กที่เหลือตรงกลาง (ถ้ามี)
        remaining = n_bars - 4
        if remaining > 0:
            # วางแทรกในเลเยอร์ บน-ล่าง และ ตรงกลาง
            mid_bars = remaining / 2
            self.bars.extend([{'as': as_bar, 'd': d_prime}] * int(mid_bars/2))
            self.bars.extend([{'as': as_bar, 'd': h/2}] * int(remaining - (mid_bars/2)*2))
            self.bars.extend([{'as': as_bar, 'd': h - d_prime}] * int(mid_bars/2))

    def solve(self):
        points = []
        # วนลูปค่า c (Neutral Axis) ให้ละเอียดมาก
        c_list = np.logspace(np.log10(0.1), np.log10(self.h * 5), 300)
        
        for c in c_list:
            # Concrete Force
            a = min(self.beta1 * c, self.h)
            Cc = 0.85 * self.fc * a * self.b
            Mc = Cc * (self.h/2 - a/2) # Moment รอบกึ่งกลางหน้าตัด
            
            # Steel Forces
            Pn_s, Mn_s = 0, 0
            et = 0
            max_d = max(b['d'] for b in self.bars)
            
            for bar in self.bars:
                eps = 0.003 * (c - bar['d']) / c
                fs = np.clip(eps * self.es, -self.fy, self.fy)
                Pn_s += bar['as'] * fs
                Mn_s += bar['as'] * fs * (self.h/2 - bar['d'])
                if bar['d'] == max_d: et = abs(eps) if c < bar['d'] else 0

            # Combined
            Pn, Mn = (Cc + Pn_s)/1000, abs(Mc + Mn_s)/100000
            
            # Phi Factor (ACI 318-19)
            ey = self.fy / self.es
            phi = 0.65 if et <= ey else 0.90 if et >= 0.005 else 0.65 + 0.25*(et - ey)/(0.005 - ey)
            
            points.append({'Pn': Pn, 'Mn': Mn, 'phiPn': phi*Pn, 'phiMn': phi*Mn})

        # จุด Pure Tension
        ast_total = sum(b['as'] for b in self.bars)
        p_tension = -ast_total * self.fy / 1000
        points.append({'Pn': p_tension, 'Mn': 0, 'phiPn': 0.9 * p_tension, 'phiMn': 0})

        # จุด Pure Compression (Capped)
        Po = (0.85 * self.fc * (self.b * self.h - ast_total) + self.fy * ast_total) / 1000
        phiPn_max = 0.80 * 0.65 * Po
        
        df = pd.DataFrame(points).sort_values('Pn', ascending=False)
        return df, phiPn_max, Po

# --- UI Setup ---
st.set_page_config(page_title="Senior RC Designer", layout="wide")
st.title("👨‍💼 Professional RC Column Design System")

with st.sidebar:
    st.header("Materials & Geometry")
    fc = st.number_input("f'c (ksc)", 210, 560, 280)
    fy = st.number_input("fy (ksc)", 3000, 5000, 4000)
    b = st.slider("Width b (cm)", 20, 100, 40)
    h = st.slider("Depth h (cm)", 20, 100, 60)
    db = st.selectbox("Bar Size (mm)", [16, 20, 25, 28, 32], index=2)
    n_bars = st.number_input("Number of Bars", 4, 32, 8, step=4)

engine = SeniorRCEngine(fc, fy, b, h, db, n_bars, 4.0)
df_pm, phi_pn_max, po = engine.solve()

# --- Visual Layout ---
col_graph, col_data = st.columns([2, 1])

with col_graph:
    fig = go.Figure()
    # Nominal
    fig.add_trace(go.Scatter(x=df_pm['Mn'], y=df_pm['Pn'], name="Nominal (Pn-Mn)", 
                             line=dict(color='gray', dash='dash')))
    # Design (Capped)
    # ตัดยอดกราฟที่ phi_pn_max
    df_design = df_pm[df_pm['phiPn'] <= phi_pn_max].copy()
    # เพิ่มจุดหักที่เส้น Cap
    cap_x = np.interp(phi_pn_max, df_pm['phiPn'][::-1], df_pm['phiMn'][::-1])
    
    fig.add_trace(go.Scatter(x=[0, cap_x], y=[phi_pn_max, phi_pn_max], 
                             line=dict(color='navy', width=3), showlegend=False))
    fig.add_trace(go.Scatter(x=df_design['phiMn'], y=df_design['phiPn'], 
                             fill='tozeroy', name="Design (ΦPn-ΦMn)", line=dict(color='navy', width=3)))

    fig.update_layout(xaxis_title="Moment (ton-m)", yaxis_title="Axial (ton)", height=650, plot_bgcolor='white')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    st.plotly_chart(fig, use_container_width=True)

with col_data:
    st.subheader("Section Properties")
    st.write(f"**Gross Area ($A_g$):** {b*h} cm²")
    st.write(f"**Steel Area ($A_{{st}}$):** {sum(b['as'] for b in engine.bars):.2f} cm²")
    st.write(f"**Steel Ratio ($\rho$):** {(sum(b['as'] for b in engine.bars)/(b*h))*100:.2f}%")
    
    st.divider()
    st.subheader("Key Capacities")
    st.metric("Max Axial (ΦPn,max)", f"{phi_pn_max:.1f} ton")
    st.metric("Pure Tension (ΦTn)", f"{0.9 * (sum(b['as'] for b in engine.bars) * fy / 1000):.1f} ton")
