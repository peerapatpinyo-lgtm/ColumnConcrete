import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class RCColumnEngine:
    def __init__(self, fc, fy, b, h, db_mm, n_bars, cover_cm, L_m=0, k=1.0):
        self.fc, self.fy, self.b, self.h = fc, fy, b, h
        self.L, self.k = L_m * 100, k # แปลงเป็น cm
        self.es = 2.04e6  # modulus of elasticity (ksc)
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))
        
        # จัดเลเยอร์เหล็กเสริม (สมมติเป็นเสาสมมาตร จัดวาง 2 แถวขอบบน-ล่าง)
        as_single = np.pi * (db_mm/20)**2 / 4
        d_prime = cover_cm + 0.9 + (db_mm/20) # ระยะถึง ศก. เหล็กบน
        d = h - d_prime                     # ระยะถึง ศก. เหล็กล่าง
        
        self.layers = [
            {'as': (n_bars/2) * as_single, 'd': d_prime},
            {'as': (n_bars/2) * as_single, 'd': d}
        ]

    def solve(self):
        points = []
        # วนลูปค่า Neutral Axis (c) ตั้งแต่หน้าตัดรับแรงอัดเต็มๆ จนถึงรับแรงดึงเต็มๆ
        # เพื่อสร้างกราฟที่สมบูรณ์แบบ
        c_values = np.concatenate([
            np.linspace(self.h * 2, self.h, 50),     # ช่วงแรงอัดสูง
            np.linspace(self.h, 0.1, 200)            # ช่วงรอยต่อจนถึงแรงดึง
        ])

        for c in c_values:
            # 1. แรงอัดคอนกรีต (Whitney Stress Block)
            a = min(self.beta1 * c, self.h)
            force_c = 0.85 * self.fc * a * self.b
            mom_c = force_c * (self.h/2 - a/2) # Moment รอบจุดศูนย์กลางหน้าตัด (Plastic Centroid)

            # 2. แรงในเหล็กเสริมแต่ละชั้น
            force_s_total = 0
            mom_s_total = 0
            et = 0 # Net tensile strain

            for layer in self.layers:
                eps = 0.003 * (c - layer['d']) / c
                fs = np.clip(eps * self.es, -self.fy, self.fy)
                f_s = layer['as'] * fs
                force_s_total += f_s
                mom_s_total += f_s * (self.h/2 - layer['d'])
                
                # เก็บค่า strain ของเหล็กชั้นที่อยู่ไกลจากแรงอัดที่สุด (ด้านล่าง)
                if layer['d'] == max(l['d'] for l in self.layers):
                    et = abs(0.003 * (layer['d'] - c) / c) if c < layer['d'] else 0

            pn = (force_c + force_s_total) / 1000 # ton
            mn = (mom_c + mom_s_total) / 100000  # ton-m
            
            # 3. คำนวณ Phi Factor (ACI 318-19)
            ey = self.fy / self.es
            phi = 0.65 if et <= ey else 0.90 if et >= 0.005 else 0.65 + 0.25*(et - ey)/(0.005 - ey)
            
            points.append({'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn, 'et': et})

        # เพิ่มจุด Pure Tension (ด้านล่างสุดของกราฟ)
        ast_total = sum(l['as'] for l in self.layers)
        p_tension = -ast_total * self.fy / 1000
        points.append({'Pn': p_tension, 'Mn': 0, 'phiPn': 0.9 * p_tension, 'phiMn': 0})

        # คำนวณจุด Pure Compression Cap (เส้นหัวตัด)
        ag = self.b * self.h
        po = (0.85 * self.fc * (ag - ast_total) + self.fy * ast_total) / 1000
        phi_pn_max = 0.65 * 0.80 * po # สำหรับเสาปลอกเดี่ยว

        df = pd.DataFrame(points).sort_values('Pn', ascending=False)
        return df, phi_pn_max

    def check_slenderness(self, pu, mu):
        """ตรวจสอบผลของเสายาว (Moment Magnification)"""
        r = 0.3 * self.h
        slenderness = (self.k * self.L) / r
        if slenderness <= 22: return mu # เสาสั้น
        
        EI = (0.4 * self.fc * (self.b * self.h**3 / 12)) / (1 + 0.5)
        pc = (np.pi**2 * EI) / (self.k * self.L)**2 / 1000
        delta = max(1.0, 1.0 / (1 - (pu / (0.75 * pc))))
        return mu * delta

# --- Streamlit Interface ---
st.set_page_config(page_title="Professional RC Column Designer", layout="wide")
st.title("🏗️ Professional RC Column Design (Industrial Grade)")

with st.sidebar:
    st.header("⚙️ หน้าตัดและวัสดุ")
    fc = st.number_input("กำลังอัดคอนกรีต f'c (ksc)", 210, 560, 280)
    fy = st.number_input("กำลังดึงเหล็ก fy (ksc)", 3000, 5000, 4000)
    b = st.slider("ความกว้าง b (cm)", 20, 100, 40)
    h = st.slider("ความลึก h (cm)", 20, 100, 60)
    
    st.header("🔩 เหล็กเสริมยืน")
    db = st.selectbox("ขนาดเหล็ก (mm)", [16, 20, 25, 28, 32], index=2)
    n_bars = st.number_input("จำนวนเหล็ก (เส้น)", 4, 32, 8, step=2)
    
    st.header("📏 ความสูงและจุดรองรับ")
    L_m = st.number_input("ความสูงเสา (m)", 1.0, 15.0, 5.0)
    k_val = st.selectbox("ค่า K (ตามสภาพจุดรองรับ)", [0.7, 1.0, 1.2, 2.0], index=1)

# คำนวณผล
engine = RCColumnEngine(fc, fy, b, h, db, n_bars, 4.0, L_m, k_val)
df_pm, phi_pn_max = engine.solve()

# ส่วนแสดงผล
tab1, tab2 = st.tabs(["📉 Interaction Diagram", "📋 รายการคำนวณ"])

with tab1:
    st.subheader("P-M Interaction Diagram")
    
    # แก้ไขแรงที่กระทำ
    st.write("ใส่แรงที่กระทำ (Ultimate Load):")
    load_input = pd.DataFrame([{'Case': 'Load 1', 'Pu (ton)': 150.0, 'Mu (ton-m)': 20.0}], index=[0])
    edited_loads = st.data_editor(load_input, num_rows="dynamic")
    
    # สร้างกราฟ
    fig = go.Figure()
    
    # 1. เส้น Nominal (เส้นประสีเทา - แบบที่เรียนในห้อง)
    fig.add_trace(go.Scatter(x=df_pm['Mn'], y=df_pm['Pn'], name="Nominal (Pn-Mn)",
                             line=dict(color='gray', dash='dash')))
    
    # 2. เส้น Design (เส้นทึบสีน้ำเงิน - ที่ใช้จริง)
    fig.add_trace(go.Scatter(x=df_pm['phiMn'], y=df_pm['phiPn'].clip(upper=phi_pn_max),
                             fill='tozeroy', name="Design (ΦPn-ΦMn)", line=dict(color='navy', width=3)))
    
    # 3. พล็อตจุดแรงกระทำ
    for _, row in edited_loads.iterrows():
        mu_magnified = engine.check_slenderness(row['Pu (ton)'], row['Mu (ton-m)'])
        fig.add_trace(go.Scatter(x=[mu_magnified], y=[row['Pu (ton)']], mode='markers+text',
                                 text=[row['Case']], textposition="top center",
                                 marker=dict(color='red', size=12, symbol='x'), name="Input Load"))

    fig.update_layout(xaxis_title="Moment (ton-m)", yaxis_title="Axial Load (ton)", height=700)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("สถิติหน้าตัดเสา")
    col1, col2, col3 = st.columns(3)
    col1.metric("พื้นที่หน้าตัด (Ag)", f"{b*h} cm²")
    col2.metric("พื้นที่เหล็ก (Ast)", f"{sum(l['as'] for l in engine.layers):.2f} cm²")
    rho = (sum(l['as'] for l in engine.layers) / (b*h)) * 100
    col3.metric("อัตราส่วนเหล็ก (ρ)", f"{rho:.2f} %")
    
    st.write("---")
    st.markdown("""
    **คำอธิบายกราฟ:**
    * **เส้นประสีเทา (Nominal):**คือกำลังตามทฤษฎี (Pn, Mn) ที่ยังไม่คูณค่าความปลอดภัย เหมือนในตำราเรียน
    * **เส้นทึบสีน้ำเงิน (Design):** คือกำลังที่ยอมให้ใช้จริง (ΦPn, ΦMn) ซึ่งมีการตัดยอดกราฟ (Compression Cap) และลดกำลังด้วยค่า Φ ตามมาตรฐาน ACI/วสท.
    * **จุด X สีแดง:** คือแรงที่คุณป้อนเข้าไป หากจุดตกอยู่ในพื้นที่สีน้ำเงิน แสดงว่าเสา **"ปลอดภัย"**
    """)
