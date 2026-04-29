import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import interp1d

class RCColumnProBiaxial:
    def __init__(self, shape, layout, b, h, fc, fy, db_mm, n_bars, nx, ny, cover_cm):
        self.shape = shape
        self.layout = layout
        self.b, self.h = b, h
        self.fc, self.fy = fc, fy
        self.Es = 2.04e6
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))
        
        if self.shape == "Rectangular":
            self.Ag = self.b * self.h
            self.Igx = (self.b * self.h**3) / 12  # Bending about X (depth = h)
            self.Igy = (self.h * self.b**3) / 12  # Bending about Y (depth = b)
            self.rx = 0.3 * self.h
            self.ry = 0.3 * self.b
        else: 
            self.D = self.h
            self.Ag = (np.pi * self.D**2) / 4
            self.Igx = self.Igy = (np.pi * self.D**4) / 64
            self.rx = self.ry = 0.25 * self.D
            
        self.Ec = 15100 * np.sqrt(self.fc)
        self.db_cm = db_mm / 10
        self.as_single = (np.pi * self.db_cm**2) / 4
        self.d_prime = cover_cm + 0.9 + (self.db_cm/2) 
        
        self.bars = []
        if self.shape == "Rectangular":
            x_min, x_max = -self.b/2 + self.d_prime, self.b/2 - self.d_prime
            y_min, y_max = -self.h/2 + self.d_prime, self.h/2 - self.d_prime
            
            if self.layout == "2-Faces (Top/Bottom)":
                x_coords = np.linspace(x_min, x_max, n_bars // 2)
                for x in x_coords:
                    self.bars.append({'x': x, 'y': y_max}) 
                    self.bars.append({'x': x, 'y': y_min}) 
            elif self.layout == "4-Faces (Uniform)":
                x_coords = np.linspace(x_min, x_max, nx)
                for x in x_coords:
                    self.bars.append({'x': x, 'y': y_max}) 
                    self.bars.append({'x': x, 'y': y_min})
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

    def solve_pm(self, axis='X'):
        if self.shape == "Circular":
            depth, width = self.D, self.D
            get_y = lambda bar: bar['y']
        else:
            if axis == 'X':
                depth, width = self.h, self.b
                get_y = lambda bar: bar['y']
            else:
                depth, width = self.b, self.h
                get_y = lambda bar: bar['x'] # Swap axis for Y-bending

        dt = depth/2 - min(get_y(bar) for bar in self.bars)
        results = []
        c_values = np.concatenate([np.linspace(0.001, dt, 200), np.linspace(dt, depth * 3, 200)])
        
        for c in c_values:
            a = min(self.beta1 * c, depth)
            
            if self.shape == "Rectangular":
                Cc = 0.85 * self.fc * a * width
                Mc = Cc * (depth/2 - a/2)
            else: 
                R = self.D / 2
                if a >= self.D:
                    Ac, y_bar = np.pi * R**2, 0
                else:
                    theta = 2 * np.arccos((R - a) / R)
                    Ac = (R**2 / 2) * (theta - np.sin(theta))
                    y_bar = (4 * R * np.sin(theta/2)**3) / (3 * (theta - np.sin(theta))) if Ac > 0 else R
                Cc = 0.85 * self.fc * Ac
                Mc = Cc * y_bar 
            
            Pn_s, Mn_s = 0, 0
            for bar in self.bars:
                d_i = depth/2 - get_y(bar) 
                eps_s = 0.003 * (c - d_i) / c
                fs = np.clip(eps_s * self.Es, -self.fy, self.fy)
                Fsi = self.as_single * fs
                Pn_s += Fsi
                Mn_s += Fsi * get_y(bar) 

            pn = (Cc + Pn_s) / 1000 
            mn = (Mc + Mn_s) / 100000 
            
            et = 0.003 * (dt - c) / c
            ey = self.fy / self.Es
            phi_comp = 0.75 if self.shape == "Circular" else 0.65 
            
            if et >= 0.005: phi = 0.90
            elif et <= ey: phi = phi_comp
            else: phi = phi_comp + (0.90 - phi_comp) * (et - ey) / (0.005 - ey)
            
            results.append({'c': c, 'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn})

        tn = -self.total_as * self.fy / 1000
        results.append({'c': 0, 'Pn': tn, 'Mn': 0, 'phiPn': 0.9 * tn, 'phiMn': 0})

        po = (0.85 * self.fc * (self.Ag - self.total_as) + self.fy * self.total_as) / 1000
        phi_max_factor = 0.85 if self.shape == "Circular" else 0.80
        phi_comp = 0.75 if self.shape == "Circular" else 0.65
        phi_pn_max = phi_comp * phi_max_factor * po

        df = pd.DataFrame(results).sort_values('Pn', ascending=True)
        df['phiPn'] = df['phiPn'].clip(upper=phi_pn_max)
        return df, phi_pn_max
        
    def slenderness_magnifier(self, Pu, K, Lu_m, axis, Cm, beta_d):
        Lu_cm = Lu_m * 100
        r = self.rx if axis == 'X' else self.ry
        Ig = self.Igx if axis == 'X' else self.Igy
        
        kl_r = (K * Lu_cm) / r
        EI = (0.4 * self.Ec * Ig) / (1 + beta_d)
        Pc = (np.pi**2 * EI) / (K * Lu_cm)**2 / 1000
        
        if Pu >= (0.75 * Pc):
            delta = 999.9 
        else:
            delta = max(1.0, Cm / (1 - (Pu / (0.75 * Pc))))
        return kl_r, Pc, delta

# --- STREAMLIT UI ---
st.set_page_config(page_title="Ultimate RC Column", layout="wide")
st.title("🏗️ RC Column (Biaxial & Sway Analysis)")

col1, col2 = st.columns([1, 2.5])

with col1:
    with st.expander("1. Section & Reinforcement", expanded=True):
        shape = st.radio("Section Shape", ["Rectangular", "Circular"], horizontal=True)
        fc = st.number_input("f'c (ksc)", value=280)
        fy = st.number_input("fy (ksc)", value=4000)
        
        if shape == "Rectangular":
            c1, c2 = st.columns(2)
            b = c1.number_input("Width, b (X-axis, cm)", value=40)
            h = c2.number_input("Depth, h (Y-axis, cm)", value=60)
            layout = st.selectbox("Rebar Layout", ["4-Faces (Uniform)", "2-Faces (Top/Bottom)"])
            
            if layout == "2-Faces (Top/Bottom)":
                n_bars = st.number_input("Total Bars (Even)", 4, 40, 8, step=2)
                nx, ny = 0, 0
            else:
                c3, c4 = st.columns(2)
                nx = c3.number_input("Bars in X", 2, 20, 3)
                ny = c4.number_input("Bars in Y", 2, 20, 4)
                n_bars = (2 * nx) + (2 * ny) - 4
        else:
            b = h = st.number_input("Diameter, D (cm)", value=50)
            layout = "Circular"
            n_bars = st.number_input("Total Bars (min 6)", 6, 60, 8, step=1)
            nx, ny = 0, 0
            
        db = st.selectbox("Bar Size (mm)", [16, 20, 25, 28, 32], index=2)
        cover = st.number_input("Covering (cm)", value=4.0)

    with st.expander("2. Loads & Frame Type", expanded=True):
        Pu = st.number_input("Factored Axial, Pu (ton)", value=150.0)
        c5, c6 = st.columns(2)
        Mux = c5.number_input("Mux (ton-m)", value=15.0, help="Moment about X-axis")
        Muy = c6.number_input("Muy (ton-m)", value=10.0, help="Moment about Y-axis")
        st.markdown("---")
        frame_type = st.radio("Frame Type", ["Non-Sway (Braced)", "Sway (Unbraced)"], horizontal=True)
        
        if frame_type == "Non-Sway (Braced)":
            st.write("Slenderness Parameters (Lu, K)")
            cx1, cx2, cx3 = st.columns(3)
            Lu_x = cx1.number_input("Lu (X) [m]", value=4.0, step=0.5)
            K_x = cx2.number_input("K (X)", value=1.0, step=0.1)
            Cm_x = cx3.number_input("Cm (X)", value=1.0, step=0.1)
            
            cy1, cy2, cy3 = st.columns(3)
            Lu_y = cy1.number_input("Lu (Y) [m]", value=4.0, step=0.5)
            K_y = cy2.number_input("K (Y)", value=1.0, step=0.1)
            Cm_y = cy3.number_input("Cm (Y)", value=1.0, step=0.1)
            
            beta_d = st.slider("Beta_d (Sustained Load Ratio)", 0.0, 1.0, 0.6)
            delta_sx, delta_sy = 1.0, 1.0
        else:
            st.warning("For Sway Frames, enter the Sway Magnification Factor (δs) directly.")
            cs1, cs2 = st.columns(2)
            delta_sx = cs1.number_input("δs (X-axis)", value=1.2, step=0.05)
            delta_sy = cs2.number_input("δs (Y-axis)", value=1.2, step=0.05)
            Lu_x = K_x = Cm_x = Lu_y = K_y = Cm_y = beta_d = 0 # Not used for direct sway

engine = RCColumnProBiaxial(shape, layout, b, h, fc, fy, db, n_bars, nx, ny, cover)
df_x, phi_pn_max = engine.solve_pm(axis='X')
df_y, _ = engine.solve_pm(axis='Y')

# --- Calculate Magnified Moments ---
e_min_x = Pu * (0.015 + 0.03 * (h / 100))
e_min_y = Pu * (0.015 + 0.03 * (b / 100))
Mu_x_dsgn = max(Mux, e_min_x)
Mu_y_dsgn = max(Muy, e_min_y)

if frame_type == "Non-Sway (Braced)":
    kl_rx, Pcx, del_x = engine.slenderness_magnifier(Pu, K_x, Lu_x, 'X', Cm_x, beta_d)
    kl_ry, Pcy, del_y = engine.slenderness_magnifier(Pu, K_y, Lu_y, 'Y', Cm_y, beta_d)
    Mcx = del_x * Mu_x_dsgn if kl_rx > 22 else Mu_x_dsgn
    Mcy = del_y * Mu_y_dsgn if kl_ry > 22 else Mu_y_dsgn
else:
    kl_rx = kl_ry = Pcx = Pcy = 0
    Mcx = delta_sx * Mu_x_dsgn
    Mcy = delta_sy * Mu_y_dsgn

# --- Biaxial Interaction Check (PCA Load Contour) ---
try:
    fx = interp1d(df_x['phiPn'], df_x['phiMn'], kind='linear', fill_value=0, bounds_error=False)
    fy_interp = interp1d(df_y['phiPn'], df_y['phiMn'], kind='linear', fill_value=0, bounds_error=False)
    
    phi_Mnox = fx(Pu)
    phi_Mnoy = fy_interp(Pu)
    
    if shape == "Circular":
        alpha = 2.0
        demand_ratio = (Mcx / phi_Mnox)**2 + (Mcy / phi_Mnoy)**2 if phi_Mnox > 0 else 999
    else:
        alpha = 1.5 # standard for rectangular
        demand_ratio = (Mcx / phi_Mnox)**alpha + (Mcy / phi_Mnoy)**alpha if phi_Mnox > 0 else 999
        
    is_safe = (demand_ratio <= 1.0) and (Pu <= phi_pn_max)
except:
    is_safe = False
    demand_ratio = 999

with col2:
    st.markdown("### 📋 Executive Biaxial Summary")
    
    m1, m2, m3 = st.columns(3)
    rho_pct = engine.rho * 100
    m1.metric("Steel Ratio (ρ)", f"{rho_pct:.2f} %", "OK" if 1 <= rho_pct <= 8 else "Fail", delta_color="normal" if 1 <= rho_pct <= 8 else "inverse")
    m2.metric("Design Mcx", f"{Mcx:.2f} t-m", f"Magnifier: {max(Mcx/Mu_x_dsgn, 1.0):.2f}x", delta_color="off")
    m3.metric("Design Mcy", f"{Mcy:.2f} t-m", f"Magnifier: {max(Mcy/Mu_y_dsgn, 1.0):.2f}x", delta_color="off")

    st.markdown("---")
    
    if is_safe:
        st.success(f"### ✅ **STATUS: SAFE**\nBiaxial Demand Ratio = **{demand_ratio:.3f}** ≤ 1.0")
    else:
        st.error(f"### ❌ **STATUS: UNSAFE**\nBiaxial Demand Ratio = **{demand_ratio:.3f}** > 1.0")


    tab1, tab2, tab3, tab4 = st.tabs(["🌐 3D/Biaxial Interaction", "📊 P-M Curves", "📐 Section", "📖 คู่มือพารามิเตอร์"])

    with tab1:
        st.markdown(f"**Load Contour at Pu = {Pu} ton (α = {alpha})**")
        fig_bi = go.Figure()
        
        # Plot Contour
        if phi_Mnox > 0 and phi_Mnoy > 0:
            x_vals = np.linspace(0, phi_Mnox, 100)
            y_vals = phi_Mnoy * (1 - (x_vals/phi_Mnox)**alpha)**(1/alpha)
            fig_bi.add_trace(go.Scatter(x=x_vals, y=y_vals, fill='tozeroy', name='Capacity Envelope', line=dict(color='purple', width=2)))
        
        # Plot Demand
        fig_bi.add_trace(go.Scatter(x=[Mcx], y=[Mcy], mode='markers', name='Applied Demand (Mcx, Mcy)', marker=dict(color='red', size=12, symbol='x')))
        
        fig_bi.update_layout(xaxis_title="Moment X-axis, Mcx (ton-m)", yaxis_title="Moment Y-axis, Mcy (ton-m)", plot_bgcolor='whitesmoke', height=450)
        st.plotly_chart(fig_bi, use_container_width=True)

    with tab2:
        fig_pm = go.Figure()
        fig_pm.add_trace(go.Scatter(x=df_x['phiMn'], y=df_x['phiPn'], name="X-axis Capacity", line=dict(color='navy', width=2)))
        fig_pm.add_trace(go.Scatter(x=df_y['phiMn'], y=df_y['phiPn'], name="Y-axis Capacity", line=dict(color='forestgreen', width=2, dash='dash')))
        fig_pm.add_trace(go.Scatter(x=[Mcx, Mcy], y=[Pu, Pu], mode='markers', name="Demands", marker=dict(color=['blue', 'green'], size=10)))
        fig_pm.update_layout(xaxis_title="Moment, M (ton-m)", yaxis_title="Axial Load, P (ton)", plot_bgcolor='white', height=450)
        st.plotly_chart(fig_pm, use_container_width=True)

    with tab3:
        fig_sec = go.Figure()
        if shape == "Rectangular":
            fig_sec.add_trace(go.Scatter(x=[-b/2, b/2, b/2, -b/2, -b/2], y=[-h/2, -h/2, h/2, h/2, -h/2], mode='lines', line=dict(color='black')))
        else:
            theta = np.linspace(0, 2*np.pi, 100)
            fig_sec.add_trace(go.Scatter(x=(b/2)*np.cos(theta), y=(b/2)*np.sin(theta), mode='lines', line=dict(color='black')))
        
        fig_sec.add_trace(go.Scatter(x=[bar['x'] for bar in engine.bars], y=[bar['y'] for bar in engine.bars], mode='markers', marker=dict(color='red', size=8)))
        fig_sec.update_layout(xaxis_title="Width / X (cm)", yaxis_title="Depth / Y (cm)", yaxis=dict(scaleanchor="x", scaleratio=1), plot_bgcolor='whitesmoke', height=450, showlegend=False)
        st.plotly_chart(fig_sec, use_container_width=True)

    with tab4:
        st.markdown("### 📖 คู่มือคำอธิบายพารามิเตอร์ (Parameter Guide)")
        st.markdown("---")
        
        st.markdown("#### 1. แรงกระทำ (Applied Loads)")
        st.markdown("* **Pu (Factored Axial Load):** แรงอัดประลัยตามแนวแกนที่กระทำกับเสา ซึ่งผ่านการคูณตัวคูณน้ำหนักบรรทุก (Load Factor) มาแล้ว เช่น 1.2DL + 1.6LL (หน่วย: ตัน)")
        st.markdown("* **Mux, Muy (Factored Moment):** โมเมนต์ดัดประลัยที่กระทำรอบแกน X และแกน Y ของเสา (หน่วย: ตัน-เมตร)")
        
        st.markdown("#### 2. ประเภทของโครงสร้าง (Frame Type)")
        st.markdown("* **Non-Sway Frame (โครงสร้างแบบไม่เซ หรือ Braced Frame):** โครงสร้างที่มีระบบต้านทานแรงด้านข้างที่แข็งแรงมาก เช่น มีกำแพงรับแรงเฉือน (Shear Wall) หรือปล่องลิฟต์คอนกรีตคอยรับแรงลม/แผ่นดินไหว ทำให้จุดต่อของเสาแทบไม่มีการเคลื่อนตัวทางด้านข้างเลยเมื่อรับแรง")
        st.markdown("* **Sway Frame (โครงสร้างแบบเซ หรือ Unbraced Frame):** โครงสร้างที่ไม่มีการค้ำยัน ต้องอาศัยความสเถียรของเสาและคานในการต้านทานแรงด้านข้าง ทำให้จุดต่อมีการเคลื่อนตัวทางด้านข้าง และก่อให้เกิดโมเมนต์ส่วนเพิ่มจากผลของ P-Delta Effect")
        
        st.markdown("#### 3. พารามิเตอร์ด้านความชะลูด (Slenderness Parameters)")
        st.markdown("* **Lu (Unsupported Length):** ความยาวเสาที่ไม่มีการรองรับ หรือความยาวสุทธิ (Clear Height) วัดจากหลังพื้น/คานชั้นล่าง ถึง ท้องพื้น/คานชั้นบน (หน่วย: เมตร)")
        st.markdown("* **K (Effective Length Factor):** ตัวคูณความยาวประสิทธิผล ขึ้นอยู่กับสภาพความยึดรั้งที่ปลายเสาทั้งบนและล่าง")
        st.markdown("  * **Non-Sway Frame:** ค่า K จะมีค่าอยู่ระหว่าง **0.5 ถึง 1.0**")
        st.markdown("  * **Sway Frame:** ค่า K จะมีค่าตั้งแต่ **1.0 ขึ้นไป**")
        st.markdown("* **Cm (Equivalent Moment Factor):** ตัวคูณปรับแก้โมเมนต์เทียบเท่า **(ใช้เฉพาะใน Non-Sway Frame)** เพื่อปรับสภาพของโมเมนต์ดัดที่ปลายบนและล่างของเสาที่ไม่เท่ากัน ให้กลายเป็นโมเมนต์ดัดแบบสม่ำเสมอเทียบเท่า โดยคำนวณจากสูตรมาตรฐาน ACI ดังนี้:")
        st.latex(r"C_m = 0.6 + 0.4 \left( \frac{M_1}{M_2} \right) \ge 0.4")
        st.markdown("> *โดย **M1** คือโมเมนต์ค่าน้อย และ **M2** คือโมเมนต์ค่ามาก (หากเสาดัดโค้งทางเดียวหรือ Single Curvature อัตราส่วน M1/M2 จะมีค่าเป็นบวก)*")
