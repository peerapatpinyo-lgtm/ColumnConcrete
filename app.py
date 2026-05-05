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
            self.Igx = (self.b * self.h**3) / 12  
            self.Igy = (self.h * self.b**3) / 12  
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

        self.Ise_x = sum(self.as_single * (bar['y']**2) for bar in self.bars)
        self.Ise_y = sum(self.as_single * (bar['x']**2) for bar in self.bars)

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
                get_y = lambda bar: bar['x']

        y_bars = np.array([get_y(bar) for bar in self.bars])
        d_bars = depth / 2 - y_bars
        dt = np.max(d_bars)

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
                    ratio = np.clip((R - a) / R, -1.0, 1.0)
                    theta = 2 * np.arccos(ratio)
                    Ac = (R**2 / 2) * (theta - np.sin(theta))
                    y_bar = (4 * R * np.sin(theta/2)**3) / (3 * (theta - np.sin(theta))) if Ac > 0 else R
                Cc = 0.85 * self.fc * Ac
                Mc = Cc * y_bar 
            
            eps_s = 0.003 * (c - d_bars) / c
            fs = np.clip(eps_s * self.Es, -self.fy, self.fy)
            Fsi = self.as_single * fs
            
            Pn_s = np.sum(Fsi)
            Mn_s = np.sum(Fsi * y_bars)

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
        results.append({'c': 9999, 'Pn': po, 'Mn': 0, 'phiPn': phi_comp * po, 'phiMn': 0})

        df = pd.DataFrame(results).sort_values('Pn', ascending=True)
        df['phiPn'] = df['phiPn'].clip(upper=phi_pn_max)
        df = df.drop_duplicates(subset=['phiPn'], keep='first')
        
        return df, phi_pn_max
        
    def slenderness_magnifier(self, Pu, K, Lu_m, axis, Cm, beta_d):
        Lu_cm = Lu_m * 100
        r = self.rx if axis == 'X' else self.ry
        Ig = self.Igx if axis == 'X' else self.Igy
        Ise = self.Ise_x if axis == 'X' else self.Ise_y
        
        kl_r = (K * Lu_cm) / r
        EI = (0.2 * self.Ec * Ig + self.Es * Ise) / (1 + beta_d)
        Pc = (np.pi**2 * EI) / (K * Lu_cm)**2 / 1000
        
        if Pu >= (0.75 * Pc):
            delta = 999.9 
        else:
            delta = max(1.0, Cm / (1 - (Pu / (0.75 * Pc))))
            
        return kl_r, Pc, delta, Ise, EI

    def check_clear_spacing(self, nx, ny):
        min_req = max(2.5, 1.5 * self.db_cm)
        
        if self.shape == "Rectangular":
            s_x = 999.0
            s_y = 999.0
            if self.layout == "2-Faces (Top/Bottom)":
                n_x_face = len(self.bars) // 2
                if n_x_face > 1:
                    s_x = (self.b - 2 * self.d_prime) / (n_x_face - 1) - self.db_cm
            elif self.layout == "4-Faces (Uniform)":
                if nx > 1:
                    s_x = (self.b - 2 * self.d_prime) / (nx - 1) - self.db_cm
                if ny > 1:
                    s_y = (self.h - 2 * self.d_prime) / (ny - 1) - self.db_cm
            actual_spacing = min(s_x, s_y)
            
        elif self.shape == "Circular":
            Rs = self.D / 2 - self.d_prime
            chord_length = 2 * Rs * np.sin(np.pi / len(self.bars))
            actual_spacing = chord_length - self.db_cm
            
        is_ok = actual_spacing >= min_req
        return actual_spacing, min_req, is_ok

    def get_dynamic_alpha(self, Pu):
        if self.shape == "Circular":
            return 2.0
        else:
            po = (0.85 * self.fc * (self.Ag - self.total_as) + self.fy * self.total_as) / 1000
            phi_comp = 0.65 
            phi_po = phi_comp * po
            
            if phi_po <= 0: return 1.15
            ratio = Pu / phi_po
            
            if ratio < 0.1: return 1.15
            else:
                alpha = 1.15 + (ratio - 0.1) * (1.55 - 1.15) / (1.0 - 0.1)
                return min(1.55, max(1.15, alpha))

    def generate_3d_surface(self, df_x, df_y, alpha):
        p_min = df_x['phiPn'].min() + 0.001
        p_max = df_x['phiPn'].max() - 0.001
        p_steps = np.linspace(p_min, p_max, 30) 
        
        fx = interp1d(df_x['phiPn'], df_x['phiMn'], kind='linear', bounds_error=True)
        fy = interp1d(df_y['phiPn'], df_y['phiMn'], kind='linear', bounds_error=True)
        theta = np.linspace(0, np.pi/2, 20) 
        
        X, Y, Z = [], [], []
        for p in p_steps:
            mx_cap = fx(p)
            my_cap = fy(p)
            x_row, y_row, z_row = [], [], []
            for t in theta:
                if mx_cap > 0 and my_cap > 0:
                    denom = ((np.cos(t) / mx_cap)**alpha + (np.sin(t) / my_cap)**alpha)**(1/alpha)
                    r = 1 / denom
                else: r = 0
                x_row.append(r * np.cos(t))
                y_row.append(r * np.sin(t))
                z_row.append(p)
            X.append(x_row)
            Y.append(y_row)
            Z.append(z_row)
            
        return np.array(X), np.array(Y), np.array(Z)


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
            st.write("Sway Magnification Factor (δs) Parameters")
            sway_method = st.radio("Sway Calculation Method", ["Stability Index (Q)", "Sum of Loads (ΣPu, ΣPc)", "Direct Input"], horizontal=True)
            if sway_method == "Stability Index (Q)":
                Q_val = st.number_input("Stability Index (Q)", min_value=0.0, max_value=0.99, value=0.05, step=0.01)
                delta_s_auto = 1 / (1 - Q_val)
                delta_sx = delta_sy = max(1.0, delta_s_auto)
                st.info(f"Calculated δs = {delta_sx:.3f}")
            elif sway_method == "Sum of Loads (ΣPu, ΣPc)":
                cs1, cs2 = st.columns(2)
                sum_Pu = cs1.number_input("ΣPu (ton)", min_value=0.1, value=500.0)
                sum_Pc = cs2.number_input("ΣPc (ton)", min_value=0.1, value=2000.0)
                if sum_Pu < 0.75 * sum_Pc:
                    delta_s_auto = 1 / (1 - (sum_Pu / (0.75 * sum_Pc)))
                else:
                    delta_s_auto = 999.0
                    st.error("⚠️ ΣPu exceeds 0.75ΣPc, frame is unstable.")
                delta_sx = delta_sy = max(1.0, delta_s_auto)
                st.info(f"Calculated δs = {delta_sx:.3f}")
            else: 
                cs1, cs2 = st.columns(2)
                delta_sx = cs1.number_input("δs (X-axis)", value=1.2, step=0.05)
                delta_sy = cs2.number_input("δs (Y-axis)", value=1.2, step=0.05)
            
            Lu_x = K_x = Cm_x = Lu_y = K_y = Cm_y = beta_d = 0 

# --- Create Engine and Solve ---
engine = RCColumnProBiaxial(shape, layout, b, h, fc, fy, db, n_bars, nx, ny, cover)
df_x, phi_pn_max = engine.solve_pm(axis='X')
df_y, _ = engine.solve_pm(axis='Y')

e_min_x = Pu * (0.015 + 0.03 * (h / 100))
e_min_y = Pu * (0.015 + 0.03 * (b / 100))
Mu_x_dsgn = max(Mux, e_min_x)
Mu_y_dsgn = max(Muy, e_min_y)

if frame_type == "Non-Sway (Braced)":
    kl_rx, Pcx, del_x, Ise_x, EIx = engine.slenderness_magnifier(Pu, K_x, Lu_x, 'X', Cm_x, beta_d)
    kl_ry, Pcy, del_y, Ise_y, EIy = engine.slenderness_magnifier(Pu, K_y, Lu_y, 'Y', Cm_y, beta_d)
    
    Mcx = del_x * Mu_x_dsgn if kl_rx > 22 else Mu_x_dsgn
    Mcy = del_y * Mu_y_dsgn if kl_ry > 22 else Mu_y_dsgn
else:
    M_sway_x = delta_sx * Mu_x_dsgn
    M_sway_y = delta_sy * Mu_y_dsgn
    
    kl_rx, Pcx, del_x_ns, Ise_x, EIx = engine.slenderness_magnifier(Pu, K_x, Lu_x, 'X', Cm_x, beta_d)
    kl_ry, Pcy, del_y_ns, Ise_y, EIy = engine.slenderness_magnifier(Pu, K_y, Lu_y, 'Y', Cm_y, beta_d)
    
    Mcx = del_x_ns * M_sway_x if kl_rx > 22 else M_sway_x
    Mcy = del_y_ns * M_sway_y if kl_ry > 22 else M_sway_y

# --- Biaxial Interaction Check ---
error_status = None
if Pu > phi_pn_max:
    error_status = f"Axial load exceeds section capacity (Pu = {Pu:.1f} t > φPn,max = {phi_pn_max:.1f} t)"
    is_safe = False
    demand_ratio = 999.0
    phi_Mnox = 0
    phi_Mnoy = 0
    alpha = 1.5
else:
    try:
        fx = interp1d(df_x['phiPn'], df_x['phiMn'], kind='linear', fill_value=0, bounds_error=False)
        fy_interp = interp1d(df_y['phiPn'], df_y['phiMn'], kind='linear', fill_value=0, bounds_error=False)
        
        phi_Mnox = float(fx(Pu))
        phi_Mnoy = float(fy_interp(Pu))
        
        if phi_Mnox <= 0 or phi_Mnoy <= 0:
            error_status = "Axial load is out of bound for moment interaction."
            is_safe = False
            demand_ratio = 999.0
            alpha = 2.0 if shape == "Circular" else 1.5
        else:
            # 🎯 FIX 2: Dynamic Alpha Call
            alpha = engine.get_dynamic_alpha(Pu)
            demand_ratio = (Mcx / phi_Mnox)**alpha + (Mcy / phi_Mnoy)**alpha
            is_safe = (demand_ratio <= 1.0)
    except Exception as e:
        error_status = f"Calculation Error: {str(e)}"
        is_safe = False
        demand_ratio = 999.0
        phi_Mnox = 0
        phi_Mnoy = 0
        alpha = 1.5

actual_space, min_req_space, space_ok = engine.check_clear_spacing(nx, ny)

with col2:
    st.markdown("### 📋 Executive Biaxial Summary")
    
    m1, m2, m3, m4 = st.columns(4) 
    rho_pct = engine.rho * 100
    m1.metric("Steel Ratio (ρ)", f"{rho_pct:.2f} %", "OK" if 1 <= rho_pct <= 8 else "Fail", delta_color="normal" if 1 <= rho_pct <= 8 else "inverse")
    m2.metric("Clear Spacing", f"{actual_space:.2f} cm", "OK" if space_ok else "Tight!", delta_color="normal" if space_ok else "inverse")
    m3.metric("Design Mcx", f"{Mcx:.2f} t-m", f"Magnifier: {max(Mcx/Mu_x_dsgn, 1.0):.2f}x", delta_color="off")
    m4.metric("Design Mcy", f"{Mcy:.2f} t-m", f"Magnifier: {max(Mcy/Mu_y_dsgn, 1.0):.2f}x", delta_color="off")

    st.markdown("---")
    
    if not space_ok:
        st.warning(f"⚠️ **Constructability Warning:** ระยะห่างเหล็กเสริมจริง ({actual_space:.2f} cm) น้อยกว่าค่ามาตรฐานที่กำหนด ({min_req_space:.2f} cm) อาจทำให้เทคอนกรีตได้ยากและเกิดรอยโพรง (Honeycomb)")
    
    if error_status:
        st.error(f"### ❌ **STATUS: CAPACITY EXCEEDED**\n{error_status}")
    elif is_safe:
        st.success(f"### ✅ **STATUS: SAFE**\nBiaxial Demand Ratio = **{demand_ratio:.3f}** ≤ 1.0")
    else:
        st.error(f"### ❌ **STATUS: UNSAFE**\nBiaxial Demand Ratio = **{demand_ratio:.3f}** > 1.0")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌐 3D/Biaxial Interaction", "📊 P-M Curves", "📐 Section", "📖 Parameter Guide", "📝 Calculation Report"])

    with tab1:
        if not error_status:
            try:
                mx_m, my_m, p_m = engine.generate_3d_surface(df_x, df_y, alpha)
                fig_3d = go.Figure()
                fig_3d.add_trace(go.Surface(x=mx_m, y=my_m, z=p_m, colorscale='Viridis', opacity=0.7, name='Capacity Surface', showscale=False))
                fig_3d.add_trace(go.Scatter3d(x=[Mcx], y=[Mcy], z=[Pu], mode='markers', marker=dict(size=8, color='red', symbol='diamond'), name='Applied Demand'))
                fig_3d.update_layout(scene=dict(xaxis_title='Mx (t-m)', yaxis_title='My (t-m)', zaxis_title='Axial P (ton)', aspectmode='manual', aspectratio=dict(x=1, y=1, z=1.2)), margin=dict(l=0, r=0, b=0, t=0), height=550)
                st.plotly_chart(fig_3d, use_container_width=True)
            except Exception as e:
                st.info("ℹ️ กำลังรอคำนวณข้อมูล 3D Surface...")
        else:
            st.info("⚠️ ไม่สามารถจำลอง 3D Surface ได้ เนื่องจากแรงแนวแกน (Pu) เกินขีดจำกัดหน้าตัด")

    with tab2:
        # --- คำนวณเส้นอ้างอิง 1% และ 8% เพื่อใช้เป็นขอบเขต (Optimal Range) ---
        # สร้าง Engine ชั่วคราวเพื่อหา P-M Curve ที่ 1% และ 8%
        # โดยใช้เหล็กเสริมแบบกระจายรอบด้าน (4-Faces) เป็นมาตรฐานในการเปรียบเทียบ
        def get_ref_curve(target_rho):
            target_as = target_rho * engine.Ag
            # ประมาณการจำนวนเหล็กเพื่อให้ได้ Rho ที่ต้องการ (ใช้เหล็ก DB20 เป็นตัวอ้างอิง)
            ref_n_bars = max(4, int(target_as / 3.14)) 
            if shape == "Rectangular":
                # กระจายเหล็กแบบ Uniform ง่ายๆ เพื่อทำเส้น Guide
                ref_nx = max(2, int(np.sqrt(ref_n_bars * (b/h))))
                ref_ny = max(2, int(ref_n_bars / 2) - ref_nx + 2)
                ref_engine = RCColumnProBiaxial(shape, "4-Faces (Uniform)", b, h, fc, fy, 20, 0, ref_nx, ref_ny, cover)
            else:
                ref_engine = RCColumnProBiaxial(shape, "Circular", b, h, fc, fy, 20, ref_n_bars, 0, 0, cover)
            
            # คำนวณ P-M เฉพาะแกน X เพื่อใช้เป็นตัวแทนขอบเขต
            ref_df, _ = ref_engine.solve_pm(axis='X')
            return ref_df

        with st.spinner("Calculating optimal range boundaries..."):
            df_1pct = get_ref_curve(0.01)
            df_8pct = get_ref_curve(0.08)

        # --- สร้างกราฟ Plotly ---
        fig_pm = go.Figure()
        
        # 1. เส้นขอบเขต 1% (Lower Bound - Economy)
        fig_pm.add_trace(go.Scatter(
            x=df_1pct['phiMn'], y=df_1pct['phiPn'],
            name="Minimum Limit (ρ=1%)",
            line=dict(color='rgba(0,0,0,0.3)', width=2, dash='dash'),
            fill=None
        ))
        
        # 2. เส้นขอบเขต 8% (Upper Bound - Practical Limit)
        fig_pm.add_trace(go.Scatter(
            x=df_8pct['phiMn'], y=df_8pct['phiPn'],
            name="Maximum Limit (ρ=8%)",
            line=dict(color='rgba(255,0,0,0.2)', width=2, dash='dash'),
            fill='tonexty', # ระบายสีพื้นที่ระหว่าง 1% และ 8%
            fillcolor='rgba(0, 255, 0, 0.05)' # สีเขียวจางๆ แสดงโซนที่เหมาะสม
        ))

        # 3. เส้น Capacity จริงของหน้าตัดที่ออกแบบ
        fig_pm.add_trace(go.Scatter(x=df_x['phiMn'], y=df_x['phiPn'], name="X-axis Capacity (Actual)", line=dict(color='navy', width=3)))
        fig_pm.add_trace(go.Scatter(x=df_y['phiMn'], y=df_y['phiPn'], name="Y-axis Capacity (Actual)", line=dict(color='forestgreen', width=3, dash='dot')))
        
        # 4. จุดสถานะ Load ปัจจุบัน
        fig_pm.add_trace(go.Scatter(x=[Mcx, Mcy], y=[Pu, Pu], mode='markers', name="Design Demands", 
                                   marker=dict(color=['blue', 'green'], size=12, symbol='x')))

        fig_pm.update_layout(
            title="P-M Interaction Diagram with ρ Boundaries",
            xaxis_title="Design Moment, φMn (ton-m)",
            yaxis_title="Design Axial Strength, φPn (ton)",
            plot_bgcolor='white',
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pm, use_container_width=True)
        st.caption("💡 พื้นที่สีเขียวจางๆ แสดงช่วงการออกแบบที่เหมาะสม (Optimal Range: ρ = 1% - 8%)")

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
        st.markdown("### 📖 Parameter Guide")
        st.markdown("---")
        st.markdown("#### 1. Applied Loads")
        st.markdown("* **Pu (Factored Axial Load):** The ultimate axial load acting on the column. *(Unit: tons)*")
        st.markdown("* **Mux, Muy (Factored Moments):** The ultimate bending moments acting about the X and Y axes. *(Unit: ton-m)*")
        st.markdown("#### 2. Frame Type")
        st.markdown("* **Non-Sway Frame (Braced Frame):** A structure equipped with a stiff lateral force-resisting system. Joints experience practically no lateral translation.")
        st.markdown("* **Sway Frame (Unbraced Frame):** Relies entirely on the stiffness of its beams and columns. Joints can translate laterally, generating P-Delta effect.")

    with tab5:
        st.markdown("### 📝 Detailed Calculation Report")
        st.info("💡 รายงานนี้แสดงการคำนวณแบบ Step-by-Step พร้อมระบุตัวแปรที่ใช้ใน Source Code")
        st.markdown("---")

        with st.expander("1. Section & Material Properties", expanded=False):
            st.markdown("#### 1.1 Geometry & Section Properties")
            if shape == "Rectangular":
                st.latex(f"A_g = {b} \\times {h} = {engine.Ag:,.2f} \\text{{ cm}}^2")
                st.latex(f"I_{{gx}} = \\frac{{{b} \\times {h}^3}}{{12}} = {engine.Igx:,.2f} \\text{{ cm}}^4")
                st.latex(f"I_{{gy}} = \\frac{{{h} \\times {b}^3}}{{12}} = {engine.Igy:,.2f} \\text{{ cm}}^4")
            else:
                st.latex(f"A_g = \\frac{{\\pi \\times {b}^2}}{{4}} = {engine.Ag:,.2f} \\text{{ cm}}^2")
                st.latex(f"I_{{gx}} = I_{{gy}} = \\frac{{\\pi \\times {b}^4}}{{64}} = {engine.Igx:,.2f} \\text{{ cm}}^4")
            st.markdown("#### 1.2 Material Properties")
            st.latex(f"E_c = 15100 \\sqrt{{{fc}}} = {engine.Ec:,.0f} \\text{{ ksc}}")

        with st.expander("2. Minimum Design Moments (ACI 318-19)", expanded=False):
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("**X-Axis (Major):**")
                st.latex(f"M_{{u,min,x}} = {Pu} \\times (0.015 + 0.03 \\times \\frac{{{h}}}{{100}}) = {e_min_x:,.3f} \\text{{ t-m}}")
                st.latex(f"M_{{ux,dsgn}} = \\max({Mux:,.2f}, {e_min_x:,.3f}) = {Mu_x_dsgn:,.2f} \\text{{ t-m}}")
            with col_m2:
                st.markdown("**Y-Axis (Minor):**")
                st.latex(f"M_{{u,min,y}} = {Pu} \\times (0.015 + 0.03 \\times \\frac{{{b}}}{{100}}) = {e_min_y:,.3f} \\text{{ t-m}}")
                st.latex(f"M_{{uy,dsgn}} = \\max({Muy:,.2f}, {e_min_y:,.3f}) = {Mu_y_dsgn:,.2f} \\text{{ t-m}}")

        # --- Part 3: Moment Magnification (X-Axis) ---
        with st.expander(f"3. Moment Magnification (X-Axis) - {frame_type}", expanded=False):
            if frame_type == "Non-Sway (Braced)":
                st.markdown("**3.1 Effective Stiffness ($EI_x$):** *(Ref: ACI 318-19, 6.6.4.4.4)*")
                st.latex(r"EI_x = \frac{0.2 E_c I_{gx} + E_s I_{se,x}}{1 + \beta_d}")
                
                # 🎯 REFACTOR: Retrieve Ise_x directly from the engine attribute
                Ise_x = engine.Ise_x
                EIx_val = (0.2 * engine.Ec * engine.Igx + engine.Es * Ise_x) / (1 + beta_d)
                
                st.latex(f"I_{{se,x}} = {Ise_x:,.2f} \\text{{ cm}}^4")
                st.latex(f"EI_x = \\frac{{(0.2 \\times {engine.Ec:,.0f} \\times {engine.Igx:,.0f}) + ({engine.Es:,.0f} \\times {Ise_x:,.2f})}}{{1 + {beta_d}}} = {EIx_val:,.0f} \\text{{ kg-cm}}^2")
                
                st.markdown("**3.2 Euler Critical Load ($P_{cx}$):**")
                st.latex(r"P_{cx} = \frac{\pi^2 EI_x}{(K_x L_{ux})^2}")
                st.latex(f"P_{{cx}} = \\frac{{\\pi^2 \\times {EIx_val:,.0f}}}{{({K_x} \\times {Lu_x} \\times 100)^2}} \\times 10^{{-3}} = {Pcx:,.2f} \\text{{ ton}}")
                
                st.markdown("**3.3 Magnification Factor ($\delta_x$):**")
                if kl_rx > 22:
                    st.latex(r"\delta_x = \frac{C_{mx}}{1 - \frac{P_u}{0.75 P_{cx}}} \ge 1.0")
                    st.latex(f"\\delta_x = \\frac{{{Cm_x}}}{{1 - \\frac{{{Pu}}}{{0.75 \\times {Pcx:,.2f}}}}} = {del_x:,.3f}")
                else:
                    st.write(f"Slenderness ignored (kl/r = {kl_rx:.2f} $\\le$ 22)")
                    st.latex(r"\delta_x = 1.0")

                st.markdown("**3.4 Final Magnified Moment ($M_{cx}$):**")
                st.latex(f"M_{{cx}} = \\delta_x \\times M_{{ux,dsgn}} = {Mcx:,.2f} \\text{{ ton-m}}")
            else:
                st.markdown("**Sway Frame Design:**")
                st.latex(r"M_{cx} = \delta_{sx} M_{ux,dsgn}")
                st.latex(f"M_{{cx}} = {delta_sx:.2f} \\times {Mu_x_dsgn:,.2f} = {Mcx:,.2f} \\text{{ ton-m}}")
            st.caption("💻 *Code Vars: `Ise_x`, `Pcx`, `del_x`, `Mcx`*")

        # --- Part 4: Moment Magnification (Y-Axis) ---
        with st.expander(f"4. Moment Magnification (Y-Axis) - {frame_type}", expanded=False):
            if frame_type == "Non-Sway (Braced)":
                st.markdown("**4.1 Effective Stiffness ($EI_y$):**")
                st.latex(r"EI_y = \frac{0.2 E_c I_{gy} + E_s I_{se,y}}{1 + \beta_d}")
                
                # 🎯 FIX: Retrieve Ise_y directly from the engine to optimize performance
                Ise_y = engine.Ise_y 
                EIy_val = (0.2 * engine.Ec * engine.Igy + engine.Es * Ise_y) / (1 + beta_d)
                
                st.latex(f"I_{{se,y}} = {Ise_y:,.2f} \\text{{ cm}}^4")
                st.latex(f"EI_y = \\frac{{(0.2 \\times {engine.Ec:,.0f} \\times {engine.Igy:,.0f}) + ({engine.Es:,.0f} \\times {Ise_y:,.2f})}}{{1 + {beta_d}}} = {EIy_val:,.0f} \\text{{ kg-cm}}^2")
                
                st.markdown("**4.2 Euler Critical Load ($P_{cy}$):**")
                st.latex(f"P_{{cy}} = \\frac{{\\pi^2 \\times {EIy_val:,.0f}}}{{({K_y} \\times {Lu_y} \\times 100)^2}} \\times 10^{{-3}} = {Pcy:,.2f} \\text{{ ton}}")
                
                st.markdown("**4.3 Magnification Factor ($\delta_y$):**")
                if kl_ry > 22:
                    st.latex(r"\delta_y = \frac{C_{my}}{1 - \frac{P_u}{0.75 P_{cy}}} \ge 1.0")
                    st.latex(f"\\delta_y = \\frac{{{Cm_y}}}{{1 - \\frac{{{Pu}}}{{0.75 \\times {Pcy:,.2f}}}}} = {del_y:,.3f}")
                else:
                    st.write(f"Slenderness ignored (kl/r = {kl_ry:.2f} $\\le$ 22)")
                    st.latex(r"\delta_y = 1.0")

                st.markdown("**4.4 Final Magnified Moment ($M_{cy}$):**")
                st.latex(f"M_{{cy}} = \\delta_y \\times M_{{uy,dsgn}} = {Mcy:,.2f} \\text{{ ton-m}}")
            else:
                st.markdown("**Sway Frame Design:**")
                st.latex(r"M_{cy} = \delta_{sy} M_{uy,dsgn}")
                st.latex(f"M_{{cy}} = {delta_sy:.2f} \\times {Mu_y_dsgn:,.2f} = {Mcy:,.2f} \\text{{ ton-m}}")
            st.caption("💻 *Code Vars: `Ise_y`, `Pcy`, `del_y`, `Mcy`*")

        # --- Part 5: Biaxial Bending Check ---
        with st.expander("5. Biaxial Bending Interaction (PCA Method)", expanded=True):
            st.markdown("**PCA Load Contour Method** *(Ref: PCA Notes on ACI 318)*")
            st.markdown(f"At factored axial load $P_u = {Pu:,.2f}$ tons, the program evaluates the intersection on the P-M Curve to determine the Uniaxial Moment Capacities:")
            st.latex(f"\\phi M_{{nox}} = {phi_Mnox:,.2f} \\text{{ ton-m}}")
            st.latex(f"\\phi M_{{noy}} = {phi_Mnoy:,.2f} \\text{{ ton-m}}")
            
            st.markdown("**Interaction Equation:**")
            st.latex(r"\left( \frac{M_{cx}}{\phi M_{nox}} \right)^\alpha + \left( \frac{M_{cy}}{\phi M_{noy}} \right)^\alpha \le 1.0")
            
            if phi_Mnox > 0 and phi_Mnoy > 0:
                st.markdown("**Substituting the values:**")
                # 🎯 FIX: Alpha value is already retrieved dynamically using engine.get_dynamic_alpha(Pu)
                st.latex(f"\\text{{Ratio}} = \\left( \\frac{{{Mcx:,.2f}}}{{{phi_Mnox:,.2f}}} \\right)^{{{alpha:.3f}}} + \\left( \\frac{{{Mcy:,.2f}}}{{{phi_Mnoy:,.2f}}} \\right)^{{{alpha:.3f}}} = {demand_ratio:,.3f}")
                st.caption(f"💻 *Code Vars: `alpha` (={alpha:.3f}), `demand_ratio`, `phi_Mnox`, `phi_Mnoy`*")
            else:
                st.error("⚠️ Unable to calculate because the applied axial load ($P_u$) exceeds the maximum compressive strength of the section.")

        # --- Final Part: Conclusion ---
        st.markdown("---")
        if is_safe:
            st.success(f"✅ **Check Summary:** The Demand Ratio = **{demand_ratio:,.3f}** which is $\\le$ 1.0 $\\rightarrow$ **SECTION IS SAFE**")
        else:
            st.error(f"❌ **Check Summary:** The Demand Ratio = **{demand_ratio:,.3f}** which is > 1.0 $\\rightarrow$ **SECTION IS UNSAFE**")
