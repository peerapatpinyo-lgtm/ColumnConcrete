import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
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

    # --- ส่วนที่แก้ไข: Expander สำหรับ Shear, Torsion และ Seismic ---
    with st.expander("3. Shear, Torsion & Seismic", expanded=True):
        st.subheader("🛡️ Shear Design (Stirrups)")
        
        # เพิ่มคอลัมน์รับค่าแรงเฉือนแยกแกน X และ Y
        c_vux, c_vuy = st.columns(2)
        vux_ton = c_vux.number_input("Factored Shear X, Vux (ton)", value=5.0, step=1.0)
        vuy_ton = c_vuy.number_input("Factored Shear Y, Vuy (ton)", value=5.0, step=1.0)
        
        c7, c8 = st.columns(2)
        tie_dia = c7.selectbox("Tie Diameter", [6, 9, 12, 16], index=1, format_func=lambda x: f"RB{x}" if x<10 else f"DB{x}")
        tie_legs = c8.number_input("Stirrup Legs (Max)", value=2, min_value=2, step=1)
        
        st.markdown("---")
        st.subheader("🌪️ Torsion & Seismic")
        tu_tonm = st.number_input("Factored Torsion, Tu (ton-m)", value=0.0, step=0.5)
        is_seismic = st.toggle("Seismic Detailing (Special Moment Frame)", value=True)

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

# --- 🛡️ NEW: Shear, Torsion & Seismic Calculation ---
phi_v = 0.75
Pu_kg = Pu * 1000

# 🔴 อัปเดต: ดึงค่าจากตัวแปร vux_ton และ vuy_ton ที่เราสร้างใหม่
Vux_kg = vux_ton * 1000
Vuy_kg = vuy_ton * 1000
Vu_kg = max(Vux_kg, Vuy_kg)  # ใช้ค่าแรงเฉือนที่มากที่สุดเพื่อการคำนวณเช็คค่าเบื้องต้น

# Effective depth (d) & Web width (bw)
if shape == "Rectangular":
    d_eff = min(h - engine.d_prime, b - engine.d_prime)
    bw = min(b, h)
    Acp = b * h
    pcp = 2 * (b + h)
else:
    d_eff = 0.8 * engine.D
    bw = engine.D
    Acp = engine.Ag
    pcp = np.pi * engine.D

# 1. Concrete Shear Capacity (Vc) - Considering Axial Compression Benefit
Vc_kg = 0.53 * np.sqrt(fc) * bw * d_eff * (1 + Pu_kg / (140 * engine.Ag))
phi_Vc_ton = (phi_v * Vc_kg) / 1000

# 2. Required Stirrup Spacing (Shear)
Av = tie_legs * (np.pi * (tie_dia / 10)**2 / 4)

# 🔴 อัปเดต: ดึงค่าสูงสุดจาก vux_ton และ vuy_ton แทน vu_ton ตัวเก่า
Vu_ton_max = max(vux_ton, vuy_ton)
Vs_req_ton = max(0.0, (Vu_ton_max - phi_Vc_ton) / phi_v)
Vs_req_kg = Vs_req_ton * 1000

s_req = (Av * fy * d_eff) / Vs_req_kg if Vs_req_kg > 0 else 999.0
s_max_code = min(d_eff / 2, 60.0)

# 3. Torsion Threshold Check (Tu > Tth)
Tu_kgcm = tu_tonm * 100000
Tth_kgcm = 0.26 * phi_v * np.sqrt(fc) * (Acp**2 / pcp)
Tth_tonm = Tth_kgcm / 100000
is_torsion_significant = tu_tonm > Tth_tonm

# 4. Seismic Spacing (Special Moment Frame - SMF)
s_seismic = 999.0
if is_seismic:
    s_seismic = min(bw / 4, 6 * engine.db_cm, 15.0)

# Final Tie Spacing Calculation
s_design = min(s_req, s_max_code, s_seismic)
s_design_final = np.floor(s_design)

shear_status = "Fail" if s_design_final < 5.0 else "OK"


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
    st.markdown("### 📋 Executive Design Summary")
    
    # อัปเดต Column ให้รองรับผล Shear/Seismic
    m1, m2, m3, m4, m5 = st.columns(5) 
    rho_pct = engine.rho * 100
    m1.metric("Steel (ρ)", f"{rho_pct:.2f} %", "OK" if 1 <= rho_pct <= 8 else "Fail", delta_color="normal" if 1 <= rho_pct <= 8 else "inverse")
    m2.metric("Clear Space", f"{actual_space:.2f} cm", "OK" if space_ok else "Tight!", delta_color="normal" if space_ok else "inverse")
    m3.metric("Design Mcx", f"{Mcx:.1f} t-m", f"Mag: {max(Mcx/Mu_x_dsgn, 1.0):.2f}x", delta_color="off")
    m4.metric("Design Mcy", f"{Mcy:.1f} t-m", f"Mag: {max(Mcy/Mu_y_dsgn, 1.0):.2f}x", delta_color="off")
    
    shear_label = "Seismic Tie" if is_seismic else "Shear Tie"
    m5.metric(shear_label, f"@ {s_design_final:.0f} cm", "OK" if shear_status == "OK" else "Too Dense!", delta_color="normal" if shear_status == "OK" else "inverse")

    st.markdown("---")
    
    if is_torsion_significant:
        st.warning(f"🌪️ **Torsion Alert:** Factored Torsion (Tu = {tu_tonm:.2f} t-m) exceeds the threshold (Tth = {Tth_tonm:.2f} t-m). Additional closed stirrups and longitudinal bars are strongly required!")
        
    if not space_ok:
        st.warning(f"⚠️ **Constructability Warning:** ระยะห่างเหล็กเสริมจริง ({actual_space:.2f} cm) น้อยกว่าค่ามาตรฐานที่กำหนด ({min_req_space:.2f} cm) อาจทำให้เทคอนกรีตได้ยากและเกิดรอยโพรง (Honeycomb)")
    
    if error_status:
        st.error(f"### ❌ **STATUS: CAPACITY EXCEEDED**\n{error_status}")
    elif is_safe:
        st.success(f"### ✅ **STATUS: SAFE**\nBiaxial Demand Ratio = **{demand_ratio:.3f}** ≤ 1.0")
    else:
        st.error(f"### ❌ **STATUS: UNSAFE**\nBiaxial Demand Ratio = **{demand_ratio:.3f}** > 1.0")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📥 Input & Overview", 
        "📊 P-M Interaction", 
        "🧊 BIM & CAD Detail", 
        "🌪️ Shear & Seismic",  
        "📖 Parameter Guide", 
        "📝 Calc Report"
    ])

    with tab1:
        st.markdown("### 🌐 3D Biaxial Interaction & PCA Contour")
        st.markdown("Interactive 3D Failure Surface and 2D Cross-Section Slice at the specific factored axial load (Pu).")
        
        # --- ส่วนที่ 1: กราฟ 3D Surface (Premium Lighting & Shadow) ---
        if not error_status:
            try:
                mx_m, my_m, p_m = engine.generate_3d_surface(df_x, df_y, alpha)
                fig_3d = go.Figure()
                
                # 3D Capacity Surface
                fig_3d.add_trace(go.Surface(
                    x=mx_m, y=my_m, z=p_m, 
                    colorscale='Viridis', opacity=0.8, 
                    name='Capacity Surface', showscale=False,
                    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.5, fresnel=0.2)
                ))
                
                # Design Point
                marker_color = '#2ecc71' if is_safe else '#e74c3c'
                fig_3d.add_trace(go.Scatter3d(
                    x=[Mcx], y=[Mcy], z=[Pu], 
                    mode='markers+text', 
                    marker=dict(size=8, color=marker_color, symbol='diamond', line=dict(width=2, color='white')), 
                    name='Factored Demand',
                    text=["Demand (Mux, Muy, Pu)"], textposition="top center"
                ))
                
                # Drop Line to XY Plane (ช่วยให้ดูพิกัดง่ายขึ้น)
                fig_3d.add_trace(go.Scatter3d(
                    x=[Mcx, Mcx], y=[Mcy, Mcy], z=[0, Pu],
                    mode='lines', line=dict(color=marker_color, width=3, dash='dot'), showlegend=False
                ))

                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title='<b>Mx (t-m)</b>', 
                        yaxis_title='<b>My (t-m)</b>', 
                        zaxis_title='<b>Axial P (ton)</b>',
                        xaxis=dict(showbackground=True, backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
                        yaxis=dict(showbackground=True, backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
                        zaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
                        aspectmode='manual', aspectratio=dict(x=1, y=1, z=1.2),
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                    ), 
                    margin=dict(l=0, r=0, b=0, t=0), height=600
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            except Exception as e:
                st.info("ℹ️ Calculating 3D Surface data...")
        else:
            st.error("⚠️ Cannot generate 3D Surface because the applied axial load (Pu) far exceeds the section's ultimate capacity.")

        st.markdown("---")

        # --- ส่วนที่ 2: Biaxial PCA Contour & Dashboard ---
        st.markdown(f"#### 🎯 2D PCA Contour Slice at Pu = {Pu:,.2f} ton")
        
        col1, col2, col3 = st.columns([1, 1, 1.5])
        
        with col1:
            st.metric(label="Demand Ratio", value=f"{demand_ratio:.3f}", delta="SAFE" if is_safe else "UNSAFE", delta_color="inverse")
            st.caption("Limit: <= 1.0")
            
        with col2:
            st.metric(label="Contour Exponent (α)", value=f"{alpha:.3f}")
            st.caption("PCA Parameter")

        with col3:
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = demand_ratio,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Capacity Utilization", 'font': {'size': 14}},
                gauge = {
                    'axis': {'range': [0, 1.5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#2c3e50"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.8], 'color': 'rgba(46, 204, 113, 0.3)'},   
                        {'range': [0.8, 1.0], 'color': 'rgba(241, 196, 15, 0.3)'}, 
                        {'range': [1.0, 1.5], 'color': 'rgba(231, 76, 60, 0.3)'}   
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1.0}
                }
            ))
            fig_gauge.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # --- ส่วนที่ 3: SMART FAILURE DIAGNOSIS & RECOMMENDATIONS ---
        if not is_safe:
            st.markdown("---")
            st.error("### ❌ Design Failed: Section Capacity Exceeded")
            
            recommendations = []
            max_Pn = df_x['phiPn'].max()

            if Pu > max_Pn:
                st.warning(f"**Diagnosis:** Pure Axial Failure. The applied axial load (Pu = {Pu:.2f} ton) exceeds the maximum pure compressive strength of the column (φPn,max = {max_Pn:.2f} ton).")
                recommendations.append(f"**Increase Section Size:** Enlarge the column dimensions (current: {b}x{h} cm) to provide more concrete area (Ag).")
                recommendations.append(f"**Increase Concrete Strength:** Upgrade f'c (current: {fc} ksc) to higher strength concrete.")
            else:
                st.warning(f"**Diagnosis:** Biaxial Bending Failure. The interaction of magnified moments and axial load results in a Demand Ratio of {demand_ratio:.3f} (> 1.0).")
                
                if engine.rho < 0.04:
                    recommendations.append(f"**Increase Reinforcement:** The current reinforcement ratio is relatively low (ρ = {engine.rho*100:.2f}%). Try increasing the number of bars or using larger bar sizes (e.g., DB25, DB28).")
                
                if Mcx > Mux * 1.5 or Mcy > Muy * 1.5:
                    recommendations.append("**Check Slenderness Effects:** The moments are heavily magnified due to the column's slenderness (δ > 1.5). Consider increasing the column dimensions to increase stiffness (EI), or providing intermediate bracing.")
                
                recommendations.append(f"**Increase Section Dimensions:** Slightly increasing the depth or width will significantly boost the bending capacity (Ig).")

            st.markdown("#### 💡 Engineering Recommendations:")
            for rec in recommendations:
                st.markdown(f"- {rec}")
            st.markdown("---")

        # --- วาดกราฟ PCA Contour ---
        if phi_Mnox > 0 and phi_Mnoy > 0:
            mx_vals = np.linspace(0, phi_Mnox, 100)
            my_vals = []
            for mx in mx_vals:
                ratio_x = (mx / phi_Mnox) ** alpha
                if ratio_x > 1.0: ratio_x = 1.0
                my = phi_Mnoy * ((1 - ratio_x) ** (1 / alpha))
                my_vals.append(my)

            fig_contour = go.Figure()

            # Capacity Boundary
            fig_contour.add_trace(go.Scatter(x=mx_vals, y=my_vals, mode='lines', name=f"Capacity Boundary (α={alpha:.2f})", line=dict(color='#8e44ad', width=3), fill='tozeroy', fillcolor='rgba(142, 68, 173, 0.1)', hovertemplate="<b>Boundary</b><br>Mcx: %{x:.2f} t-m<br>Mcy: %{y:.2f} t-m<extra></extra>"))

            # Uniaxial Capacities
            fig_contour.add_trace(go.Scatter(x=[phi_Mnox, 0], y=[0, phi_Mnoy], mode='markers+text', name="Uniaxial Capacities", marker=dict(color='#2c3e50', size=8, symbol='square'), text=[f"φMnox = {phi_Mnox:.2f}", f"φMnoy = {phi_Mnoy:.2f}"], textposition=["top right", "top right"]))

            # Design Demand
            marker_color = '#2ecc71' if is_safe else '#e74c3c'
            fig_contour.add_trace(go.Scatter(x=[Mcx], y=[Mcy], mode='markers+text', name="Factored Demand", marker=dict(color=marker_color, size=14, symbol='cross', line=dict(width=2, color='white')), text=["Design Point"], textposition="top right", hovertemplate="<b>Demand</b><br>Mcx: %{x:.2f} t-m<br>Mcy: %{y:.2f} t-m<extra></extra>"))

            # Vector Line
            fig_contour.add_shape(type="line", x0=0, y0=0, x1=Mcx, y1=Mcy, line=dict(color=marker_color, width=2, dash="dashdot"))

            fig_contour.update_layout(xaxis=dict(title="<b>Magnified Moment X-Axis, Mcx (ton-m)</b>", showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.2)', range=[0, max(phi_Mnox, Mcx) * 1.2]), yaxis=dict(title="<b>Magnified Moment Y-Axis, Mcy (ton-m)</b>", showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.2)', range=[0, max(phi_Mnoy, Mcy) * 1.2]), plot_bgcolor='white', paper_bgcolor='white', height=500, legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)', bordercolor='gray', borderwidth=1), margin=dict(l=40, r=40, t=20, b=40))
            st.plotly_chart(fig_contour, use_container_width=True)
            st.markdown("---")
            
        st.markdown("#### 📈 Uniaxial P-M Projections (Side Views)")
        st.markdown("Examine the column's behavior along the principal axes. The shaded region represents the safe design envelope. The red cross indicates your factored demand.")
        
        # สร้าง 2 คอลัมน์สำหรับกราฟ P-Mx และ P-My
        col_pmx, col_pmy = st.columns(2)
        
        # --- กราฟซ้าย: P-Mx (Major Axis) ---
        with col_pmx:
            fig_pmx = go.Figure()
            
            # เส้น Capacity และพื้นที่ปลอดภัย (แกน X)
            fig_pmx.add_trace(go.Scatter(
                x=df_x['phiMn'], y=df_x['phiPn'], 
                mode='lines', line=dict(color='#2980b9', width=2.5), 
                fill='tozeroy', fillcolor='rgba(41, 128, 185, 0.08)', 
                name="X-Axis Capacity",
                hovertemplate="<b>Capacity</b><br>φMn: %{x:.2f} t-m<br>φPn: %{y:.2f} ton<extra></extra>"
            ))
            
            # จุด Demand (Mcx, Pu)
            fig_pmx.add_trace(go.Scatter(
                x=[Mcx], y=[Pu], 
                mode='markers+text', 
                marker=dict(color='#e74c3c', size=12, symbol='cross', line=dict(width=2, color='white')), 
                name="Demand Point", text=["Demand (Mcx, Pu)"], textposition="top right",
                hovertemplate="<b>Demand</b><br>Mcx: %{x:.2f} t-m<br>Pu: %{y:.2f} ton<extra></extra>"
            ))
            
            # เส้นนำสายตา
            fig_pmx.add_shape(type="line", x0=0, y0=Pu, x1=Mcx, y1=Pu, line=dict(color="#e74c3c", width=1, dash="dot"))
            fig_pmx.add_shape(type="line", x0=Mcx, y0=0, x1=Mcx, y1=Pu, line=dict(color="#e74c3c", width=1, dash="dot"))

            fig_pmx.update_layout(
                title=dict(text="<b>P-Mx Interaction (Major Axis)</b>", font=dict(size=14, color="#2c3e50")),
                xaxis=dict(title="<b>Magnified Moment X, Mcx (t-m)</b>", showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.1)', rangemode='tozero'),
                yaxis=dict(title="<b>Axial Load, Pu (ton)</b>", showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.1)', range=[0, df_x['phiPn'].max() * 1.1]),
                plot_bgcolor='white', paper_bgcolor='white', height=400, showlegend=False, margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_pmx, use_container_width=True)
            
        # --- กราฟขวา: P-My (Minor Axis) ---
        with col_pmy:
            fig_pmy = go.Figure()
            
            # เส้น Capacity และพื้นที่ปลอดภัย (แกน Y)
            fig_pmy.add_trace(go.Scatter(
                x=df_y['phiMn'], y=df_y['phiPn'], 
                mode='lines', line=dict(color='#27ae60', width=2.5), 
                fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.08)', 
                name="Y-Axis Capacity",
                hovertemplate="<b>Capacity</b><br>φMn: %{x:.2f} t-m<br>φPn: %{y:.2f} ton<extra></extra>"
            ))
            
            # จุด Demand (Mcy, Pu)
            fig_pmy.add_trace(go.Scatter(
                x=[Mcy], y=[Pu], 
                mode='markers+text', 
                marker=dict(color='#e74c3c', size=12, symbol='cross', line=dict(width=2, color='white')), 
                name="Demand Point", text=["Demand (Mcy, Pu)"], textposition="top right",
                hovertemplate="<b>Demand</b><br>Mcy: %{x:.2f} t-m<br>Pu: %{y:.2f} ton<extra></extra>"
            ))
            
            # เส้นนำสายตา
            fig_pmy.add_shape(type="line", x0=0, y0=Pu, x1=Mcy, y1=Pu, line=dict(color="#e74c3c", width=1, dash="dot"))
            fig_pmy.add_shape(type="line", x0=Mcy, y0=0, x1=Mcy, y1=Pu, line=dict(color="#e74c3c", width=1, dash="dot"))

            fig_pmy.update_layout(
                title=dict(text="<b>P-My Interaction (Minor Axis)</b>", font=dict(size=14, color="#2c3e50")),
                xaxis=dict(title="<b>Magnified Moment Y, Mcy (t-m)</b>", showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.1)', rangemode='tozero'),
                # บังคับ Y-axis ให้เท่ากับกราฟซ้าย เพื่อให้เปรียบเทียบสัดส่วนด้วยตาเปล่าได้ง่าย
                yaxis=dict(title="<b>Axial Load, Pu (ton)</b>", showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.1)', range=[0, df_x['phiPn'].max() * 1.1]),
                plot_bgcolor='white', paper_bgcolor='white', height=400, showlegend=False, margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_pmy, use_container_width=True)

    with tab2:
        st.markdown("### 📊 Advanced P-M Interaction Diagram")
        
        # --- UI Controls สำหรับกราฟ ---
        col_ctrl1, col_ctrl2 = st.columns([1, 1])
        with col_ctrl1:
            show_boundaries = st.toggle("Show ACI Boundaries (ρ = 1% - 8%)", value=True)
        with col_ctrl2:
            show_keypoints = st.toggle("Highlight Key Points (Max, Balance, Min)", value=True)
            
        st.markdown("---")

        # --- สร้าง High-End Plotly Chart ---
        fig_pm = go.Figure()

        # 1. จัดการเส้นขอบเขต 1% และ 8% (ถ้าเปิดใช้งาน)
        if show_boundaries:
            def get_ref_curve(target_rho):
                target_as = target_rho * engine.Ag
                ref_n_bars = max(4, int(target_as / 3.14)) 
                if shape == "Rectangular":
                    ref_nx = max(2, int(np.sqrt(ref_n_bars * (b/h))))
                    ref_ny = max(2, int(ref_n_bars / 2) - ref_nx + 2)
                    ref_engine = RCColumnProBiaxial(shape, "4-Faces (Uniform)", b, h, fc, fy, 20, 0, ref_nx, ref_ny, cover)
                else:
                    ref_engine = RCColumnProBiaxial(shape, "Circular", b, h, fc, fy, 20, ref_n_bars, 0, 0, cover)
                ref_df, _ = ref_engine.solve_pm(axis='X')
                return ref_df

            with st.spinner("Rendering ACI boundary limits..."):
                df_1pct = get_ref_curve(0.01)
                df_8pct = get_ref_curve(0.08)

            fig_pm.add_trace(go.Scatter(
                x=df_1pct['phiMn'], y=df_1pct['phiPn'],
                name="Min Limit (1%)", mode='lines',
                line=dict(color='rgba(100, 100, 100, 0.4)', width=1.5, dash='dot'),
                hoverinfo='skip'
            ))
            fig_pm.add_trace(go.Scatter(
                x=df_8pct['phiMn'], y=df_8pct['phiPn'],
                name="Max Limit (8%)", mode='lines',
                line=dict(color='rgba(231, 76, 60, 0.3)', width=1.5, dash='dot'),
                fill='tonexty', fillcolor='rgba(46, 204, 113, 0.08)',
                hoverinfo='skip'
            ))

        # 2. เส้น Capacity จริงของหน้าตัด
        fig_pm.add_trace(go.Scatter(
            x=df_x['phiMn'], y=df_x['phiPn'], 
            name=f"X-Axis Capacity", mode='lines',
            line=dict(color='#1f77b4', width=3),
            hovertemplate="<b>X-Axis</b><br>φMn: %{x:.2f} t-m<br>φPn: %{y:.2f} ton<extra></extra>"
        ))
        fig_pm.add_trace(go.Scatter(
            x=df_y['phiMn'], y=df_y['phiPn'], 
            name=f"Y-Axis Capacity", mode='lines',
            line=dict(color='#2ca02c', width=3, dash='dash'),
            hovertemplate="<b>Y-Axis</b><br>φMn: %{x:.2f} t-m<br>φPn: %{y:.2f} ton<extra></extra>"
        ))

        # 3. จุด Key Points (จุดสูงสุด, จุด Balance, จุดดัดล้วน)
        if show_keypoints:
            # คำนวณหาจุด Balance Point โดยประมาณ (จุดที่ Moment สูงสุด) สำหรับแกน X
            bal_idx = df_x['phiMn'].idxmax()
            bal_M, bal_P = df_x.loc[bal_idx, 'phiMn'], df_x.loc[bal_idx, 'phiPn']
            max_P = df_x['phiPn'].max()
            max_M = df_x.loc[df_x['phiPn'] <= 0.01, 'phiMn'].max() if not df_x[df_x['phiPn'] <= 0.01].empty else df_x['phiMn'].iloc[-1]

            # เพิ่มข้อความชี้จุด
            annotations = [
                dict(x=0, y=max_P, xref="x", yref="y", text="Pure Compression", showarrow=True, arrowhead=2, ax=50, ay=0, font=dict(size=10, color="#7f8c8d")),
                dict(x=bal_M, y=bal_P, xref="x", yref="y", text="Balance Point", showarrow=True, arrowhead=2, ax=40, ay=-30, font=dict(size=10, color="#7f8c8d")),
                dict(x=max_M, y=0, xref="x", yref="y", text="Pure Bending", showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(size=10, color="#7f8c8d"))
            ]
            fig_pm.update_layout(annotations=annotations)

        # 4. จุด Demand Load พร้อมเส้นนำสายตา (Crosshairs)
        fig_pm.add_trace(go.Scatter(
            x=[Mcx, Mcy], y=[Pu, Pu], 
            mode='markers', name="Factored Demands", 
            marker=dict(color=['#e74c3c', '#e67e22'], size=14, symbol='cross', line=dict(width=2, color='white')),
            hovertemplate="<b>Demand</b><br>Mc: %{x:.2f} t-m<br>Pu: %{y:.2f} ton<extra></extra>"
        ))

        # ลากเส้นนำสายตาไปยังแกน X และ Y สำหรับ Mcx (สีแดง)
        fig_pm.add_shape(type="line", x0=0, y0=Pu, x1=Mcx, y1=Pu, line=dict(color="#e74c3c", width=1, dash="dot"))
        fig_pm.add_shape(type="line", x0=Mcx, y0=0, x1=Mcx, y1=Pu, line=dict(color="#e74c3c", width=1, dash="dot"))
        # ลากเส้นนำสายตาสำหรับ Mcy (สีส้ม)
        fig_pm.add_shape(type="line", x0=Mcy, y0=0, x1=Mcy, y1=Pu, line=dict(color="#e67e22", width=1, dash="dot"))

        # --- การตกแต่ง Layout ขั้นสุด ---
        fig_pm.update_layout(
            xaxis=dict(
                title="<b>Design Moment, φMn (ton-m)</b>",
                showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)',
                zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.2)',
                rangemode='tozero'
            ),
            yaxis=dict(
                title="<b>Design Axial Strength, φPn (ton)</b>",
                showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)',
                zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.2)'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=650,
            hovermode="closest",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1
            ),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        st.plotly_chart(fig_pm, use_container_width=True)
        
        # กล่องสรุปสถานะใต้กราฟ
        st.markdown(
            f"""
            <div style="padding: 15px; border-radius: 5px; background-color: #f8f9fa; border-left: 5px solid {'#2ecc71' if is_safe else '#e74c3c'};">
                <h4 style="margin-top: 0px; color: #2c3e50;">📊 P-M Analysis Result</h4>
                <p style="margin-bottom: 0px;">The current reinforcement ratio is <strong>{engine.rho*100:.2f}%</strong>. 
                Demand coordinates (M, P) must fall strictly <em>inside</em> the solid capacity curves to be considered structurally safe. 
                Ensure your design also falls within the green optimal zone (1% - 8%) for constructability.</p>
            </div>
            """, unsafe_allow_html=True
        )
        
    with tab3:
        st.markdown("### 🏛️ God-Tier Structural Blueprint & BIM Dashboard")
        
        # --- เตรียมข้อมูลทางวิศวกรรม (Engineering Context) ---
        total_ast = engine.Ag * engine.rho
        # คำนวณ Inertia (พื้นฐานคอนกรีต)
        if shape == "Rectangular":
            Ix = (b * h**3) / 12
            Iy = (h * b**3) / 12
        else:
            Ix = Iy = (np.pi * b**4) / 64

        # ส่วนหัว Dashboard สไตล์ Enterprise
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; padding: 20px; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);">
                <div style="text-align: left;">
                    <p style="margin: 0; color: #38bdf8; font-size: 11px; font-weight: 700; letter-spacing: 1.5px;">PROJECT SECTION</p>
                    <h2 style="margin: 0; color: #ffffff;">{shape.upper()} {b}x{h if shape == 'Rectangular' else b}</h2>
                    <p style="margin: 0; color: #94a3b8; font-size: 13px;">Design Code: ACI-318 / SDM</p>
                </div>
                <div style="text-align: right; border-left: 1px solid #334155; padding-left: 20px;">
                    <p style="margin: 0; color: #94a3b8; font-size: 11px; font-weight: 700;">REBAR RATIO (ρ)</p>
                    <h2 style="margin: 0; color: {'#4ade80' if 0.01 <= engine.rho <= 0.08 else '#fb7185'};">{engine.rho*100:.2f}%</h2>
                    <p style="margin: 0; color: #64748b; font-size: 12px;">{'PASS' if 0.01 <= engine.rho <= 0.08 else 'CHECK LIMIT'}</p>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

        view_2d, view_3d, view_export = st.tabs(["📊 2D Engineering Detail", "🧊 3D BIM Model", "💾 CAD Data & Export"])

        # --- 1. SETUP PARAMETERS (ส่วนกลางที่ใช้ร่วมกันใน tab3) ---
        cv = 4.0 # Covering
        max_d = max(b, h) if shape == "Rectangular" else b
        offset = max_d * 0.25 # ระยะ Offset สำหรับเส้น Dimension
        limit = max_d * 0.8
        
        t_blue = '#38bdf8'
        t_red = '#ef4444'
        t_gold = '#fbbf24'
        t_dark = '#020617'
        t_dim = '#64748b'

        # --- เตรียมพิกัด CONCRETE & TIES ---
        if shape == "Rectangular":
            x_c = [-b/2, b/2, b/2, -b/2, -b/2]
            y_c = [-h/2, -h/2, h/2, h/2, -h/2]
            x_t = [-(b/2-cv), (b/2-cv), (b/2-cv), -(b/2-cv), -(b/2-cv)]
            y_t = [-(h/2-cv), -(h/2-cv), (h/2-cv), (h/2-cv), -(h/2-cv)]
        else:
            theta = np.linspace(0, 2*np.pi, 100)
            x_c, y_c = (b/2)*np.cos(theta), (b/2)*np.sin(theta)
            x_t, y_t = (b/2-cv)*np.cos(theta), (b/2-cv)*np.sin(theta)

        # --- เตรียมพิกัด REBARS ---
        bx = [bar['x'] for bar in engine.bars]
        by = [bar['y'] for bar in engine.bars]

        with view_2d:
            # --- Drawing Controls ---
            c1, c2, c3, c4 = st.columns(4)
            draw_dim = c1.toggle("Dimensions", value=True)
            draw_id = c2.toggle("Rebar Labels", value=True)
            draw_grid = c3.toggle("Grid Lines", value=False)
            draw_spec = c4.toggle("Material Specs", value=True)

            fig = go.Figure()
            
            # Draw Concrete
            fig.add_trace(go.Scatter(x=x_c, y=y_c, mode='lines', line=dict(color=t_blue, width=3), fill='toself', fillcolor='rgba(56, 189, 248, 0.1)', name='Concrete'))
            # Draw Ties
            fig.add_trace(go.Scatter(x=x_t, y=y_t, mode='lines', line=dict(color=t_gold, width=1.5, dash='dash'), name='Stirrups'))

            # --- DIMENSIONS ---
            if draw_dim:
                if shape == "Rectangular":
                    # --- Width (B) Dimension ---
                    y_dim = -h/2 - offset
                    # Extension Lines
                    fig.add_trace(go.Scatter(x=[-b/2, -b/2], y=[-h/2-2, y_dim-2], mode='lines', line=dict(color=t_dim, width=1), showlegend=False))
                    fig.add_trace(go.Scatter(x=[b/2, b/2], y=[-h/2-2, y_dim-2], mode='lines', line=dict(color=t_dim, width=1), showlegend=False))
                    # Main Dim Line
                    fig.add_trace(go.Scatter(x=[-b/2, b/2], y=[y_dim, y_dim], mode='lines+markers', marker=dict(symbol='line-ew-open', size=12, color=t_dim), line=dict(width=1.5), showlegend=False))
                    fig.add_annotation(x=0, y=y_dim, text=f"B = {b} cm", showarrow=False, yshift=12, font=dict(color="white", size=12))

                    # --- Depth (H) Dimension ---
                    x_dim = -b/2 - offset
                    fig.add_trace(go.Scatter(x=[-b/2-2, x_dim-2], y=[-h/2, -h/2], mode='lines', line=dict(color=t_dim, width=1), showlegend=False))
                    fig.add_trace(go.Scatter(x=[-b/2-2, x_dim-2], y=[h/2, h/2], mode='lines', line=dict(color=t_dim, width=1), showlegend=False))
                    fig.add_trace(go.Scatter(x=[x_dim, x_dim], y=[-h/2, h/2], mode='lines+markers', marker=dict(symbol='line-ns-open', size=12, color=t_dim), line=dict(width=1.5), showlegend=False))
                    fig.add_annotation(x=x_dim, y=0, text=f"H = {h} cm", showarrow=False, xshift=-15, textangle=-90, font=dict(color="white", size=12))
                else:
                    # Circular Diameter
                    y_dim = -b/2 - offset
                    fig.add_trace(go.Scatter(x=[-b/2, b/2], y=[y_dim, y_dim], mode='lines+markers', marker=dict(symbol='line-ew-open', size=12, color=t_dim), line=dict(width=1.5), showlegend=False))
                    fig.add_annotation(x=0, y=y_dim, text=f"Ø = {b} cm", showarrow=False, yshift=12, font=dict(color="white", size=12))

            # --- REBARS & LABELS ---
            fig.add_trace(go.Scatter(
                x=bx, y=by, mode='markers+text' if draw_id else 'markers',
                marker=dict(color=t_dark, size=12, line=dict(color=t_red, width=2.5)),
                text=[str(i+1) for i in range(len(bx))], textposition="top center",
                textfont=dict(color="white", size=9),
                name='Main Rebars'
            ))

            # --- MATERIAL SPEC TAGS ---
            if draw_spec:
                spec_text = f"<b>SPECIFICATIONS</b><br>fc' = {fc} MPa<br>fy = {fy} MPa<br>Ast = {total_ast:.2f} cm²"
                fig.add_annotation(
                    xref="paper", yref="paper", x=0.98, y=0.02,
                    text=spec_text, showarrow=False, align="right",
                    bgcolor="rgba(15, 23, 42, 0.8)", bordercolor=t_dim, borderpad=10,
                    font=dict(color=t_blue, size=11)
                )

            # --- Layout Optimization ---
            fig.update_layout(
                plot_bgcolor=t_dark, paper_bgcolor=t_dark,
                xaxis=dict(showgrid=draw_grid, gridcolor='#1e293b', range=[-limit-offset, limit+offset], zeroline=False),
                yaxis=dict(showgrid=draw_grid, gridcolor='#1e293b', range=[-limit-offset, limit+offset], scaleanchor="x", scaleratio=1, zeroline=False),
                height=700, margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(font=dict(color="white"), orientation="h", y=1.05, x=0.5, xanchor="center")
            )
            st.plotly_chart(fig, use_container_width=True)

        with view_3d:
            st.markdown("#### 🧊 Interactive 3D BIM Cage")
            l_col = max_d * 4
            fig3d = go.Figure()
            
            # Rebars 3D
            for i, (x, y) in enumerate(zip(bx, by)):
                fig3d.add_trace(go.Scatter3d(x=[x, x], y=[y, y], z=[0, l_col], mode='lines', line=dict(color=t_red, width=5), name=f"Bar {i+1}"))
            
            # Ties 3D
            for z_pos in np.linspace(10, l_col-10, 8):
                fig3d.add_trace(go.Scatter3d(x=x_t, y=y_t, z=[z_pos]*len(x_t), mode='lines', line=dict(color=t_gold, width=3), showlegend=False))

            fig3d.update_layout(
                scene=dict(aspectmode='data', xaxis_title="X (cm)", yaxis_title="Y (cm)", zaxis_title="Height (cm)",
                            xaxis=dict(backgroundcolor=t_dark, gridcolor="#1e293b"),
                            yaxis=dict(backgroundcolor=t_dark, gridcolor="#1e293b"),
                            zaxis=dict(backgroundcolor=t_dark, gridcolor="#1e293b")),
                margin=dict(l=0, r=0, t=0, b=0), height=600, paper_bgcolor=t_dark
            )
            st.plotly_chart(fig3d, use_container_width=True)

        with view_export:
            c_e1, c_e2 = st.columns(2)
            with c_e1:
                st.markdown("#### 📋 Section Properties")
                prop_df = pd.DataFrame({
                    "Parameter": ["Width (B)", "Depth (H)", "Gross Area (Ag)", "Steel Area (Ast)", "Inertia Ix", "Inertia Iy"],
                    "Value": [b, h if shape == "Rectangular" else b, f"{engine.Ag:.2f}", f"{total_ast:.2f}", f"{Ix:,.0f}", f"{Iy:,.0f}"],
                    "Unit": ["cm", "cm", "cm²", "cm²", "cm⁴", "cm⁴"]
                })
                st.table(prop_df)
            
            with c_e2:
                st.markdown("#### ⌨️ AutoCAD CLI Script")
                st.caption("Paste into AutoCAD Command Line")
                
                # --- แก้ไขให้รองรับเสากลมและเสาเหลี่ยม ---
                if shape == "Rectangular":
                    cad_script = f"COLOR 4\nRECTANG {-b/2},{-h/2} {b/2},{h/2}\nCOLOR 2\nRECTANG {-(b/2-cv)},{-(h/2-cv)} {(b/2-cv)},{(h/2-cv)}\nCOLOR 1\n"
                else:
                    cad_script = f"COLOR 4\nCIRCLE 0,0 {b/2}\nCOLOR 2\nCIRCLE 0,0 {b/2-cv}\nCOLOR 1\n"
                    
                for rx, ry in zip(bx, by):
                    cad_script += f"CIRCLE {rx},{ry} 1.0\n"
                cad_script += "ZOOM E"
                st.code(cad_script, language="bash")

    with tab4:
        st.markdown("### 🏆 Ultimate Shear, Torsion & Detailing Report")
        
        # --- 1. Parameter Initialization ---
        Vux = vux_ton       
        Vuy = vuy_ton       
        Tu = tu_tonm
        cv = cover
        db_cm = db / 10.0  
        d_tie = tie_dia / 10.0 
        tie_str = f"RB{tie_dia}" if tie_dia < 10 else f"DB{tie_dia}"
        
        H_cm = Lu_x * 100
        max_dim = max(b, h) if shape == "Rectangular" else b
        min_dim = min(b, h) if shape == "Rectangular" else b
        
        # Effective Depth (d) parallel to X and Y axes
        dx = b - cv - d_tie - (db_cm / 2) if shape == "Rectangular" else b - cv - d_tie - (db_cm / 2)
        dy = h - cv - d_tie - (db_cm / 2) if shape == "Rectangular" else b - cv - d_tie - (db_cm / 2)

        # --- 2. Shear & Torsion Calculations (ACI 318) ---
        phi_V = 0.75
        
        # Concrete Shear Capacity (Vc)
        Vcx_ton = 0.17 * math.sqrt(fc) * (h * 10) * (dx * 10) / 10000 if shape == "Rectangular" else 0.17 * math.sqrt(fc) * (b * 10) * (dx * 10) / 10000
        Vcy_ton = 0.17 * math.sqrt(fc) * (b * 10) * (dy * 10) / 10000 if shape == "Rectangular" else Vcx_ton
        
        # Torsion Threshold (Tth)
        Acp = b * h if shape == "Rectangular" else math.pi * (b/2)**2
        pcp = 2 * (b + h) if shape == "Rectangular" else math.pi * b
        Tth_tonm = (0.26 * math.sqrt(fc) * (Acp**2) / pcp) / 100000 
        is_torsion_significant = Tu > Tth_tonm

        # --- 3. Lap Splice Length Requirements ---
        lap_compression = max(0.071 * fy * db_cm, 30.0) 
        lap_tension = max(1.3 * 0.12 * (fy / math.sqrt(fc)) * db_cm, 30.0) 

        # --- 4. Seismic Detailing Rules ---
        if is_seismic: 
            seismic_frame_label = "Special Moment Frame (SMF)"
            L0 = max(max_dim, H_cm / 6, 45.0) 
            S0_max = min(min_dim / 4, 6 * db_cm, 15.0)
            S_mid = min(6 * db_cm, 15.0) * 2 
            
            splice_len = lap_tension
            splice_type = "Class B Tension Splice (Seismic Requirement)"
            splice_loc_y0 = (H_cm / 2) - (splice_len / 2)
        else: 
            seismic_frame_label = "Ordinary Frame (Gravity / Wind)"
            L0 = 0 
            S0_max = min(16 * db_cm, 48 * d_tie, min_dim)
            S_mid = S0_max
            
            splice_len = lap_compression
            splice_type = "Compression Splice"
            splice_loc_y0 = 0 

        S0_design = max(math.floor(S0_max / 2.5) * 2.5, 5.0)
        Smid_design = max(math.floor(S_mid / 5.0) * 5.0, 10.0)
        
        Av_x = tie_legs * (math.pi * (d_tie**2) / 4) 
        Av_y = tie_legs * (math.pi * (d_tie**2) / 4) 
        
        Vs_prov_x = (Av_x * fy * dx) / S0_design / 10 
        Vs_prov_y = (Av_y * fy * dy) / Smid_design / 10 
        
        phi_Vnx = phi_V * (Vcx_ton + Vs_prov_x)
        phi_Vny = phi_V * (Vcy_ton + Vs_prov_y)
        
        is_x_safe = phi_Vnx >= Vux
        is_y_safe = phi_Vny >= Vuy

        # --- 5. Dashboard Metrics ---
        st.markdown(f"**Applied Code Provisions:** `{seismic_frame_label}` (ACI 318-19)")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Max Shear X (Vux)", f"{Vux:.2f} ton", delta="SAFE" if is_x_safe else "UNSAFE", delta_color="normal" if is_x_safe else "inverse")
        col_m2.metric("Shear Cap X (φVnx)", f"{phi_Vnx:.2f} ton")
        col_m3.metric("Max Shear Y (Vuy)", f"{Vuy:.2f} ton", delta="SAFE" if is_y_safe else "UNSAFE", delta_color="normal" if is_y_safe else "inverse")
        col_m4.metric("Shear Cap Y (φVny)", f"{phi_Vny:.2f} ton")
        
        col_m5, col_m6, col_m7, col_m8 = st.columns(4)
        col_m5.metric("Factored Torsion (Tu)", f"{Tu:.2f} ton-m")
        col_m6.metric("Torsion Action", "Critical" if is_torsion_significant else "Ignorable", delta=f"Threshold: {Tth_tonm:.2f}", delta_color="off" if not is_torsion_significant else "inverse")
        col_m7.metric("Lap Splice Length", f"{splice_len:.0f} cm")
        col_m8.metric("Splice Rule", f"{splice_type}")

        # --- 6. Detailed Calculation Report (ACI 318-19 Professional Edition) ---
        with st.expander("📝 Comprehensive Calculation Report (Strict ACI 318-19)", expanded=False):
            # Section 1 & 2 remain similar but with clearer formatting
            st.markdown(f"""
            #### 1. Effective Depth Calculation
            * **X-Axis ($d_x$):** $b - cover - d_{{tie}} - (d_b/2) = {b} - {cv} - {d_tie:.2f} - {db_cm/2:.2f} = $ **{dx:.2f} cm**
            * **Y-Axis ($d_y$):** $h - cover - d_{{tie}} - (d_b/2) = {h} - {cv} - {d_tie:.2f} - {db_cm/2:.2f} = $ **{dy:.2f} cm**

            #### 2. Concrete Shear Capacity ($V_c$)
            Ref. ACI 318-19 Table 22.5.5.1: $\phi V_c = \phi (0.17 \lambda \sqrt{{f'_c}} b_w d)$
            * **$\phi V_{{cx}}$:** $0.75 \times 0.17 \times \sqrt{{{fc}}} \times {h*10} \times {dx*10} / 10000 = $ **{phi_V * Vcx_ton:.2f} ton**
            * **$\phi V_{{cy}}$:** $0.75 \times 0.17 \times \sqrt{{{fc}}} \times {b*10} \times {dy*10} / 10000 = $ **{phi_V * Vcy_ton:.2f} ton**

            #### 3. Steel Shear Contribution ($V_s$)
            Ref. ACI 318-19 Sec. 22.5.8.5.3: $V_s = (A_v f_{{yt}} d) / s$
            * **Provided:** `{tie_str}` @ `{S0_design:.1f}` cm ( {tie_legs} legs ) $\rightarrow A_v = {Av_x:.2f} \text{{ cm}}^2$
            * **$\phi V_{{sx}}$:** $0.75 \times ({Av_x:.2f} \times {fy} \times {dx:.2f}) / ({S0_design:.1f} \times 10) = $ **{phi_V * Vs_prov_x:.2f} ton**
            * **$\phi V_{{sy}}$:** $0.75 \times ({Av_y:.2f} \times {fy} \times {dy:.2f}) / ({Smid_design:.1f} \times 10) = $ **{phi_V * Vs_prov_y:.2f} ton**

            #### 4. Total Shear Design & Verification (Demand vs. Capacity)
            Ref. ACI 318-19 Sec. 22.5.1.1: $\phi V_n \geq V_u$
            * **X-Direction:** $\phi V_{{nx}} = {phi_V * Vcx_ton:.2f} + {phi_V * Vs_prov_x:.2f} = $ **{phi_Vnx:.2f} ton**
                * Demand $V_{{ux}} = {Vux:.2f}$ ton | **D/C Ratio:** `{Vux/phi_Vnx:.2f}` $\rightarrow$ **{"✅ PASS" if is_x_safe else "❌ FAIL"}**
            * **Y-Direction:** $\phi V_{{ny}} = {phi_V * Vcy_ton:.2f} + {phi_V * Vs_prov_y:.2f} = $ **{phi_Vny:.2f} ton**
                * Demand $V_{{uy}} = {Vuy:.2f}$ ton | **D/C Ratio:** `{Vuy/phi_Vny:.2f}` $\rightarrow$ $\rightarrow$ **{"✅ PASS" if is_y_safe else "❌ FAIL"}**

            #### 5. Torsional Threshold & Interaction
            Ref. ACI 318-19 Sec. 22.7.4.1
            * **Threshold $T_{{th}}$:** $0.26 \sqrt{{f'_c}} (A_{{cp}}^2 / p_{{cp}}) = $ **{Tth_tonm:.2f} ton-m**
            * **Check:** $T_u ({Tu:.2f})$ vs $T_{{th}} ({Tth_tonm:.2f})$ $\rightarrow$ **{"Action Required" if is_torsion_significant else "Negligible"}**

            #### 6. Development & Splice Length (Detailed Selection)
            Ref. ACI 318-19 Chapter 25 & 18
            * **Design Criteria:** `{seismic_frame_label}`
            * **Compression Splice ($l_{{sc}}$):** $0.071 f_y d_b \geq 30cm = $ **{lap_compression:.2f} cm**
            * **Tension Splice ($l_{{st}}$ - Class B):** $1.3 \times l_d = $ **{lap_tension:.2f} cm**
            * **Final Decision:** Used **{splice_len:.0f} cm** ({splice_type}).
            * *Note: For Special Moment Frames, ACI 18.7.5.3 mandates Class B tension splices located in the center half of the column.*
            """)
        
        st.markdown("---")

        # --- 7. Detailing Visualizations ---
        st.markdown("#### 📐 Engineering Detailing Views")
        col_plot1, col_plot2 = st.columns([1.2, 1])
        
        with col_plot1:
            st.caption(f"📍 Elevation View (Rebar Layout Profile)")
            fig_elev = go.Figure()
            
            # Column Outline
            fig_elev.add_shape(type="rect", x0=0, y0=0, x1=max_dim, y1=H_cm, line=dict(color="#e2e8f0"), fillcolor="#f8fafc")
            fig_elev.add_shape(type="line", x0=cv, y0=0, x1=cv, y1=H_cm, line=dict(color="#ef4444", width=4))
            fig_elev.add_shape(type="line", x0=max_dim-cv, y0=0, x1=max_dim-cv, y1=H_cm, line=dict(color="#ef4444", width=4))
            
            # Lap Splice Zone
            fig_elev.add_shape(type="rect", x0=cv-5, y0=splice_loc_y0, x1=max_dim-cv+5, y1=splice_loc_y0+splice_len, 
                               line=dict(color="#f97316", width=2, dash="dash"), fillcolor="rgba(249, 115, 22, 0.1)")
            fig_elev.add_annotation(x=max_dim/2, y=splice_loc_y0+(splice_len/2), text=f"Lap Splice<br>{splice_len:.0f} cm", 
                                    showarrow=False, font=dict(color="#c2410c", size=12, weight="bold"))

            # Transverse Reinforcement (Ties)
            y_ties = []
            current_y = S0_design / 2
            while current_y <= H_cm:
                y_ties.append(current_y)
                current_y += S0_design if (current_y <= L0 or current_y >= (H_cm - L0)) else Smid_design
            for ty in y_ties:
                fig_elev.add_shape(type="line", x0=cv, y0=ty, x1=max_dim-cv, y1=ty, line=dict(color="#3b82f6", width=2))
            
            # Seismic Zones Annotations
            if is_seismic:
                fig_elev.add_shape(type="rect", x0=-20, y0=0, x1=0, y1=L0, fillcolor="rgba(16, 185, 129, 0.15)", line_width=0)
                fig_elev.add_annotation(x=-25, y=L0/2, text=f"L0: {tie_str}@{S0_design:.1f}cm", showarrow=False, textangle=-90, font=dict(color="#059669", size=11))
                
                fig_elev.add_shape(type="rect", x0=-20, y0=H_cm-L0, x1=0, y1=H_cm, fillcolor="rgba(16, 185, 129, 0.15)", line_width=0)
                fig_elev.add_annotation(x=-25, y=H_cm-(L0/2), text=f"L0: {tie_str}@{S0_design:.1f}cm", showarrow=False, textangle=-90, font=dict(color="#059669", size=11))
                
                fig_elev.add_annotation(x=-25, y=H_cm/2, text=f"Mid: {tie_str}@{Smid_design:.1f}cm", showarrow=False, textangle=-90, font=dict(color="#64748b", size=11))

            fig_elev.update_layout(xaxis=dict(visible=False, range=[-45, max_dim+15]), yaxis=dict(title="Clear Height (cm)", range=[-10, H_cm+10]), height=550, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_elev, use_container_width=True)

        with col_plot2:
            st.caption("📍 Cross-Section & Load Application")
            fig_plan = go.Figure()
            
            h_plot = h if shape == "Rectangular" else b
            cx, cy = b / 2, h_plot / 2
            
            # Cross Section Outline & Stirrups
            if shape == "Rectangular":
                fig_plan.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h_plot, line=dict(color="#94a3b8", width=2), fillcolor="#f1f5f9")
                tie_x0, tie_y0 = cv, cv
                tie_x1, tie_y1 = b - cv, h_plot - cv
                fig_plan.add_shape(type="rect", x0=tie_x0, y0=tie_y0, x1=tie_x1, y1=tie_y1, line=dict(color="#3b82f6", width=3))
            else:
                fig_plan.add_shape(type="circle", x0=0, y0=0, x1=b, y1=h_plot, line=dict(color="#94a3b8", width=2), fillcolor="#f1f5f9")
                tie_x0, tie_y0 = cv, cv
                tie_x1, tie_y1 = b - cv, h_plot - cv
                fig_plan.add_shape(type="circle", x0=tie_x0, y0=tie_y0, x1=tie_x1, y1=tie_y1, line=dict(color="#3b82f6", width=3))
            
            # Longitudinal Reinforcement
            dot_r = max(1.5, db_cm/2)
            if shape == "Rectangular":
                corners = [(tie_x0, tie_y0), (tie_x1, tie_y0), (tie_x0, tie_y1), (tie_x1, tie_y1)]
                for p_x, p_y in corners:
                    fig_plan.add_shape(type="circle", x0=p_x-dot_r, y0=p_y-dot_r, x1=p_x+dot_r, y1=p_y+dot_r, fillcolor="#ef4444", line_color="#b91c1c")
            else:
                for i in range(8): 
                    angle = i * (2 * math.pi / 8)
                    p_x = cx + ((b/2 - cv) * math.cos(angle))
                    p_y = cy + ((b/2 - cv) * math.sin(angle))
                    fig_plan.add_shape(type="circle", x0=p_x-dot_r, y0=p_y-dot_r, x1=p_x+dot_r, y1=p_y+dot_r, fillcolor="#ef4444", line_color="#b91c1c")

            # Force Annotations
            fig_plan.add_annotation(x=b*1.1, y=cy, ax=b*0.6, ay=cy, xref="x", yref="y", axref="x", ayref="y", text="Vux", showarrow=True, arrowhead=3, arrowsize=1.5, arrowcolor="#f59e0b", font=dict(color="#d97706", size=14, weight="bold"))
            fig_plan.add_annotation(x=cx, y=h_plot*1.1, ax=cx, ay=h_plot*0.6, xref="x", yref="y", axref="x", ayref="y", text="Vuy", showarrow=True, arrowhead=3, arrowsize=1.5, arrowcolor="#059669", font=dict(color="#047857", size=14, weight="bold"))
            fig_plan.add_annotation(x=cx, y=cy, text="↺ Tu", showarrow=False, font=dict(color="#8b5cf6", size=20, weight="bold"))

            # Cover Label
            fig_plan.add_annotation(x=b*0.1, y=h_plot*0.9, text=f"Cover: {cv} cm", showarrow=False, font=dict(color="#64748b", size=11))

            fig_plan.update_layout(
                xaxis=dict(visible=False, range=[-b*0.3, b*1.3]),
                yaxis=dict(visible=False, range=[-h_plot*0.3, h_plot*1.3], scaleanchor="x", scaleratio=1),
                height=550, margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_plan, use_container_width=True)
            
            # Critical Warning
            if is_torsion_significant:
                st.error("🚨 **CRITICAL TORSION ALERT:** The factored torsional moment exceeds the allowable threshold. ACI 318 dictates the implementation of closed stirrups with 135-degree seismic hooks and supplemental longitudinal reinforcement.")

    with tab5:
        st.markdown("### 📖 Parameter Guide")
        st.markdown("---")
        st.markdown("#### 1. Applied Loads")
        st.markdown("* **Pu (Factored Axial Load):** The ultimate axial load acting on the column. *(Unit: tons)*")
        st.markdown("* **Mux, Muy (Factored Moments):** The ultimate bending moments acting about the X and Y axes. *(Unit: ton-m)*")
        st.markdown("#### 2. Frame Type")
        st.markdown("* **Non-Sway Frame (Braced Frame):** A structure equipped with a stiff lateral force-resisting system. Joints experience practically no lateral translation.")
        st.markdown("* **Sway Frame (Unbraced Frame):** Relies entirely on the stiffness of its beams and columns. Joints can translate laterally, generating P-Delta effect.")

    with tab6:
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
