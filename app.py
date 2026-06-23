"""
RC Column Pro – Biaxial & Sway Analysis
ACI 318-19 / SDM (MKS Unit System: ksc, ton, cm, ton-m)

UNIT SYSTEM (consistent throughout):
  Force       : ton   (1 ton = 1000 kgf)
  Moment      : ton-m
  Length      : cm
  Stress (fc) : ksc   (kgf/cm²)
  Stress (fy) : ksc
  Ec, Es      : ksc
  Area        : cm²
  Inertia     : cm⁴
  EI          : ksc·cm²  → Pc in kgf → /1000 → ton

All ACI formulae that reference √f'c use the MKS coefficient (0.53, 1.06, etc.)
rather than the SI coefficient (0.17, 0.33).  Conversions:
  0.53 √f'c [ksc] ≡ 0.17 √f'c [MPa]  (since 1 MPa ≈ 10.2 ksc → √10.2 ≈ 3.19)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
from scipy.interpolate import interp1d

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
PHI_FLEX   = 0.90   # Tension-controlled sections
PHI_COMP_T = 0.65   # Compression-controlled, tied
PHI_COMP_S = 0.75   # Compression-controlled, spiral/circular
PHI_SHEAR  = 0.75   # Shear & torsion
ES_KSC     = 2_040_000.0   # ksc  (≈ 200 GPa)
EPS_CU     = 0.003          # ACI ultimate concrete strain
EPS_TY_LIM = 0.005          # Tension-controlled limit

# ──────────────────────────────────────────────────────────────────────────────
# ENGINE
# ──────────────────────────────────────────────────────────────────────────────
class RCColumn:
    """
    Generates P-M interaction data and helper calculations for a rectangular
    or circular RC column.  All inputs/outputs in MKS (ksc, ton, cm).
    """

    def __init__(self, shape, layout, b, h, fc, fy, db_mm, n_bars,
                 nx, ny, cover_cm):
        self.shape  = shape
        self.layout = layout
        self.b, self.h = float(b), float(h)
        self.fc, self.fy = float(fc), float(fy)
        self.Es = ES_KSC

        # β₁  ACI 318-19 Table 22.2.2.4.3  (fc in ksc; threshold 280 ksc ≈ 28 MPa)
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (self.fc - 280.0) / 70.0))

        # Tension-controlled strain limit (ACI 318-19)
        self.eps_ty_lim = (self.fy / self.Es) + 0.003

        # Section geometry
        if shape == "Rectangular":
            self.Ag  = self.b * self.h
            self.Igx = self.b * self.h**3 / 12.0
            self.Igy = self.h * self.b**3 / 12.0
            self.rx  = 0.3 * self.h   # radius of gyration about X-axis (bending in h-direction)
            self.ry  = 0.3 * self.b   # radius of gyration about Y-axis
        else:  # Circular
            self.D   = self.h          # diameter = h input
            self.Ag  = math.pi * self.D**2 / 4.0
            self.Igx = self.Igy = math.pi * self.D**4 / 64.0
            self.rx  = self.ry = 0.25 * self.D

        # Concrete modulus  (Ec = 15100√f'c  for normal-weight concrete, MKS)
        self.Ec = 15_100.0 * math.sqrt(self.fc)

        # Bar geometry
        self.db_cm    = db_mm / 10.0
        self.as_bar   = math.pi * self.db_cm**2 / 4.0
        # Clear cover → centroid of bar:  cover + tie(≈9mm) + db/2
        self.d_prime  = cover_cm + 0.9 + self.db_cm / 2.0

        # Bar layout
        self.bars = self._place_bars(n_bars, nx, ny)
        self.n_bars    = len(self.bars)
        self.total_as  = self.n_bars * self.as_bar
        self.rho       = self.total_as / self.Ag

        # Steel moment of inertia (used in EI calculation)
        self.Ise_x = sum(self.as_bar * bar['y']**2 for bar in self.bars)
        self.Ise_y = sum(self.as_bar * bar['x']**2 for bar in self.bars)

    # ── bar placement ──────────────────────────────────────────────────────────
    def _place_bars(self, n_bars, nx, ny):
        bars = []
        dp = self.d_prime
        if self.shape == "Rectangular":
            x_min = -self.b / 2 + dp
            x_max =  self.b / 2 - dp
            y_min = -self.h / 2 + dp
            y_max =  self.h / 2 - dp

            if self.layout == "2-Faces (Top/Bottom)":
                n_each = max(2, n_bars // 2)
                for x in np.linspace(x_min, x_max, n_each):
                    bars.append({'x': float(x), 'y': float(y_max)})
                    bars.append({'x': float(x), 'y': float(y_min)})

            else:  # 4-Faces
                nx = max(2, int(nx))
                ny = max(2, int(ny))
                for x in np.linspace(x_min, x_max, nx):
                    bars.append({'x': float(x), 'y': float(y_max)})
                    bars.append({'x': float(x), 'y': float(y_min)})
                if ny > 2:
                    for y in np.linspace(y_min, y_max, ny)[1:-1]:
                        bars.append({'x': float(x_min), 'y': float(y)})
                        bars.append({'x': float(x_max), 'y': float(y)})
        else:  # Circular
            Rs = self.D / 2.0 - dp
            for i in range(n_bars):
                theta = i * 2 * math.pi / n_bars
                bars.append({'x': Rs * math.sin(theta), 'y': Rs * math.cos(theta)})
        return bars

    # ── P-M interaction ────────────────────────────────────────────────────────
    def solve_pm(self, axis='X'):
        """
        Returns (DataFrame of P-M points, phi*Pn_max).
        Columns: c, Pn [ton], Mn [ton-m], phiPn [ton], phiMn [ton-m]
        Sign convention: compression positive.
        """
        is_circular = (self.shape == "Circular")
        phi_comp    = PHI_COMP_S if is_circular else PHI_COMP_T

        if is_circular:
            depth = self.D
            width = self.D
            get_coord = lambda bar: bar['y']
        else:
            if axis == 'X':
                depth, width = self.h, self.b
                get_coord = lambda bar: bar['y']
            else:
                depth, width = self.b, self.h
                get_coord = lambda bar: bar['x']

        y_bars = np.array([get_coord(b) for b in self.bars], dtype=float)
        d_bars = depth / 2.0 - y_bars   # distance from compression face
        dt     = float(np.max(d_bars))  # extreme tension steel depth

        results = []

        # Sweep c from deep compression (pure compression) to near-zero (near pure tension)
        c_vals = np.concatenate([
            np.linspace(depth * 10.0, dt + 0.001, 150),
            np.linspace(dt, 0.01 * depth, 150)
        ])
        c_vals = np.unique(np.clip(c_vals, 1e-6, None))

        for c in c_vals:
            a = min(self.beta1 * c, depth)

            # ── Concrete compression resultant ──────────────────────────────
            if is_circular:
                R = self.D / 2.0
                if a >= self.D:
                    Ac    = math.pi * R**2
                    y_bar = 0.0
                else:
                    ratio = max(-1.0, min(1.0, (R - a) / R))
                    theta_c = 2.0 * math.acos(ratio)
                    Ac    = (R**2 / 2.0) * (theta_c - math.sin(theta_c))
                    if Ac > 1e-10:
                        y_bar = (4.0 * R * math.sin(theta_c / 2.0)**3) / \
                                (3.0 * (theta_c - math.sin(theta_c)))
                    else:
                        y_bar = R
                Cc = 0.85 * self.fc * Ac           # kgf
                Mc = Cc * y_bar                     # kgf·cm  (from section centroid)
            else:
                Cc = 0.85 * self.fc * a * width     # kgf
                Mc = Cc * (depth / 2.0 - a / 2.0)  # kgf·cm

            # ── Steel forces ────────────────────────────────────────────────
            eps_s = EPS_CU * (c - d_bars) / c      # + = compression
            fs    = np.clip(eps_s * self.Es, -self.fy, self.fy)  # ksc
            Fsi   = self.as_bar * fs                 # kgf per bar

            Pn_s  = float(np.sum(Fsi))              # kgf
            Mn_s  = float(np.sum(Fsi * y_bars))    # kgf·cm

            # ── Totals (convert to ton and ton-m) ──────────────────────────
            Pn = (Cc + Pn_s)  / 1_000.0            # ton
            Mn = (Mc + Mn_s)  / 100_000.0          # ton-m

            # ── φ factor (extreme tension strain in outermost steel) ────────
            et   = EPS_CU * (dt - c) / c
            ey   = self.fy / self.Es
            if et >= EPS_TY_LIM:
                phi = PHI_FLEX
            elif et <= ey:
                phi = phi_comp
            else:
                phi = phi_comp + (PHI_FLEX - phi_comp) * \
                      (et - ey) / (EPS_TY_LIM - ey)

            results.append({
                'c': c, 'Pn': Pn, 'Mn': abs(Mn),
                'phiPn': phi * Pn, 'phiMn': phi * abs(Mn)
            })

        # ── Pure tension point ──────────────────────────────────────────────
        Pn_tension = -self.total_as * self.fy / 1_000.0  # ton (negative)
        results.append({'c': 0.0, 'Pn': Pn_tension, 'Mn': 0.0,
                        'phiPn': PHI_FLEX * Pn_tension, 'phiMn': 0.0})

        # ── Maximum axial compression (pure squash) ─────────────────────────
        Po = (0.85 * self.fc * (self.Ag - self.total_as) +
              self.fy * self.total_as) / 1_000.0          # ton
        phi_max_factor = 0.85 if is_circular else 0.80    # ACI 318-19 §22.4.2
        phi_pn_max     = phi_comp * phi_max_factor * Po

        results.append({'c': 1e6, 'Pn': Po, 'Mn': 0.0,
                        'phiPn': phi_comp * Po, 'phiMn': 0.0})

        df = pd.DataFrame(results).sort_values('Pn').reset_index(drop=True)
        # Clip phiPn to the code-limited maximum (ACI 22.4.2)
        df['phiPn'] = df['phiPn'].clip(upper=phi_pn_max)
        # Remove duplicate phiPn values to keep interp1d monotonic
        df = df.drop_duplicates(subset=['phiPn'], keep='first')

        return df, phi_pn_max

    # ── Slenderness magnification (Non-Sway) ───────────────────────────────────
    def slenderness_magnifier(self, Pu_ton, K, Lu_m, axis, Cm, beta_d):
        """
        ACI 318-19 §6.6.4  –  Moment Magnification (Nonsway frames).
        Returns (kl/r, Pc [ton], δ, Ise [cm⁴], EI [ksc·cm²])
        """
        if K <= 0 or Lu_m <= 0:
            # Guard against degenerate sway inputs
            r  = self.rx if axis == 'X' else self.ry
            Ig = self.Igx if axis == 'X' else self.Igy
            Ise = self.Ise_x if axis == 'X' else self.Ise_y
            EI  = 0.2 * self.Ec * Ig + self.Es * Ise
            return 0.0, 1e9, 1.0, Ise, EI

        Lu_cm = Lu_m * 100.0
        r    = self.rx  if axis == 'X' else self.ry
        Ig   = self.Igx if axis == 'X' else self.Igy
        Ise  = self.Ise_x if axis == 'X' else self.Ise_y

        kl_r = K * Lu_cm / r

        # EI  ACI 318-19 Eq. (6.6.4.4.4a)
        EI = (0.2 * self.Ec * Ig + self.Es * Ise) / (1.0 + beta_d)

        # Euler critical load   [ton]
        Pc = (math.pi**2 * EI) / (K * Lu_cm)**2 / 1_000.0

        # δns  ACI 318-19 Eq. (6.6.4.5.2)
        denom = 1.0 - Pu_ton / (0.75 * Pc) if Pc > 0 else -1.0
        if denom <= 0:
            delta = 999.9
        else:
            delta = max(1.0, Cm / denom)

        return kl_r, Pc, delta, Ise, EI

    # ── Clear spacing check ────────────────────────────────────────────────────
    def check_clear_spacing(self, nx, ny):
        """
        ACI 318-19 §25.2.3: clear spacing ≥ max(4/3·dagg, db, 1 in.)
        Using simplified MKS: ≥ max(2.5 cm, 1.5·db)
        """
        min_req = max(2.5, 1.5 * self.db_cm)

        if self.shape == "Rectangular":
            dp   = self.d_prime
            sx = sy = 999.0
            if self.layout == "2-Faces (Top/Bottom)":
                n_each = self.n_bars // 2
                if n_each > 1:
                    sx = (self.b - 2 * dp) / (n_each - 1) - self.db_cm
            else:
                nx = max(2, int(nx)); ny = max(2, int(ny))
                if nx > 1:
                    sx = (self.b - 2 * dp) / (nx - 1) - self.db_cm
                if ny > 1:
                    sy = (self.h - 2 * dp) / (ny - 1) - self.db_cm
            actual = min(sx, sy)
        else:
            Rs     = self.D / 2.0 - self.d_prime
            chord  = 2.0 * Rs * math.sin(math.pi / self.n_bars)
            actual = chord - self.db_cm

        return actual, min_req, actual >= min_req

    # ── PCA exponent α ─────────────────────────────────────────────────────────
    def get_alpha(self, Pu_ton):
        """
        Interpolated PCA α between 1.15 (low axial) and 1.55 (high axial).
        For circular sections α = 2.0 (Bresler bilinear is exact for circles).
        """
        if self.shape == "Circular":
            return 2.0
        Po_ton  = (0.85 * self.fc * (self.Ag - self.total_as) +
                   self.fy * self.total_as) / 1_000.0
        phi_Po  = PHI_COMP_T * Po_ton
        if phi_Po <= 0:
            return 1.15
        ratio = max(0.0, min(1.0, Pu_ton / phi_Po))
        return 1.15 + (ratio - 0.1) * (1.55 - 1.15) / 0.9 if ratio > 0.1 else 1.15

    # ── 3D surface ──────────────────────────────────────────────────────────────
    def generate_3d_surface(self, df_x, df_y, alpha, n_p=30, n_t=25):
        """
        Builds the biaxial failure surface using the PCA load-contour method.
        Vectorized with NumPy for fast Plotly rendering.
        """
        p_lo = max(df_x['phiPn'].min(), df_y['phiPn'].min()) + 0.01
        p_hi = min(df_x['phiPn'].max(), df_y['phiPn'].max()) - 0.01
        
        if p_lo >= p_hi:
            return np.array([]), np.array([]), np.array([])

        p_steps = np.linspace(p_lo, p_hi, n_p)
        theta = np.linspace(0, math.pi / 2.0, n_t)
        
        # Create 2D arrays for vectorization
        T, P = np.meshgrid(theta, p_steps)
        
        # NumPy interpolation requires strictly ascending x-coordinates
        mx_cap = np.interp(P, df_x['phiPn'], df_x['phiMn'])
        my_cap = np.interp(P, df_y['phiPn'], df_y['phiMn'])
        
        # Suppress warnings for division by zero inside the contour boundary
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = ((np.cos(T) / mx_cap)**alpha + (np.sin(T) / my_cap)**alpha) ** (1.0 / alpha)
            R = np.where(denom > 0, 1.0 / denom, 0.0)
            
        X = R * np.cos(T)
        Y = R * np.sin(T)
        Z = P
        
        return X, Y, Z


# ──────────────────────────────────────────────────────────────────────────────
# SHEAR & TORSION HELPER  (all in MKS: ksc, ton, cm)
# ──────────────────────────────────────────────────────────────────────────────
def shear_torsion_check(engine, Pu_ton, Vux_ton, Vuy_ton, Tu_tonm,
                        tie_dia_mm, tie_legs, is_seismic, Lu_x_m):
    """
    ACI 318-19 shear check with exact Table 22.5.5.1 equation via MPa conversion.
    """
    fc   = engine.fc
    fy   = engine.fy
    phi  = PHI_SHEAR
    Pu_kgf = Pu_ton * 1_000.0

    d_tie  = tie_dia_mm / 10.0  # cm
    At     = math.pi * d_tie**2 / 4.0  # cm²
    Av     = tie_legs * At              # cm² per stirrup set

    if engine.shape == "Rectangular":
        dx = engine.h - engine.d_prime
        dy = engine.b - engine.d_prime
        bwx = engine.b
        bwy = engine.h
        Acp = engine.b * engine.h
        pcp = 2.0 * (engine.b + engine.h)
        min_dim = min(engine.b, engine.h)
    else:
        dx = dy = 0.8 * engine.D
        bwx = bwy = engine.D
        Acp = engine.Ag
        pcp = math.pi * engine.D
        min_dim = engine.D

    # ── Concrete shear capacity (ACI 318-19 Table 22.5.5.1) ────────────
    # Convert parameters to MPa for the standard ACI formula
    fc_mpa = fc / 10.197
    stress_pu_mpa = (Pu_kgf / engine.Ag) / 10.197
    
    # ACI limit: Nu/(6*Ag) cannot exceed 0.05*f'c (in MPa) [ACI 22.5.5.1.1]
    stress_pu_mpa = min(stress_pu_mpa, 0.05 * fc_mpa)

    # Vc stress in MPa: 0.17·√f'c + Nu/(6·Ag)
    vc_stress_mpa = 0.17 * math.sqrt(fc_mpa) + (stress_pu_mpa / 6.0)
    
    # Convert back to MKS stress (ksc) and calculate capacity
    vc_stress_ksc = vc_stress_mpa * 10.197
    
    Vcx_kgf = vc_stress_ksc * bwx * dx
    Vcy_kgf = vc_stress_ksc * bwy * dy
    Vcx_ton = Vcx_kgf / 1_000.0
    Vcy_ton = Vcy_kgf / 1_000.0

    # ── Torsion threshold  ACI 318-19 §22.7.4.1 (MKS) ─────────────
    Tth_kgcm = phi * 0.026 * math.sqrt(fc) * (Acp**2 / pcp)
    Tth_tonm = Tth_kgcm / 100_000.0
    torsion_critical = Tu_tonm > Tth_tonm

    # ── Spacing limits ──────────────────────────────────────
    s_max_x = min(dx / 2.0, 60.0)
    s_max_y = min(dy / 2.0, 60.0)

    s_seismic = 999.0
    if is_seismic:
        s_seismic = min(min_dim / 4.0, 6.0 * engine.db_cm, 15.0)

    # ── Required stirrup spacing ──────────────────────────
    Vsx_req_ton = max(0.0, Vux_ton / phi - Vcx_ton)
    Vsy_req_ton = max(0.0, Vuy_ton / phi - Vcy_ton)

    def req_spacing(Vs_ton, Av_cm2, d_cm):
        Vs_kgf = Vs_ton * 1_000.0
        if Vs_kgf < 1.0: return 999.0
        return (Av_cm2 * fy * d_cm) / Vs_kgf

    sx_shear = req_spacing(Vsx_req_ton, Av, dx)
    sy_shear = req_spacing(Vsy_req_ton, Av, dy)

    s_gov = min(sx_shear, sy_shear, s_max_x, s_max_y, s_seismic)
    s_design = max(5.0, math.floor(s_gov / 2.5) * 2.5)

    Vsx_prov_ton = (Av * fy * dx) / (s_design * 1_000.0)
    Vsy_prov_ton = (Av * fy * dy) / (s_design * 1_000.0)
    phiVnx = phi * (Vcx_ton + Vsx_prov_ton)
    phiVny = phi * (Vcy_ton + Vsy_prov_ton)

    Vs_max_x_ton = 2.12 * math.sqrt(fc) * bwx * dx / 1_000.0
    Vs_max_y_ton = 2.12 * math.sqrt(fc) * bwy * dy / 1_000.0
    section_adequate_x = Vsx_req_ton <= Vs_max_x_ton
    section_adequate_y = Vsy_req_ton <= Vs_max_y_ton

    return {
        'dx': dx, 'dy': dy, 'bwx': bwx, 'bwy': bwy,
        'Av': Av, 'At': At, 'd_tie': d_tie,
        'Vcx_ton': Vcx_ton, 'Vcy_ton': Vcy_ton,
        'Vsx_prov_ton': Vsx_prov_ton, 'Vsy_prov_ton': Vsy_prov_ton,
        'phiVnx': phiVnx, 'phiVny': phiVny,
        'Tth_tonm': Tth_tonm, 'torsion_critical': torsion_critical,
        's_design': s_design, 's_seismic': s_seismic,
        's_max_x': s_max_x, 's_max_y': s_max_y,
        'sx_shear': sx_shear, 'sy_shear': sy_shear,
        'Vs_max_x_ton': Vs_max_x_ton, 'Vs_max_y_ton': Vs_max_y_ton,
        'section_adequate_x': section_adequate_x,
        'section_adequate_y': section_adequate_y,
        'x_ok': phiVnx >= Vux_ton,
        'y_ok': phiVny >= Vuy_ton,
        'Acp': Acp, 'pcp': pcp,
        'min_dim': min_dim,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LAP SPLICE (ACI 318-19 §25.5)
# ──────────────────────────────────────────────────────────────────────────────
def lap_splice_lengths(fc_ksc, fy_ksc, db_cm):
    """
    Lap splice lengths in cm (MKS system).

    Tension development length (ACI 318-19 §25.5.2.1 converted to MKS):
      ld = (3 fy [MPa]) / (40 λ √fc [MPa]) × ψ_factors / [(cb+Ktr)/db] × db [mm]  → mm
    Assumptions: normal-weight concrete (λ=1), uncoated bars (ψe=1, ψt=1, ψs=1),
    confined case (cb+Ktr)/db = 2.5 (adequate cover + ties).
    Convert MPa → ksc (/10.2), mm → cm (/10).

    Class B Tension Splice  = 1.3 × ld  (ACI §25.5.2.1)
    Compression Splice      = max(0.00711 × fy[ksc] × db[cm], 30 cm)
                              [derived from ACI §25.5.5.1(a): 0.0005 fy_psi × db_in]
    """
    fy_mpa = fy_ksc / 10.2
    fc_mpa = fc_ksc / 10.2
    db_mm  = db_cm  * 10.0
    ratio  = 2.5       # (cb+Ktr)/db — confined, adequate cover
    ld_mm  = (3.0 * fy_mpa) / (40.0 * math.sqrt(fc_mpa)) * db_mm / ratio
    ld_cm  = max(ld_mm / 10.0, 30.0)
    l_splice_B    = max(1.3 * ld_cm, 30.0)
    l_compression = max(0.00711 * fy_ksc * db_cm, 30.0)   # ACI §25.5.5.1
    return l_splice_B, l_compression


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RC Column Pro", layout="wide", page_icon="🏗️")
st.title("🏗️ RC Column Pro — Biaxial & Sway Analysis (ACI 318-19, MKS)")

col1, col_main = st.columns([1, 2.5])

# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL – INPUTS
# ══════════════════════════════════════════════════════════════════════════════
with col1:
    with st.expander("1. Section & Reinforcement", expanded=True):
        shape = st.radio("Section Shape", ["Rectangular", "Circular"], horizontal=True, key="shape")
        fc    = st.number_input("f'c (ksc)", value=280, min_value=140, max_value=700, key="fc_main")
        fy    = st.number_input("fy  (ksc)", value=4000, min_value=2400, max_value=6000, key="fy_main")

        if shape == "Rectangular":
            c1, c2 = st.columns(2)
            b = c1.number_input("Width b (cm) [X]", value=40, min_value=20, key="b_main")
            h = c2.number_input("Depth h (cm) [Y]", value=60, min_value=20, key="h_main")
            layout = st.selectbox("Rebar Layout", ["4-Faces (Uniform)", "2-Faces (Top/Bottom)"], key="layout_main")
            if layout == "2-Faces (Top/Bottom)":
                n_bars = st.number_input("Total Bars (even)", 4, 40, 8, step=2, key="n_bars_2f")
                nx = ny = 0
            else:
                c3, c4 = st.columns(2)
                nx = c3.number_input("Bars on X-faces (nx)", 2, 20, 3, key="nx_main")
                ny = c4.number_input("Bars on Y-faces (ny)", 2, 20, 4, key="ny_main")
                n_bars = 2 * nx + 2 * (ny - 2) + 4  # corner bars shared
        else:
            b = h = st.number_input("Diameter D (cm)", value=50, min_value=30, key="D_main")
            layout   = "Circular"
            n_bars   = st.number_input("Total Bars (≥ 6)", 6, 60, 8, key="n_bars_circ")
            nx = ny  = 0

        db     = st.selectbox("Bar Size (mm)", [16, 20, 25, 28, 32], index=2, key="db_main")
        cover  = st.number_input("Clear Cover (cm)", value=4.0, min_value=2.5, key="cover_main")

    with st.expander("2. Loads & Frame Type", expanded=True):
        Pu    = st.number_input("Factored Axial Pu (ton)", value=150.0, min_value=0.0, key="Pu_main")
        c5, c6 = st.columns(2)
        Mux   = c5.number_input("Mux (ton-m)", value=15.0, key="Mux_main", help="Moment about X-axis (bending in h-direction)")
        Muy   = c6.number_input("Muy (ton-m)", value=10.0, key="Muy_main", help="Moment about Y-axis (bending in b-direction)")

        st.markdown("---")
        frame_type = st.radio("Frame Type", ["Non-Sway (Braced)", "Sway (Unbraced)"], horizontal=True, key="frame_type_main")

        if frame_type == "Non-Sway (Braced)":
            cx1, cx2, cx3 = st.columns(3)
            Lu_x  = cx1.number_input("Lu X (m)", value=4.0, step=0.5, key="Lu_x_ns")
            K_x   = cx2.number_input("K  X", value=1.0, step=0.1, min_value=0.5, key="K_x_main")
            Cm_x  = cx3.number_input("Cm X", value=1.0, step=0.05, min_value=0.4, key="Cm_x_main")

            cy1, cy2, cy3 = st.columns(3)
            Lu_y  = cy1.number_input("Lu Y (m)", value=4.0, step=0.5, key="Lu_y_ns")
            K_y   = cy2.number_input("K  Y", value=1.0, step=0.1, min_value=0.5, key="K_y_main")
            Cm_y  = cy3.number_input("Cm Y", value=1.0, step=0.05, min_value=0.4, key="Cm_y_main")

            beta_d = st.slider("β_d (Sustained Load Ratio)", 0.0, 1.0, 0.6, key="beta_d_ns")
            delta_sx = delta_sy = 1.0

        else:  # Sway
            sway_method = st.radio("δs Method", ["Stability Index Q", "ΣPu & ΣPc", "Direct Input"], horizontal=True, key="sway_method")
            if sway_method == "Stability Index Q":
                Q_val     = st.number_input("Q", 0.0, 0.99, 0.05, step=0.01, key="Q_sway")
                delta_sx  = delta_sy = max(1.0, 1.0 / (1.0 - Q_val))
                st.info(f"δs = {delta_sx:.3f}")
            elif sway_method == "ΣPu & ΣPc":
                cs1, cs2 = st.columns(2)
                sum_Pu = cs1.number_input("ΣPu (ton)", 0.1, value=500.0, key="sum_Pu_sway")
                sum_Pc = cs2.number_input("ΣPc (ton)", 0.1, value=2000.0, key="sum_Pc_sway")
                limit  = 0.75 * sum_Pc
                if sum_Pu >= limit:
                    st.error("⚠️ ΣPu ≥ 0.75ΣPc → Frame unstable!")
                    delta_sx = delta_sy = 999.0
                else:
                    delta_sx = delta_sy = max(1.0, 1.0 / (1.0 - sum_Pu / limit))
                    st.info(f"δs = {delta_sx:.3f}")
            else:
                cs1, cs2 = st.columns(2)
                delta_sx = cs1.number_input("δs X", value=1.2, step=0.05, min_value=1.0, key="delta_sx_dir")
                delta_sy = cs2.number_input("δs Y", value=1.2, step=0.05, min_value=1.0, key="delta_sy_dir")

            # Sway frames: non-sway magnifier still required for non-sway component
            cx1, cx2, cx3 = st.columns(3)
            Lu_x  = cx1.number_input("Lu X (m)", value=4.0, step=0.5, key="Lu_x_sw")
            K_x   = cx2.number_input("K  X (NS)", value=0.5, step=0.05, min_value=0.1, key="K_x_sw")
            Cm_x  = cx3.number_input("Cm X", value=1.0, step=0.05, min_value=0.4, key="Cm_x_sw")
            cy1, cy2, cy3 = st.columns(3)
            Lu_y  = cy1.number_input("Lu Y (m)", value=4.0, step=0.5, key="Lu_y_sw")
            K_y   = cy2.number_input("K  Y (NS)", value=0.5, step=0.05, min_value=0.1, key="K_y_sw")
            Cm_y  = cy3.number_input("Cm Y", value=1.0, step=0.05, min_value=0.4, key="Cm_y_sw")
            beta_d = st.slider("β_d", 0.0, 1.0, 0.6, key="beta_d_sw")

    with st.expander("3. Shear, Torsion & Seismic", expanded=True):
        st.subheader("🛡️ Shear Design")
        cv5, cv6 = st.columns(2)
        vux_ton = cv5.number_input("Vux (ton)", value=5.0, step=1.0, key="vux_main")
        vuy_ton = cv6.number_input("Vuy (ton)", value=5.0, step=1.0, key="vuy_main")

        c7, c8 = st.columns(2)
        tie_dia  = c7.selectbox("Tie ⌀ (mm)", [6, 9, 12, 16], index=1, format_func=lambda x: f"RB{x}" if x < 10 else f"DB{x}", key="tie_dia_main")
        tie_legs = c8.number_input("Stirrup Legs", 2, 10, 2, key="tie_legs_main")

        st.markdown("---")
        tu_tonm    = st.number_input("Tu (ton-m)", value=0.0, step=0.5, key="tu_main")
        is_seismic = st.toggle("Seismic Detailing (SMF)", value=True, key="seismic_main")


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE & CALCULATION
# ══════════════════════════════════════════════════════════════════════════════
engine  = RCColumn(shape, layout, b, h, fc, fy, db, n_bars, nx, ny, cover)
df_x, phi_pn_max = engine.solve_pm(axis='X')
df_y, _          = engine.solve_pm(axis='Y')

# ── Minimum eccentricity moments  ACI 318-19 §6.6.4.5.4 ──────────────────────
e_min_x   = Pu * (0.015 + 0.03 * h / 100.0)
e_min_y   = Pu * (0.015 + 0.03 * b / 100.0)
Mu_x_dsgn = max(Mux, e_min_x)
Mu_y_dsgn = max(Muy, e_min_y)

# ── Moment magnification ──────────────────────────────────────────────────────
kl_rx, Pcx, del_x, Ise_x, EIx = engine.slenderness_magnifier(
    Pu, K_x, Lu_x, 'X', Cm_x, beta_d)
kl_ry, Pcy, del_y, Ise_y, EIy = engine.slenderness_magnifier(
    Pu, K_y, Lu_y, 'Y', Cm_y, beta_d)

if frame_type == "Non-Sway (Braced)":
    Mcx = del_x * Mu_x_dsgn if kl_rx > 22.0 else Mu_x_dsgn
    Mcy = del_y * Mu_y_dsgn if kl_ry > 22.0 else Mu_y_dsgn
else:  # Sway: M2 = δs·M2s + δns·M2ns  (simplified: apply δs to total Mu)
    Mcx_sway = delta_sx * Mu_x_dsgn
    Mcy_sway = delta_sy * Mu_y_dsgn
    Mcx = del_x * Mcx_sway if kl_rx > 22.0 else Mcx_sway
    Mcy = del_y * Mcy_sway if kl_ry > 22.0 else Mcy_sway

# ── Shear / torsion ───────────────────────────────────────────────────────────
shear = shear_torsion_check(engine, Pu, vux_ton, vuy_ton,
                             tu_tonm, tie_dia, tie_legs, is_seismic, Lu_x)

# ── Biaxial interaction (PCA load-contour) ───────────────────────────────────
error_status  = None
is_safe       = False
demand_ratio  = 999.0
phi_Mnox = phi_Mnoy = 0.0
alpha = 1.5

if Pu > phi_pn_max:
    error_status = (f"Axial load Pu = {Pu:.1f} t exceeds section capacity "
                    f"φPn,max = {phi_pn_max:.1f} t")
else:
    try:
        fx_i = interp1d(df_x['phiPn'], df_x['phiMn'],
                        kind='linear', bounds_error=False, fill_value=0.0)
        fy_i = interp1d(df_y['phiPn'], df_y['phiMn'],
                        kind='linear', bounds_error=False, fill_value=0.0)
        phi_Mnox = float(fx_i(Pu))
        phi_Mnoy = float(fy_i(Pu))

        if phi_Mnox <= 0 or phi_Mnoy <= 0:
            error_status = "Pu is outside the valid range of the P-M interaction curve."
        else:
            alpha        = engine.get_alpha(Pu)
            demand_ratio = ((Mcx / phi_Mnox)**alpha +
                            (Mcy / phi_Mnoy)**alpha)
            is_safe      = demand_ratio <= 1.0
    except Exception as e:
        error_status = f"Interpolation error: {e}"

actual_space, min_req_space, space_ok = engine.check_clear_spacing(nx, ny)
rho_pct = engine.rho * 100.0
rho_ok  = 1.0 <= rho_pct <= 8.0

# ── Lap splice lengths ────────────────────────────────────────────────────────
l_splice_B, l_compression = lap_splice_lengths(fc, fy, engine.db_cm)


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL – RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with col_main:
    st.markdown("### 📋 Executive Design Summary")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Steel (ρ)",   f"{rho_pct:.2f} %",
              "✅ OK" if rho_ok else "❌ Fail",
              delta_color="normal" if rho_ok else "inverse")
    m2.metric("Clear Spacing", f"{actual_space:.2f} cm",
              "✅ OK" if space_ok else "⚠️ Tight",
              delta_color="normal" if space_ok else "inverse")
    m3.metric("Mcx (Magnified)", f"{Mcx:.1f} t-m",
              f"×{max(Mcx / Mu_x_dsgn, 1.0):.2f}", delta_color="off")
    m4.metric("Mcy (Magnified)", f"{Mcy:.1f} t-m",
              f"×{max(Mcy / Mu_y_dsgn, 1.0):.2f}", delta_color="off")
    shear_label = "SMF Tie Spacing" if is_seismic else "Shear Tie Spacing"
    s_fin = shear['s_design']
    s_ok  = shear['x_ok'] and shear['y_ok']
    m5.metric(shear_label, f"@ {s_fin:.1f} cm",
              "✅ OK" if s_ok else "❌ Fail",
              delta_color="normal" if s_ok else "inverse")

    st.markdown("---")

    # Alerts
    if shear['torsion_critical']:
        st.warning(f"🌪️ **Torsion Design Required:** Tu = {tu_tonm:.2f} t-m > "
                   f"Tth = {shear['Tth_tonm']:.3f} t-m. "
                   "Provide closed stirrups with 135° hooks + extra longitudinal bars.")
    if not space_ok:
        st.warning(f"⚠️ Bar spacing {actual_space:.2f} cm < minimum {min_req_space:.2f} cm "
                   "— Honeycombing risk. Reduce bar count or increase section width.")
    if not rho_ok:
        st.warning(f"⚠️ ρ = {rho_pct:.2f}% is {'below 1.0%' if rho_pct < 1.0 else 'above 8.0%'}")
    if del_x > 10 or del_y > 10:
        st.error("⚠️ Slenderness magnifier > 10 — Column is critically slender. "
                 "Increase section or reduce height.")

    if error_status:
        st.error(f"### ❌ CAPACITY EXCEEDED\n{error_status}")
    elif is_safe:
        st.success(f"### ✅ SAFE — Biaxial Demand Ratio = **{demand_ratio:.3f}** ≤ 1.0  (α = {alpha:.3f})")
    else:
        st.error(f"### ❌ UNSAFE — Biaxial Demand Ratio = **{demand_ratio:.3f}** > 1.0  (α = {alpha:.3f})")

    # ══════════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📥 Overview & 3D",
        "📊 P-M Interaction",
        "🧊 Section Detail",
        "🌪️ Shear & Seismic",
        "📝 Calc Report",
        "⚡ Quick Sizing",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 – 3D surface + 2D contour + P-Mx / P-My projections
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### 🌐 3D Biaxial Failure Surface (PCA Load-Contour)")

        if not error_status:
            try:
                mx_m, my_m, p_m = engine.generate_3d_surface(df_x, df_y, alpha)
                if mx_m.size:
                    fig3d = go.Figure()
                    fig3d.add_trace(go.Surface(
                        x=mx_m, y=my_m, z=p_m,
                        colorscale='Plasma', opacity=0.75,
                        showscale=False, name='Capacity Surface',
                        lighting=dict(ambient=0.6, diffuse=0.8,
                                      roughness=0.4, specular=0.5)))
                    mc = '#2ecc71' if is_safe else '#e74c3c'
                    fig3d.add_trace(go.Scatter3d(
                        x=[Mcx], y=[Mcy], z=[Pu],
                        mode='markers+text',
                        marker=dict(size=9, color=mc, symbol='diamond',
                                    line=dict(width=2, color='white')),
                        text=["Demand"], textposition="top center",
                        name='Design Demand'))
                    fig3d.add_trace(go.Scatter3d(
                        x=[Mcx, Mcx], y=[Mcy, Mcy], z=[0, Pu],
                        mode='lines', line=dict(color=mc, width=3, dash='dot'),
                        showlegend=False))
                    fig3d.update_layout(
                        scene=dict(
                            xaxis_title='Mx (t-m)',
                            yaxis_title='My (t-m)',
                            zaxis_title='P (ton)',
                            aspectmode='manual',
                            aspectratio=dict(x=1, y=1, z=1.2),
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
                        margin=dict(l=0, r=0, b=0, t=0), height=560)
                    st.plotly_chart(fig3d, use_container_width=True)
            except Exception as e:
                st.info(f"3D surface: {e}")
        else:
            st.error("Cannot draw 3D surface — Pu exceeds section capacity.")

        st.markdown("---")
        st.markdown(f"#### 🎯 2D PCA Contour at Pu = {Pu:.2f} ton")

        col_a, col_b, col_c = st.columns([1, 1, 1.5])
        col_a.metric("Demand Ratio", f"{demand_ratio:.3f}",
                     "SAFE" if is_safe else "UNSAFE",
                     delta_color="inverse")
        col_b.metric("α (PCA Exponent)", f"{alpha:.3f}")

        with col_c:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=min(demand_ratio, 2.0),
                title={'text': "Capacity Utilisation", 'font': {'size': 13}},
                gauge={'axis': {'range': [0, 1.5]},
                       'bar': {'color': '#2c3e50'},
                       'steps': [
                           {'range': [0, 0.8],  'color': 'rgba(46,204,113,0.3)'},
                           {'range': [0.8, 1.0], 'color': 'rgba(241,196,15,0.3)'},
                           {'range': [1.0, 1.5], 'color': 'rgba(231,76,60,0.3)'}],
                       'threshold': {'line': {'color': 'red', 'width': 4},
                                     'thickness': 0.75, 'value': 1.0}}))
            fig_g.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

        # 2D PCA contour
        if phi_Mnox > 0 and phi_Mnoy > 0:
            mx_c = np.linspace(0, phi_Mnox, 120)
            my_c = phi_Mnoy * np.maximum(0, 1 - (mx_c / phi_Mnox)**alpha)**(1.0 / alpha)
            mc = '#2ecc71' if is_safe else '#e74c3c'

            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(
                x=mx_c, y=my_c, mode='lines', name=f'Capacity (α={alpha:.2f})',
                line=dict(color='#8e44ad', width=3),
                fill='tozeroy', fillcolor='rgba(142,68,173,0.10)'))
            fig_c.add_trace(go.Scatter(
                x=[phi_Mnox, 0], y=[0, phi_Mnoy], mode='markers+text',
                name='Uniaxial Caps',
                marker=dict(color='#2c3e50', size=9, symbol='square'),
                text=[f'φMnox={phi_Mnox:.1f}', f'φMnoy={phi_Mnoy:.1f}'],
                textposition=['top right', 'top right']))
            fig_c.add_trace(go.Scatter(
                x=[Mcx], y=[Mcy], mode='markers+text', name='Demand',
                marker=dict(color=mc, size=14, symbol='cross',
                            line=dict(width=2, color='white')),
                text=["Design Point"], textposition="top right"))
            fig_c.add_shape(type="line", x0=0, y0=0, x1=Mcx, y1=Mcy,
                            line=dict(color=mc, width=2, dash='dashdot'))

            mx_rng = max(phi_Mnox, Mcx) * 1.2
            my_rng = max(phi_Mnoy, Mcy) * 1.2
            fig_c.update_layout(
                xaxis=dict(title='Magnified Mcx (ton-m)', range=[0, mx_rng],
                           showgrid=True, gridcolor='rgba(0,0,0,0.06)',
                           zeroline=True, zerolinewidth=2),
                yaxis=dict(title='Magnified Mcy (ton-m)', range=[0, my_rng],
                           showgrid=True, gridcolor='rgba(0,0,0,0.06)',
                           zeroline=True, zerolinewidth=2),
                plot_bgcolor='white', paper_bgcolor='white', height=450,
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.85)',
                            bordercolor='gray', borderwidth=1),
                margin=dict(l=40, r=40, t=20, b=40))
            st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📈 Uniaxial P-M Projections")
        col_pmx, col_pmy = st.columns(2)

        def pm_side_chart(df, Mc, label, color):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['phiMn'], y=df['phiPn'], mode='lines',
                line=dict(color=color, width=2.5),
                fill='tozerox', fillcolor=f'rgba({",".join(str(c) for c in [41,128,185])},0.12)',
                name='Capacity'))
            fig.add_trace(go.Scatter(
                x=[Mc], y=[Pu], mode='markers',
                marker=dict(color='#e74c3c', size=12, symbol='cross',
                            line=dict(width=2, color='white')),
                name='Demand'))
            fig.add_shape(type="line", x0=0, y0=Pu, x1=Mc, y1=Pu,
                          line=dict(color="#e74c3c", width=1, dash="dot"))
            fig.add_shape(type="line", x0=Mc, y0=df['phiPn'].min() * 1.05, x1=Mc, y1=Pu,
                          line=dict(color="#e74c3c", width=1, dash="dot"))
            p_lo2 = df['phiPn'].min() * 1.1 if df['phiPn'].min() < 0 else -10
            p_hi2 = df['phiPn'].max() * 1.1
            fig.update_layout(
                title=dict(text=f"<b>{label}</b>", font=dict(size=13)),
                xaxis=dict(title=f'{label.split()[0]} (ton-m)', rangemode='tozero',
                           showgrid=True, gridcolor='rgba(0,0,0,0.05)',
                           zeroline=True, zerolinewidth=2),
                yaxis=dict(title='Axial φPn (ton)', range=[p_lo2, p_hi2],
                           showgrid=True, gridcolor='rgba(0,0,0,0.05)',
                           zeroline=True, zerolinewidth=2),
                plot_bgcolor='white', paper_bgcolor='white',
                height=380, showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20))
            return fig

        with col_pmx:
            st.plotly_chart(pm_side_chart(df_x, Mcx, "P-Mx (Major Axis)", "#2980b9"),
                            use_container_width=True)
        with col_pmy:
            st.plotly_chart(pm_side_chart(df_y, Mcy, "P-My (Minor Axis)", "#27ae60"),
                            use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 – Full P-M diagram with ACI limits
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 📊 P-M Interaction Diagram")
        show_bounds = st.toggle("Show ACI ρ-limits (1% & 8%)", value=True, key="show_bounds_t2")
        show_keys   = st.toggle("Label Key Points", value=True, key="show_keys_t2")

        fig_pm = go.Figure()

        if show_bounds:
            def make_ref_df(target_rho):
                try:
                    tAs   = target_rho * engine.Ag
                    ref_n = max(4, round(tAs / engine.as_bar))
                    if shape == "Rectangular":
                        ref_nx = max(2, round(math.sqrt(ref_n * b / h)))
                        ref_ny = max(2, round((ref_n - 2 * ref_nx) / 2) + 2)
                        re = RCColumn(shape, "4-Faces (Uniform)",
                                      b, h, fc, fy, 20, 0, ref_nx, ref_ny, cover)
                    else:
                        re = RCColumn(shape, "Circular",
                                      b, h, fc, fy, 20, max(6, ref_n), 0, 0, cover)
                    rd, _ = re.solve_pm(axis='X')
                    return rd
                except Exception:
                    return pd.DataFrame()

            with st.spinner("Computing boundary curves…"):
                df_1 = make_ref_df(0.01)
                df_8 = make_ref_df(0.08)

            if not df_1.empty and not df_8.empty:
                xp = list(df_8['phiMn']) + list(df_1['phiMn'])[::-1]
                yp = list(df_8['phiPn']) + list(df_1['phiPn'])[::-1]
                xp.append(xp[0]); yp.append(yp[0])
                fig_pm.add_trace(go.Scatter(
                    x=xp, y=yp, fill='toself',
                    fillcolor='rgba(46,204,113,0.10)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Optimal Zone 1–8%', hoverinfo='skip'))
                for df_lim, nm, clr in [
                        (df_1, 'Min (ρ=1%)',  'rgba(149,165,166,0.9)'),
                        (df_8, 'Max (ρ=8%)',  'rgba(231,76,60,0.6)')]:
                    fig_pm.add_trace(go.Scatter(
                        x=df_lim['phiMn'], y=df_lim['phiPn'],
                        name=nm, mode='lines',
                        line=dict(color=clr, width=1.5), hoverinfo='skip'))

        for dfpm, nm, clr in [(df_x, 'X-Axis', '#2980b9'),
                               (df_y, 'Y-Axis', '#27ae60')]:
            fig_pm.add_trace(go.Scatter(
                x=dfpm['phiMn'], y=dfpm['phiPn'],
                name=nm, mode='lines', line=dict(color=clr, width=3.5),
                hovertemplate=f"<b>{nm}</b><br>φMn: %{{x:.2f}} t-m<br>φPn: %{{y:.2f}} ton<extra></extra>"))

        if show_keys and not df_x.empty:
            bal_idx = df_x['phiMn'].idxmax()
            anns = [
                dict(x=0, y=df_x['phiPn'].max(), text="Pure Compression",
                     showarrow=True, arrowhead=2, ax=60, ay=0,
                     font=dict(size=10, color='#7f8c8d')),
                dict(x=df_x.loc[bal_idx,'phiMn'], y=df_x.loc[bal_idx,'phiPn'],
                     text="Balance Point",
                     showarrow=True, arrowhead=2, ax=40, ay=-35,
                     font=dict(size=10, color='#7f8c8d')),
                dict(x=0, y=df_x['phiPn'].min(), text="Pure Tension",
                     showarrow=True, arrowhead=2, ax=60, ay=0,
                     font=dict(size=10, color='#7f8c8d')),
            ]
            fig_pm.update_layout(annotations=anns)

        fig_pm.add_trace(go.Scatter(
            x=[Mcx, Mcy], y=[Pu, Pu], mode='markers',
            marker=dict(color=['#e74c3c', '#e67e22'], size=14,
                        symbol='cross', line=dict(width=2, color='white')),
            name='Demands (Mcx, Mcy)',
            hovertemplate="<b>Demand</b><br>Mc: %{x:.2f} t-m<br>Pu: %{y:.2f} ton<extra></extra>"))
        for Mc, clr in [(Mcx, '#e74c3c'), (Mcy, '#e67e22')]:
            fig_pm.add_shape(type="line", x0=0, y0=Pu, x1=Mc, y1=Pu,
                             line=dict(color=clr, width=1, dash='dot'))
            fig_pm.add_shape(type="line", x0=Mc, y0=0, x1=Mc, y1=Pu,
                             line=dict(color=clr, width=1, dash='dot'))

        all_pn = pd.concat([df_x['phiPn'], df_y['phiPn']])
        y_lo = all_pn.min() * 1.1 if all_pn.min() < 0 else -10
        fig_pm.update_layout(
            xaxis=dict(title='Design Moment φMn (ton-m)', rangemode='tozero',
                       showgrid=True, gridcolor='rgba(0,0,0,0.05)',
                       zeroline=True, zerolinewidth=2),
            yaxis=dict(title='Design Axial φPn (ton)',
                       range=[y_lo, all_pn.max() * 1.1],
                       showgrid=True, gridcolor='rgba(0,0,0,0.05)',
                       zeroline=True, zerolinewidth=2),
            plot_bgcolor='white', paper_bgcolor='white', height=660,
            hovermode='closest',
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='center', x=0.5,
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
            margin=dict(l=40, r=40, t=60, b=40))
        st.plotly_chart(fig_pm, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 – Section detail drawing
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🧊 Cross-Section & BIM Cage")
        bx = [bar['x'] for bar in engine.bars]
        by = [bar['y'] for bar in engine.bars]
        cv2 = cover

        sub1, sub2 = st.tabs(["2D Section", "3D Cage"])
        with sub1:
            show_dim  = st.toggle("Show Dimensions", value=True, key="show_dim_t3")
            show_lid  = st.toggle("Bar Labels", value=True, key="show_lid_t3")
            show_spec = st.toggle("Material Specs", value=True, key="show_spec_t3")

            dark = '#020617'; blue = '#38bdf8'; red = '#ef4444'; gold = '#fbbf24'
            fig2d = go.Figure()

            if shape == "Rectangular":
                xc = [-b/2, b/2, b/2, -b/2, -b/2]
                yc = [-h/2, -h/2, h/2, h/2, -h/2]
                xt = [-(b/2-cv2), (b/2-cv2), (b/2-cv2), -(b/2-cv2), -(b/2-cv2)]
                yt = [-(h/2-cv2), -(h/2-cv2), (h/2-cv2), (h/2-cv2), -(h/2-cv2)]
            else:
                th = np.linspace(0, 2*math.pi, 120)
                xc, yc = (b/2)*np.cos(th), (b/2)*np.sin(th)
                xt, yt = (b/2-cv2)*np.cos(th), (b/2-cv2)*np.sin(th)

            fig2d.add_trace(go.Scatter(x=xc, y=yc, mode='lines', name='Concrete',
                                       line=dict(color=blue, width=3),
                                       fill='toself', fillcolor='rgba(56,189,248,0.1)'))
            fig2d.add_trace(go.Scatter(x=xt, y=yt, mode='lines', name='Ties',
                                       line=dict(color=gold, width=1.5, dash='dash')))
            fig2d.add_trace(go.Scatter(
                x=bx, y=by,
                mode='markers+text' if show_lid else 'markers',
                marker=dict(color=dark, size=13, line=dict(color=red, width=2.5)),
                text=[str(i+1) for i in range(len(bx))],
                textposition='top center', textfont=dict(color='white', size=9),
                name='Rebars'))

            lim = max(b, h) / 2 + max(b, h) * 0.35
            if show_dim and shape == "Rectangular":
                yd = -h/2 - max(b, h)*0.18
                fig2d.add_shape(type="line", x0=-b/2, y0=yd, x1=b/2, y1=yd,
                                line=dict(color='#94a3b8', width=1.5))
                fig2d.add_annotation(x=0, y=yd, text=f"b = {b} cm", showarrow=False,
                                     yshift=12, font=dict(color='white', size=12))
                xd = -b/2 - max(b, h)*0.18
                fig2d.add_shape(type="line", x0=xd, y0=-h/2, x1=xd, y1=h/2,
                                line=dict(color='#94a3b8', width=1.5))
                fig2d.add_annotation(x=xd, y=0, text=f"h = {h} cm", showarrow=False,
                                     xshift=-15, textangle=-90, font=dict(color='white', size=12))
            elif show_dim and shape == "Circular":
                yd = -b/2 - b*0.2
                fig2d.add_shape(type="line", x0=-b/2, y0=yd, x1=b/2, y1=yd,
                                line=dict(color='#94a3b8', width=1.5))
                fig2d.add_annotation(x=0, y=yd, text=f"D = {b} cm", showarrow=False,
                                     yshift=12, font=dict(color='white', size=12))

            if show_spec:
                fig2d.add_annotation(
                    xref='paper', yref='paper', x=0.98, y=0.02,
                    text=(f"<b>SPECS</b><br>f'c = {fc} ksc<br>fy = {fy} ksc<br>"
                          f"Ast = {engine.total_as:.2f} cm²<br>ρ = {rho_pct:.2f}%"),
                    showarrow=False, align='right',
                    bgcolor='rgba(15,23,42,0.85)', bordercolor='#334155',
                    borderpad=10, font=dict(color=blue, size=11))

            fig2d.update_layout(
                plot_bgcolor=dark, paper_bgcolor=dark,
                xaxis=dict(visible=False, range=[-lim, lim]),
                yaxis=dict(visible=False, range=[-lim, lim],
                           scaleanchor='x', scaleratio=1),
                height=650, margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(font=dict(color='white'), orientation='h',
                            y=1.05, x=0.5, xanchor='center'))
            st.plotly_chart(fig2d, use_container_width=True)

        with sub2:
            L_col = max(b, h) * 4
            fig3d_cage = go.Figure()
            for i, (x, y) in enumerate(zip(bx, by)):
                fig3d_cage.add_trace(go.Scatter3d(
                    x=[x, x], y=[y, y], z=[0, L_col], mode='lines',
                    line=dict(color=red, width=5), name=f'Bar {i+1}'))
            n_ties_3d = max(5, int(L_col / shear['s_design']))
            for z in np.linspace(shear['s_design'], L_col - shear['s_design'], n_ties_3d):
                fig3d_cage.add_trace(go.Scatter3d(
                    x=list(xt) if shape == "Rectangular" else list(xc),
                    y=list(yt) if shape == "Rectangular" else list(yc),
                    z=[z] * (len(xt) if shape == "Rectangular" else len(xc)),
                    mode='lines', line=dict(color=gold, width=3), showlegend=False))
            fig3d_cage.update_layout(
                scene=dict(aspectmode='data',
                           xaxis_title='X (cm)', yaxis_title='Y (cm)',
                           zaxis_title='Height (cm)'),
                margin=dict(l=0, r=0, t=0, b=0), height=580, paper_bgcolor=dark)
            st.plotly_chart(fig3d_cage, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 – Shear & Seismic
    # ─────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🛡️ Shear, Torsion & Seismic Detailing (ACI 318-19, MKS)")
        s = shear

        # Dashboard metrics
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("φVnx (X-shear cap.)", f"{s['phiVnx']:.2f} ton",
                    "✅ OK" if s['x_ok'] else "❌ Fail",
                    delta_color="normal" if s['x_ok'] else "inverse")
        r1c2.metric("φVny (Y-shear cap.)", f"{s['phiVny']:.2f} ton",
                    "✅ OK" if s['y_ok'] else "❌ Fail",
                    delta_color="normal" if s['y_ok'] else "inverse")
        r1c3.metric("Tu",  f"{tu_tonm:.2f} ton-m",
                    "Critical" if s['torsion_critical'] else "Ignorable",
                    delta_color="inverse" if s['torsion_critical'] else "off")
        r1c4.metric("Tth", f"{s['Tth_tonm']:.3f} ton-m")

        st.markdown("---")

        with st.expander("📝 Full Shear Calculation (Step-by-Step)", expanded=True):
            st.markdown(f"""
#### Unit System Reminder
All calculations use **MKS**: stress in ksc, force in kgf (→ ton), length in cm.
MKS coefficient 0.53 ≡ SI coefficient 0.17 (√(10.2 ksc/MPa) ≈ 3.19; 0.53/3.19 ≈ 0.17).

---

#### 1. Effective Depths & Widths
* **dx** (depth for X-shear, bending in h-direction) = h − d' = {engine.h} − {engine.d_prime:.2f} = **{s['dx']:.2f} cm**
* **dy** (depth for Y-shear, bending in b-direction) = b − d' = {engine.b} − {engine.d_prime:.2f} = **{s['dy']:.2f} cm**
* **bwx** (web width for X-shear) = {s['bwx']} cm
* **bwy** (web width for Y-shear) = {s['bwy']} cm

#### 2. Concrete Shear Capacity (ACI 318-19 Table 22.5.5.1)
$$V_c = 0.53\\sqrt{{f'_c}}\\,b_w\\,d\\left(1 + \\frac{{N_u}}{{140 A_g}}\\right)$$
* **Vcx** = 0.53 × √{fc} × {s['bwx']} × {s['dx']:.2f} × (1 + {Pu*1000:.0f}/{140*engine.Ag:.0f}) = **{s['Vcx_ton']:.3f} ton**
* **Vcy** = 0.53 × √{fc} × {s['bwy']} × {s['dy']:.2f} × (1 + {Pu*1000:.0f}/{140*engine.Ag:.0f}) = **{s['Vcy_ton']:.3f} ton**

#### 3. Transverse Steel Provided
* Tie: {f"RB{tie_dia}" if tie_dia<10 else f"DB{tie_dia}"}, legs = {tie_legs}
* At = π×{s['d_tie']:.2f}²/4 = {s['At']:.3f} cm²
* Av = {tie_legs} × {s['At']:.3f} = **{s['Av']:.3f} cm²**

#### 4. Required Stirrup Spacing from Shear Demand
$$V_s = \\frac{{A_v f_y d}}{{s}} \\implies s = \\frac{{A_v f_y d}}{{V_s,req}}$$
* **X-demand Vs,req** = Vux/φ − Vcx = {vux_ton:.2f}/{PHI_SHEAR} − {s['Vcx_ton']:.3f} = {max(0,vux_ton/PHI_SHEAR - s['Vcx_ton']):.3f} ton
  → sx,shear = {s['Av']:.3f}×{fy}×{s['dx']:.2f} / ({max(1e-9,max(0,vux_ton/PHI_SHEAR-s['Vcx_ton']))*1000:.0f}) = **{s['sx_shear']:.1f} cm**
* **Y-demand Vs,req** = Vuy/φ − Vcy = {vuy_ton:.2f}/{PHI_SHEAR} − {s['Vcy_ton']:.3f} = {max(0,vuy_ton/PHI_SHEAR - s['Vcy_ton']):.3f} ton
  → sy,shear = **{s['sy_shear']:.1f} cm**

#### 5. Spacing Limits
| Limit | Value |
|---|---|
| Code max (d/2, 60 cm) X | {s['s_max_x']:.1f} cm |
| Code max (d/2, 60 cm) Y | {s['s_max_y']:.1f} cm |
| Seismic SMF limit | {s['s_seismic']:.1f} cm |
| **Governing design spacing** | **{s['s_design']:.1f} cm** |

#### 6. Provided Shear Capacity at s = {s['s_design']:.1f} cm
$$\\phi V_n = \\phi(V_c + V_s) = {PHI_SHEAR}\\left(V_c + \\frac{{A_v f_y d}}{{s}}\\right)$$
* **φVnx** = {PHI_SHEAR}×({s['Vcx_ton']:.3f} + {s['Vsx_prov_ton']:.3f}) = **{s['phiVnx']:.3f} ton** {'✅ ≥' if s['x_ok'] else '❌ <'} Vux = {vux_ton:.2f} ton
* **φVny** = {PHI_SHEAR}×({s['Vcy_ton']:.3f} + {s['Vsy_prov_ton']:.3f}) = **{s['phiVny']:.3f} ton** {'✅ ≥' if s['y_ok'] else '❌ <'} Vuy = {vuy_ton:.2f} ton

#### 7. Torsion Threshold Check (ACI 318-19 §22.7.4.1)
$$T_{{th}} = \\phi\\,0.026\\sqrt{{f'_c}}\\frac{{A_{{cp}}^2}}{{p_{{cp}}}}$$
* Acp = {s['Acp']:.2f} cm²,  pcp = {s['pcp']:.2f} cm
* Tth = {PHI_SHEAR}×0.026×√{fc}×{s['Acp']:.2f}²/{s['pcp']:.2f} / 100000 = **{s['Tth_tonm']:.4f} ton-m**
* Tu = {tu_tonm:.2f} ton-m → {"**Critical — provide torsional reinforcement!**" if s['torsion_critical'] else "Negligible."}

#### 8. Lap Splice Lengths (ACI 318-19 §25.5)
* Class B Tension Splice = **{l_splice_B:.0f} cm**
* Compression Splice     = **{l_compression:.0f} cm**
* Selected: {"Class B Tension (SMF requirement)" if is_seismic else "Compression splice (Ordinary frame)"}
  → **{l_splice_B if is_seismic else l_compression:.0f} cm**
""")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 – Full calculation report
    # ─────────────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown("### 📝 Detailed Calculation Report")

        with st.expander("1. Section & Material Properties", expanded=False):
            st.markdown("**Geometry**")
            if shape == "Rectangular":
                st.latex(rf"A_g = {b} \times {h} = {engine.Ag:.2f}\text{{ cm}}^2")
                st.latex(rf"I_{{gx}} = \frac{{{b} \times {h}^3}}{{12}} = {engine.Igx:,.2f}\text{{ cm}}^4")
                st.latex(rf"I_{{gy}} = \frac{{{h} \times {b}^3}}{{12}} = {engine.Igy:,.2f}\text{{ cm}}^4")
            else:
                st.latex(rf"A_g = \frac{{\pi {b}^2}}{{4}} = {engine.Ag:.2f}\text{{ cm}}^2")
                st.latex(rf"I_g = \frac{{\pi {b}^4}}{{64}} = {engine.Igx:,.2f}\text{{ cm}}^4")
            st.markdown("**Materials**")
            st.latex(rf"E_c = 15100\sqrt{{{fc}}} = {engine.Ec:,.0f}\text{{ ksc}}")
            st.latex(rf"\beta_1 = {engine.beta1:.3f}")

        with st.expander("2. Minimum Eccentricity Moments", expanded=False):
            st.latex(rf"e_{{min,x}} = P_u(0.015+0.03h/100) = {e_min_x:.3f}\text{{ t-m}}")
            st.latex(rf"M_{{ux,dsgn}} = \max({Mux:.2f},{e_min_x:.3f}) = {Mu_x_dsgn:.2f}\text{{ t-m}}")
            st.latex(rf"e_{{min,y}} = P_u(0.015+0.03b/100) = {e_min_y:.3f}\text{{ t-m}}")
            st.latex(rf"M_{{uy,dsgn}} = \max({Muy:.2f},{e_min_y:.3f}) = {Mu_y_dsgn:.2f}\text{{ t-m}}")

        with st.expander("3. Slenderness — X Axis", expanded=False):
            if frame_type == "Non-Sway (Braced)":
                st.latex(rf"kl/r_x = \frac{{{K_x}\times{Lu_x}\times100}}{{{engine.rx:.2f}}} = {kl_rx:.2f}")
                st.latex(rf"EI_x = \frac{{0.2E_cI_{{gx}}+E_sI_{{se,x}}}}{{1+\beta_d}} = {EIx:,.0f}\text{{ ksc·cm}}^2")
                st.latex(rf"P_{{cx}} = \frac{{\pi^2 EI_x}}{{(K_xL_u)^2}} = {Pcx:.2f}\text{{ ton}}")
                if kl_rx > 22:
                    st.latex(rf"\delta_x = \frac{{{Cm_x}}}{{1-P_u/(0.75P_{{cx}})}} = {del_x:.3f}")
                else:
                    st.write(f"kl/r = {kl_rx:.2f} ≤ 22 → slenderness ignored, δx = 1.0")
            else:
                st.write(f"Sway frame: Mcx = δsx × Mu,x = {delta_sx:.3f} × {Mu_x_dsgn:.2f} = {Mcx:.2f} t-m")

        with st.expander("4. Slenderness — Y Axis", expanded=False):
            if frame_type == "Non-Sway (Braced)":
                st.latex(rf"kl/r_y = \frac{{{K_y}\times{Lu_y}\times100}}{{{engine.ry:.2f}}} = {kl_ry:.2f}")
                st.latex(rf"EI_y = {EIy:,.0f}\text{{ ksc·cm}}^2")
                st.latex(rf"P_{{cy}} = {Pcy:.2f}\text{{ ton}}")
                if kl_ry > 22:
                    st.latex(rf"\delta_y = {del_y:.3f}")
                else:
                    st.write(f"kl/r = {kl_ry:.2f} ≤ 22 → slenderness ignored, δy = 1.0")
            else:
                st.write(f"Sway frame: Mcy = δsy × Mu,y = {delta_sy:.3f} × {Mu_y_dsgn:.2f} = {Mcy:.2f} t-m")

        with st.expander("5. Biaxial Bending Check (PCA)", expanded=True):
            st.markdown("**PCA Load-Contour Method** (ACI 318 Commentary)")
            st.latex(
                rf"\left(\frac{{M_{{cx}}}}{{\phi M_{{nox}}}}\right)^{{\alpha}} + "
                rf"\left(\frac{{M_{{cy}}}}{{\phi M_{{noy}}}}\right)^{{\alpha}} \le 1.0")
            if phi_Mnox > 0:
                st.latex(rf"\phi M_{{nox}} = {phi_Mnox:.2f}\text{{ t-m}},\quad"
                         rf"\phi M_{{noy}} = {phi_Mnoy:.2f}\text{{ t-m}},\quad\alpha = {alpha:.3f}")
                st.latex(
                    rf"\text{{Ratio}} = \left(\frac{{{Mcx:.2f}}}{{{phi_Mnox:.2f}}}\right)^{{{alpha:.3f}}} + "
                    rf"\left(\frac{{{Mcy:.2f}}}{{{phi_Mnoy:.2f}}}\right)^{{{alpha:.3f}}} = {demand_ratio:.4f}")
                st.success("✅ SAFE" if is_safe else "") if is_safe else st.error("❌ UNSAFE")
            else:
                st.error("Unable to evaluate — Pu out of range.")

        with st.expander("6. Reinforcement Detailing Summary", expanded=False):
            st.markdown(f"""
| Parameter | Value | Limit | Status |
|---|---|---|---|
| ρ | {rho_pct:.2f}% | 1–8% | {"✅" if rho_ok else "❌"} |
| Clear Spacing | {actual_space:.2f} cm | ≥ {min_req_space:.2f} cm | {"✅" if space_ok else "❌"} |
| Shear Spacing | {shear['s_design']:.1f} cm | ≤ {shear['s_max_x']:.1f} cm | {"✅" if shear['s_design'] <= shear['s_max_x'] else "❌"} |
| φVnx | {shear['phiVnx']:.2f} ton | ≥ {vux_ton:.2f} ton | {"✅" if shear['x_ok'] else "❌"} |
| φVny | {shear['phiVny']:.2f} ton | ≥ {vuy_ton:.2f} ton | {"✅" if shear['y_ok'] else "❌"} |
| Lap Splice (Tension) | {l_splice_B:.0f} cm | — | — |
| Lap Splice (Compression) | {l_compression:.0f} cm | — | — |
""")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 6 – Quick Sizing
    # ─────────────────────────────────────────────────────────────────────────
    with tab6:
        st.markdown("### ⚡ Quick Sizing Tool")
        st.markdown("Estimate minimum section dimensions from slenderness limits.")

        qs1, qs2, qs3 = st.columns(3)
        q_L       = qs1.number_input("Unbraced Length (m)", 1.0, 30.0, 4.0, 0.5, key="q_L_qs")
        q_K       = qs2.number_input("K factor", 0.5, 2.0, 1.0, 0.1, key="q_K_qs")
        q_klr_lim = qs3.slider("Target kl/r limit", 10, 50, 22, key="q_klr_qs")

        with st.expander("Material Settings", expanded=False):
            qm1, qm2, qm3 = st.columns(3)
            q_fc  = qm1.number_input("f'c (ksc)", 140, 700, 280, key="q_fc_qs")
            q_fy  = qm2.number_input("fy  (ksc)", 2400, 6000, 4000, key="q_fy_qs")
            q_rho = qm3.slider("Target ρ (%)", 1.0, 6.0, 2.0, 0.5, key="q_rho_qs") / 100.0

        KL_cm = q_K * q_L * 100.0
        min_h = KL_cm / (0.3 * q_klr_lim)
        min_D = KL_cm / (0.25 * q_klr_lim)
        sug_h = math.ceil(min_h / 5.0) * 5
        sug_D = math.ceil(min_D / 5.0) * 5

        def quick_cap(dim, shape_t):
            Ag_q   = dim**2 if shape_t == "Rect" else math.pi * dim**2 / 4
            Ast_q  = Ag_q * q_rho
            phi_q  = PHI_COMP_T if shape_t == "Rect" else PHI_COMP_S
            fac_q  = 0.80       if shape_t == "Rect" else 0.85
            Po_q   = (0.85 * q_fc * (Ag_q - Ast_q) + q_fy * Ast_q) / 1_000.0
            r_q    = 0.3 * dim  if shape_t == "Rect" else 0.25 * dim
            klr_q  = KL_cm / r_q
            penalty = max(0.1, 1.0 - 0.008 * max(0, klr_q - q_klr_lim))
            return phi_q * fac_q * Po_q * penalty, klr_q

        cap_r, klr_r = quick_cap(sug_h, "Rect")
        cap_c, klr_c = quick_cap(sug_D, "Circ")

        qr1, qr2 = st.columns(2)
        with qr1:
            st.markdown(f"""
<div style="background:#1e293b;padding:20px;border-radius:10px;border-top:4px solid #38bdf8;">
<p style="color:#94a3b8;margin:0;font-size:12px;">RECTANGULAR (Tied)</p>
<h2 style="color:white;margin:8px 0;">{sug_h} × {sug_h} cm</h2>
<p style="color:#94a3b8;font-size:13px;">kl/r = {klr_r:.1f} &nbsp;|&nbsp; Est. φPn,max ≈</p>
<h1 style="color:#38bdf8;margin:0;">{cap_r:,.1f} <span style="font-size:16px">ton</span></h1>
</div>""", unsafe_allow_html=True)
        with qr2:
            st.markdown(f"""
<div style="background:#1e293b;padding:20px;border-radius:10px;border-top:4px solid #4ade80;">
<p style="color:#94a3b8;margin:0;font-size:12px;">CIRCULAR (Spiral)</p>
<h2 style="color:white;margin:8px 0;">Ø {sug_D} cm</h2>
<p style="color:#94a3b8;font-size:13px;">kl/r = {klr_c:.1f} &nbsp;|&nbsp; Est. φPn,max ≈</p>
<h1 style="color:#4ade80;margin:0;">{cap_c:,.1f} <span style="font-size:16px">ton</span></h1>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Sensitivity Table (Rectangular)")
        rows = []
        for s_dim in [sug_h - 10, sug_h - 5, sug_h, sug_h + 5, sug_h + 10]:
            if s_dim <= 0: continue
            cap_s, klr_s = quick_cap(s_dim, "Rect")
            rows.append({
                "Size (cm)": f"{s_dim}×{s_dim}",
                "kl/r": round(klr_s, 1),
                "Status": "🟢 Short" if klr_s <= q_klr_lim else "🔴 Slender",
                "Est. φPn,max (ton)": round(cap_s, 1),
            })
        st.table(rows)

        with st.expander("Step-by-Step Derivation", expanded=False):
            st.latex(rf"KL = {q_K} \times {q_L} \times 100 = {KL_cm:.0f}\text{{ cm}}")
            st.markdown("**Rectangular** — r ≈ 0.3h:")
            st.latex(rf"h_{{min}} = \frac{{KL}}{{0.3 \times {q_klr_lim}}} = {min_h:.2f} \rightarrow {sug_h}\text{{ cm}}")
            st.markdown("**Circular** — r ≈ 0.25D:")
            st.latex(rf"D_{{min}} = \frac{{KL}}{{0.25 \times {q_klr_lim}}} = {min_D:.2f} \rightarrow {sug_D}\text{{ cm}}")
