import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class RCColumnEngine:
    def __init__(self, fc, fy, b, h, db_mm, n_bars, cover_cm):
        self.fc, self.fy, self.b, self.h = fc, fy, b, h
        self.es = 2.04e6 # ksc
        self.beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 280) / 70))
        
        # 1. คำนวณตำแหน่งเหล็กเสริมจริง (Assuming 4-face distribution)
        self.bars = []
        as_single = np.pi * (db_mm/20)**2 / 4
        
        # จัดเหล็กแบบง่าย: แบ่งเป็นแถวบนและแถวล่าง (สำหรับ 4 เส้น) 
        # หรือกระจายรอบนอก (สำหรับ > 4 เส้น) ในที่นี้ขอปรับเป็น General Layer Logic
        d_top = cover_cm + 0.9 + (db_mm/20)
        d_bot = h - d_top
        
        # กระจายเหล็กเป็น 2 กลุ่มหลัก (Top-Bottom) เพื่อความเข้าใจง่ายในขั้นนี้ 
        # แต่คำนวณแยกแรงต้านจริง
        self.bar_layers = [
            {'as': (n_bars/2) * as_single, 'd': d_top},
            {'as': (n_bars/2) * as_single, 'd': d_bot}
        ]

    def solve(self):
        points = []
        # วนลูปค่า c ตั้งแต่ h*2 (อัดเกือบเต็ม) จนถึง 0.1 (ดึงเกือบเต็ม)
        c_values = np.linspace(0.1, self.h * 1.5, 500)
        
        for c in c_values:
            # 1. Concrete Force & Moment
            a = min(self.beta1 * c, self.h)
            force_c = 0.85 * self.fc * a * self.b
            mom_c = force_c * (self.h/2 - a/2)
            
            # 2. Steel Forces & Moments
            force_s_total = 0
            mom_s_total = 0
            et = 0 # สำหรับเช็ค Phi
            
            for layer in self.bar_layers:
                eps = 0.003 * (c - layer['d']) / c
                fs = np.clip(eps * self.es, -self.fy, self.fy)
                f_s = layer['as'] * fs
                force_s_total += f_s
                mom_s_total += f_s * (self.h/2 - layer['d'])
                if layer['d'] == max(l['d'] for l in self.bar_layers):
                    et = abs(0.003 * (layer['d'] - c) / c)

            pn = (force_c + force_s_total) / 1000 # ton
            mn = (mom_c + mom_s_total) / 100000 # ton-m
            
            # 3. Phi Factor (ACI 318-19)
            ey = self.fy / self.es
            phi = 0.65 if et <= ey else 0.90 if et >= 0.005 else 0.65 + 0.25*(et - ey)/(0.005 - ey)
            
            points.append({'Pn': pn, 'Mn': mn, 'phiPn': phi * pn, 'phiMn': phi * mn})

        # เพิ่มจุด Pure Compression (Capped)
        ag = self.b * self.h
        ast = sum(l['as'] for l in self.bar_layers)
        po = (0.85 * self.fc * (ag - ast) + self.fy * ast) / 1000
        phi_pn_max = 0.65 * 0.80 * po
        
        df = pd.DataFrame(points).sort_values('Pn', ascending=False)
        return df, phi_pn_max

# UI ของ Streamlit (ส่วนที่เหลือคล้ายเดิมแต่เปลี่ยนการเรียกใช้)
