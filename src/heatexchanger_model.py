import numpy as np

def effectiveness_ntu(Th_i,Tc_i,mdot_h,mdot_c,cp,U,A):
    C_h = mdot_h * cp
    C_c = mdot_c * cp
    C_min = min(C_h, C_c)
    C_max = max(C_h, C_c)
    C_r = C_min / C_max
    NTU = (U * A) / C_min
    eff = 2 / (1 + C_r + np.sqrt(1 + C_r**2) * (1 + np.exp(-NTU * np.sqrt(1 + C_r**2))) / (1 - np.exp(-NTU * np.sqrt(1 + C_r**2))))
    Q_max = C_min * (Th_i - Tc_i)
    Q = eff * Q_max
    Th_o = Th_i - (Q/C_h)
    Tc_o = Tc_i + (Q/C_c)
    return {"C_h": C_h, "C_c": C_c, "C_r": C_r, "NTU": NTU, "eff": eff, "Q_max": Q_max, "Q": Q, "Th_o": Th_o, "Tc_o": Tc_o, "C_min": C_min, "C_max": C_max}