import numpy as np

#Values being used
Th_i = 80 #Temp of hot water in 
Tc_i = 20 #Temp of cold water in
mdot_h = 1.5 #Mass flow rate of hot water
mdot_c = 1.0 #Mass flow rate of cold water
cp = 4180 #specific heat capacity of water
U = 850 #Overall heat transfer coefficient
A = 12 #Heat transfer area

#Function to calculate values for 1-shell 2-tube pass
def effectiveness_ntu(Th_i,Tc_i,mdot_h,mdot_c,cp,U,A):
    C_h = mdot_h * cp
    C_c = mdot_c * cp
    C_min = min(C_h, C_c)
    C_max = max(C_h, C_c)
    C_r = C_min /  C_max
    NTU = (U * A) / C_min
    eff = 2 / (1 + C_r + np.sqrt(1 + C_r**2) * (1 + np.exp(-NTU * np.sqrt(1 + C_r**2))) / (1 - np.exp(-NTU * np.sqrt(1 + C_r**2))))
    Q_max = C_min * (Th_i - Tc_i)
    Q = eff * Q_max
    Th_o = Th_i - (Q/C_h)
    Tc_o = Tc_i + (Q/C_c)
    return {"C_h": C_h, "C_c": C_c, "C_r": C_r, "NTU": NTU, "eff": eff, "Q_max": Q_max, "Q": Q, "Th_o": Th_o, "Tc_o":  Tc_o, "C_min": C_min, "C_max": C_max}

#Call the effectiveness_ntu function to get our results
result = effectiveness_ntu(Th_i, Tc_i, mdot_h, mdot_c, cp, U, A)

#Physical checks for energy balance
Q_from_hot = result["C_h"] * (Th_i - result["Th_o"])
Q_from_cold = result["C_c"] * (result['Tc_o'] - Tc_i)

#Checking the effectiveness limits 0 to 1
assert result['eff'] >= 0, "Effectiveness should not be negative"
assert result['eff'] <= 1, "Effectiveness has a limit of 1"

#Checking if the energy balances calculated match the physical equation
assert np.isclose(result['Q'], Q_from_hot), "Energy balance failed on hot"
assert np.isclose(result['Q'], Q_from_cold), "Energy balance failed on cold"

#Check the second law of thermodynamics (hot should flow to cold not reverse unless work is applied)
assert result['Th_o'] >= Tc_i, "Hot outlet dropped below cold inlet, second law is not valid" #Hot outlet temp must be at least as high as the cold inlet (coldest thing in the system)
assert result['Tc_o'] <= Th_i, "Cold outlet is higher than the hot inlet, second law is not valid" #Cold must be as most as the highest temp in the system

#Prints results
print(f"NTU: {result['NTU']:.4f}")
print(f"Epsilon: {result['eff']:.4f}")
print(f"Q: {result['Q']:.4f}")
print(f"Th_o: {result['Th_o']:.4f}")
print(f"Tc_o: {result['Tc_o']:.4f}")
    
