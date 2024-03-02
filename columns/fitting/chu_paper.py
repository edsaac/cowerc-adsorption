import streamlit as st
import numpy as np
# from scipy.optimize import curve_fit
from scipy.special import i0
from scipy.integrate import quad

import matplotlib.pyplot as plt

def bohart_adams(t, k_BA, q_0):
    global C_0, Z, v, rho_P, epsilon
    alpha = k_BA * C_0 * (t - Z/v)
    beta = k_BA * rho_P * q_0 * Z * (1 - epsilon) / (epsilon * v) 

    return np.exp(alpha) / (np.exp(alpha) + np.exp(beta) + 1)

def J(x,y):

    def to_integrate(tau, yi):
        return np.exp(-yi - tau) * i0(2 * np.sqrt(yi * tau))  
    
    if isinstance(y, float):
        integration = quad(to_integrate, 0, x)

    else:
        integration = [quad(to_integrate, 0, x, args=(yi,)) for yi in y]
    
    integral_value = np.array([iarr[0] for iarr in integration])

    return 1 - integral_value
    

def thomas(t, k_T1, q_m, b):
    global C_0, Z, v, rho_P, epsilon

    r = 1 + b * C_0
    n = rho_P * q_m * k_T1 * Z * (1 - epsilon) / (v * epsilon)
    T = epsilon * (1/b + C_0) * (v*t/Z - 1) / (rho_P * q_m * (1 - epsilon))

    J1 = J(n/r, n*T)
    J2 = J(n, n*T/r)

    return J1 / (J1 + (1 - J2) * np.exp((1 - 1/r) * (n - n*T)))

## Constants
C_0 = 0.05 # mg/cm³
Z = 10 # cm
v = 2  # cm/s
epsilon = 0.4
rho_P = 1.5

with st.sidebar:
    
    "### Bohart-Adams"
    k_BA = st.number_input(R"$k_{BA}$", value=0.01, min_value=0.0, step=0.001, format="%.4f")
    q_0 = st.number_input(R"$q_0$", value=100.0, min_value=0.0, step=5.0)

    "****"
    "### Thomas"
    k_T1 = st.number_input(R"$k_{T1}$", value=0.01, min_value=0.0, step=0.001, format="%.4f")
    q_m = st.number_input(R"$q_m$", value=100.0, min_value=0.0, step=5.0)
    b = st.number_input(R"$b$", value=50.0, min_value=0.0, step=10.0)

time = np.linspace(10, 50_000, 200)
rel_conc_ba = bohart_adams(time, k_BA=k_BA, q_0=q_0)
rel_conc_thomas = thomas(time, k_T1=k_T1, q_m=q_m, b=b)

fig, ax = plt.subplots()
ax.plot(time, rel_conc_ba, label="Bohart-Adams")
ax.plot(time, rel_conc_thomas, label="Thomas")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Rel. Conc.")
ax.legend()
st.pyplot(fig)