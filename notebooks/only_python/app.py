import streamlit as st
from itertools import cycle

from cowerc_adsorption import PhysicalParams, Simulation, ExperimentalBreakthroughData
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patheffects import Stroke
from math import pi
from pathlib import Path

import pandas as pd
from scipy.optimize import Bounds, curve_fit

st.set_page_config(layout="wide")

if "experimental" not in st.session_state:
    excel_file = Path("../../.data/Data and Column Properties.xlsx")
    sheet_name = "WW All Contaminants 0.18-min"

    # Read Emma's excel report
    xls_setup = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=18, usecols="A:B", index_col=0)
    xls_influent = pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=11, nrows=4, usecols="D:J")
    xls_btc = pd.read_excel(excel_file, sheet_name=sheet_name, header=19, usecols="A:Q")
    setup = xls_setup.to_dict()["Value"]

    porosity = setup["Bed Voidage (ε)"]
    lenght = setup["Bed Length (cm)"]

    area = 0.25 * pi * setup["Column Diameter (cm)"] ** 2  # cm²
    pore_velocity = setup["Flow Rate (mL/min)"] / (area * porosity)  # cm/min

    experimental_data = ExperimentalBreakthroughData(
        time=np.round((xls_btc["Time (min.)"] * lenght / pore_velocity), 2).to_numpy(),
        conc=xls_btc[["PFBA C/C0", "BEZ C/C0", "PFHxA C/C0", "DCF C/C0", "PFOA C/C0", "PFHxS C/C0"]].to_numpy().T,
    )

    st.session_state["experimental"] = experimental_data

    st.session_state["physical"] = dict(
        L=lenght,
        v=pore_velocity,
        n=porosity,
    )

st.title("Fitting experimental data")
lcol, rcol = st.columns([1, 2])

with rcol:
    experimental_data = st.session_state["experimental"]

    with st.expander("Experimental data"):
        fig = experimental_data.plot_breakthrough()
        st.pyplot(fig)

with lcol:
    sm = st.number_input("$s_m$", value=2000, min_value=1, max_value=100000, step=100)

    llcol, lrcol = st.columns(2)

    with llcol:
        k_ads = [
            st.number_input(f"$\\log k_{{ads|{i}}}$", value=1.5, min_value=-8.0, max_value=10.0, step=0.1)
            for i in range(1, 7)
        ]

    with lrcol:
        k_des = (1.5, 0.5, 0.15, 0.10, 0.02, 0.01)
        k_des = [
            st.number_input(f"$\\log k_{{des|{i}}}$", value=np.log10(k), min_value=-8.0, max_value=10.0, step=0.1)
            for i, k in enumerate(k_des, start=1)
        ]

    p = PhysicalParams(
        **st.session_state["physical"],
        sm=sm,
        k_ads=[10**k for k in k_ads],
        k_des=[10**k for k in k_des],
    )

    sim = Simulation(**p.nondim)
    sim.end_time = 650
    sim.cfl = 0.8
    sim.solve()

with rcol:
    fig = sim.plot_breakthrough()
    ax = fig.axes[0]
    colors = cycle(["darkgrey", "purple", "blue", "green", "orange", "red"])

    for btc, color in zip(experimental_data.conc, colors):
        ax.scatter(
            experimental_data.time,
            btc,
            c=color,
            path_effects=[Stroke(linewidth=1, foreground="#000")],
        )

    ax.set_ylim(bottom=-0.09, top=1.4)
    st.pyplot(fig)
