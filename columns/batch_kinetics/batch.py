import streamlit as st
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


"Check https://www.sciencedirect.com/science/article/pii/S266679082030032X"


xls_data = pd.read_excel("./.data/Batch Data.xlsx", index_col="time (min)")

data = {
    "Nanopure": xls_data[[x for x in xls_data.columns if "(NP)" in x]],
    "Wastewater": xls_data[[x for x in xls_data.columns if "(WW)" in x]],
}

tabs = st.tabs(["Equations", "Data"])

with tabs[0]:
    R"""
    $$
        \dfrac{d q(t)}{dt} = k_1\left[ q_e - q(t) \right] + k_2 \left[ q_e - q(t) \right]^2
    $$

    **Pseudo-first order equation:**
    
    If $q_e \ll k_1/k_2$

    $$
        \dfrac{d q(t)}{dt} = k_1\left[ q_e - q(t) \right]
    $$

    Then, 

    $$
        \boxed{
            q(t) = q_e \left[ 1 - \exp{\left(-k_1 \, t \right)} \right]
        }
    $$

    Which is linearized to

    $$
        \ln{\left( q_e - q(t) \right)} = \ln{(q_e)} - k_1 \, t
    $$
    """


def pfo(t: NDArray, q_e: NDArray, k_1: NDArray):
    return q_e * (1 - np.exp(-k_1 * t))


def pso(t: NDArray, q_e: NDArray, k_2: NDArray):
    return (q_e**2 * k_2 * t) / (1 + t * q_e * k_2)


models = {"Pseudo-first order": pfo, "Pseudo-second order": pso}

with tabs[1]:
    selection = st.selectbox("Water?", data.keys())
    selected_dataset = data[selection]
    column = st.dataframe(
        selected_dataset, selection_mode="single-column", on_select="rerun", height=200
    )

    if col := column["selection"]["columns"]:
        col_name = col[0]

        model_select = st.selectbox("Model?", models.keys())
        popt, pcov = curve_fit(
            model := models[model_select],
            selected_dataset.index,
            selected_dataset[col_name],
            p0=[10, 0.01],
        )

        fig, ax = plt.subplots()
        selected_dataset.plot(
            y=col_name, ax=ax, kind="line", lw=0, marker="x", use_index=True
        )
        ax.set_ylim(0, 7)
        ax.set_ylabel("$q(t)$ [ng/mg]")

        ax.plot(
            model(np.arange(0.1, 1500), *popt),
            ls="dashed",
            label=f"$q_e$ = {popt[0]:.2f}\n$k$ = {popt[1]:.4f}",
        )
        ax.legend()
        st.pyplot(fig)
