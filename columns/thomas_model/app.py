from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd
from scipy.optimize import Bounds

from thomas_model_plots import (
    make_single_plot,
    variate_some_parameter,
)
from thomas_model_equations import (
    Experiment,
    ExperimentalSetup,
    BreaktroughData,
    ThomasModelInputs,
    TargetParameters,
)


def introduction():
    st.title("Thomas model")

    st.header("Governing equations", divider="rainbow")

    r"""
    Considering dispersion and diffusion processes negligible, 
    the mass balance for adsorption through a porous column is given by:

    $$
    \begin{equation}
        \dfrac{\partial C}{\partial t} 
        + v \dfrac{\partial C}{\partial z} 
        + \rho_p \left( \dfrac{1 - \epsilon}{\epsilon} \right) \dfrac{\partial q}{\partial t} 
        = 0
    \end{equation}
    $$

    Where $C$ is the contaminant concentration in the aqueous phase [µg/L], 
    $q$ is the mass of contaminant adsorbed to the adsorbant [µg/g], 
    $v$ is the fluid velocity [cm/h], 
    $\rho_p$ is the particle density [g/L], 
    $\epsilon$ is the porosity [L³/L³],
    $t$ is the time [h], and $z$ is the column height [cm].

    The initial conditions and boundary conditions are:

    $$
    \begin{align*}
        C(t=0, z) &= 0 \\
        q(t=0, z) &= 0 \\
        \\
        C(t, z = 0) &= C_0 \\
        \frac{dC}{dt} \Big|_{(t, \, z=Z)} &= 0 \\
    \end{align*}
    $$

    Where $C_0$ is the initial concentration of the contaminant [µg/L] 
    and $Z$ is the length of the column [cm].
    """

    r"""
    A non-linear isotherm model (Langmuir kinetics) is used to describe the adsorption process
    as a second-order reaction (with rate $k_T$) and desorption as a first-order reaction (with rate $k_{T,2}$):

    $$
    \begin{equation}
        \dfrac{\partial q}{\partial t} 
        = k_T C\left( q_m - q \right) - k_{T,2}\, q
    \end{equation}
    $$

    This equation is rewritten with $b = k_T/k_{T,2}$ 
    $$
    \begin{equation}
        \dfrac{\partial q}{\partial t} 
        = k_T \left[ C\left( q_m - q \right) - \dfrac{q}{b}\right]
    \end{equation}
    $$

    At equilibrium ($\partial q/\partial t = 0$), it takes the form of a Langmuir isotherm:  
    $$
    \begin{equation}
        q = q_m \dfrac{b C}{1 + bC}
    \end{equation}
    $$

    Breakthrough data can be fitted to this model by tweaking its three parameters:

    |-|Units|Description|
    |:---:|:---:|:---|
    |$k_T$|L/µg.h|First-order adsorption rate|
    |$q_m$|µg/g|Adsorbant capacity|
    |$b$|L³/µg|Langmuir-isotherm parameter|
    """

    st.header("Analytical solution", divider="rainbow")

    r"""
    This differential has the following analytical solution:

    $$
    \begin{equation}
        \dfrac{C}{C_0} = 
        \dfrac{J(n/r, nT)}{J(n/r, nT) + \left(1 - J\left(n, nT/r\right)\right)
        \exp{\left(\left(1 - 1/r\right)\left(n - nT\right)\right)}}
    \end{equation}
    $$

    With the non-dimensional parameters:

    $$
    \begin{align*}
        r &= 1 + bC_0 \\
        n &= \dfrac{\rho_p q_m k_T Z \left(1 - \epsilon\right)}{v\,\epsilon} \\
        T &= \dfrac{\epsilon \left(1/b + C_0\right)}{\rho_p q_m \left(1 - \epsilon\right)} \left(\dfrac{v t}{Z} - 1\right)
    \end{align*}
    $$

    And with the $J(x,y)$ function defined as:

    $$
    \begin{equation}
        J(x,y) = 1 - \int_{0}^{x} 
            \exp{\left(-y - \zeta\right)} \,
            I_0\left(2 \sqrt{\zeta y}\right) \,
            d\zeta
    \end{equation}
    $$

    Where $I_0(x)$ is the modified Bessel function of the first kind.
    """

    st.header("Playing with parameters", divider="rainbow")

    ## Constants
    C_0 = 2.07  # µg/L
    length = 1.13  # cm
    rho_p = 1100  # g/L
    epsilon = 0.4  # -
    flow_rate = 1.90  # cm³/min
    flow_rate *= 60  # cm³/h

    column_diameter = 0.46  # cm
    cross_area = 0.25 * np.pi * (column_diameter**2)  # cm²
    pore_velocity = flow_rate / (epsilon * cross_area)  # cm/h

    rf"""
    Consider a column experiment with:
    $$
    \begin{{align*}}
        C_0 &= {C_0} \; \textsf{{µg/L}} \\
        Z &= {length} \; \textsf{{cm}} \\
        v &= {pore_velocity:.1f} \; \textsf{{cm/h}} \\
        \rho_p &= {rho_p} \; \textsf{{g/L}} \\
        \epsilon &= {epsilon}
    \end{{align*}}
    $$
    """

    cols = st.columns([1, 2])

    with cols[0]:
        ## Parameters
        k_T_range = np.arange(5, 101, 1) * 1e-2  # L/µg.h
        k_T = st.select_slider(
            r"$k_{T}$ [L/µg.h]",
            options=k_T_range,
            value=np.take(k_T_range, len(k_T_range) // 2),
            format_func=lambda x: f"{x:.1e}",
        )

        qm_array = np.arange(5, 65, 5)  # ng/g
        qm = st.select_slider(
            r"$q_m$ [µg/g]",
            options=qm_array,
            value=np.take(qm_array, len(qm_array) // 2),
            format_func=lambda x: f"{x:.0f}",
        )

        b_array = np.power(10.0, np.arange(-1, 2.5, 0.5))  # L/µg
        b = st.select_slider(
            r"$b$ [L/µg]",
            options=b_array,
            value=np.take(b_array, len(b_array) // 2),
            format_func=lambda x: f"{x:.1e}",
        )

        model_params = ThomasModelInputs(
            k_T=k_T,
            q_m=qm,
            b=b,
            C_0=C_0,
            length=length,
            pore_velocity=pore_velocity,
            rho_p=rho_p,
            epsilon=epsilon,
        )

    with cols[1]:
        st.pyplot(make_single_plot(model_params))

    st.subheader("Summary")

    tab1, tab2, tab3 = st.tabs(["$k_{T}$", "$q_m$", "$b$"])

    with tab1:
        k_T_array = [5e-2, 1e-1, 5e-1, 1e-0]  # L/µg.h
        st.pyplot(variate_some_parameter(model_params, "k_T", k_T_array))

    with tab2:
        qm_array = [25, 40, 55]  # µg/g
        st.pyplot(variate_some_parameter(model_params, "q_m", qm_array))

    with tab3:
        b_array = [1e-1, 1e-0, 1e1, 1e2]  # L/µg
        st.pyplot(variate_some_parameter(model_params, "b", b_array))


def fitting_experiment():
    st.header("Fitting real data", divider="rainbow")

    ## Constants
    length = 1.13  # cm
    rho_p = 1100  # g/L
    epsilon = 0.4  # -
    column_diameter = 0.46  # cm
    cross_area = 0.25 * np.pi * (column_diameter**2)  # cm²

    rf"""
    Consider two column experiments with:
    $$
    \begin{{align*}}
        Z &= {length} \; \textsf{{cm}} \\
        \rho_p &= {rho_p} \; \textsf{{g/L}} \\
        \epsilon &= {epsilon} \\
        d_\textsf{{cross}} &= {column_diameter} \; \textsf{{cm}} \\
        A_\textsf{{cross}} &= {cross_area:.3f} \; \textsf{{cm²}}
    \end{{align*}}
    $$

    |Experiment|Flow rate [mL/min]|Pore velocity [cm/h]|
    |:---:|:--:|:--:|
    |3 minute|6.25|{(v_3min :=6.25 * 60 / (epsilon * cross_area)):.2f}|
    |10 minute|1.90|{(v_10min := 1.90 * 60 / (epsilon * cross_area)):.2f}|

    We will try to fit the three model parameters, $k_T$, $q_m$ and $b$, to the experimental 
    data.

    """

    experiments = [
        Experiment(
            name="3 minute",
            contaminant="PFOA",
            setup=ExperimentalSetup(
                length=length,  # cm
                pore_velocity=v_3min,  # cm/h
                rho_p=rho_p,  # g/L
                epsilon=epsilon,  # -
            ),
            btc=BreaktroughData(
                time=data_3min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_3min["PFOA"].to_numpy() / 1000,  # µg/L
                C_0=1.425,  # µg/L
            ),
        ),
        Experiment(
            name="3 minute",
            contaminant="PFBA",
            setup=ExperimentalSetup(
                length=length,  # cm
                pore_velocity=v_3min,  # cm/h
                rho_p=rho_p,  # g/L
                epsilon=epsilon,  # -
            ),
            btc=BreaktroughData(
                time=data_3min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_3min["PFBA"].to_numpy() / 1000,  # µg/L
                C_0=1.032,  # µg/L
            ),
        ),
        Experiment(
            name="10 minute",
            contaminant="PFOA",
            setup=ExperimentalSetup(
                length=length,  # cm
                pore_velocity=v_10min,  # cm/h
                rho_p=rho_p,  # g/L
                epsilon=epsilon,  # -
            ),
            btc=BreaktroughData(
                time=data_10min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_10min["PFOA"].to_numpy() / 1000,  # µg/L
                C_0=2.097,  # µg/L
            ),
        ),
        Experiment(
            name="10 minute",
            contaminant="PFBA",
            setup=ExperimentalSetup(
                length=length,  # cm
                pore_velocity=v_10min,  # cm/h
                rho_p=rho_p,  # g/L
                epsilon=epsilon,  # -
            ),
            btc=BreaktroughData(
                time=data_10min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_10min["PFBA"].to_numpy() / 1000,  # µg/L
                C_0=0.976,  # µg/L
            ),
        ),
    ]

    tabs = st.tabs([f"""{exp.name} - {exp.contaminant}""" for exp in experiments])

    config = dict(
        initial_guess=TargetParameters(k_T=1e-2, q_m=40, b=1),
        bounds=Bounds(lb=[10e-5, 0.1, 1e-4], ub=[1e1, 200_000, 1e3]),
    )

    for tab, exp in zip(tabs, experiments):
        with tab:
            exp.fit(**config)
            left, right = st.columns(2)
            left.pyplot(exp.plot_btc())
            right.pyplot(exp.plot_relative_btc())
            left, right = st.columns(2)
            left.write(exp.report_fit())
            right.pyplot(exp.plot_relative_btc(with_fit=True))


def fitting_while_fixing():
    st.header("Fitting with fixed parameters", divider="rainbow")

    ## Constants
    length = 1.13  # cm
    rho_p = 1100  # g/L
    epsilon = 0.4  # -
    column_diameter = 0.46  # cm
    cross_area = 0.25 * np.pi * (column_diameter**2)  # cm²

    ## Constants
    length = 1.13  # cm
    rho_p = 1100  # g/L
    epsilon = 0.4  # -
    column_diameter = 0.46  # cm
    cross_area = 0.25 * np.pi * (column_diameter**2)  # cm²

    rf"""
    Consider two column experiments with:
    $$
    \begin{{align*}}
        Z &= {length} \; \textsf{{cm}} \\
        \rho_p &= {rho_p} \; \textsf{{g/L}} \\
        \epsilon &= {epsilon} \\
        d_\textsf{{cross}} &= {column_diameter} \; \textsf{{cm}} \\
        A_\textsf{{cross}} &= {cross_area:.3f} \; \textsf{{cm²}}
    \end{{align*}}
    $$

    |Experiment|Flow rate [mL/min]|Pore velocity [cm/h]|
    |:---:|:--:|:--:|
    |3 minute|6.25|{(v_3min :=6.25 * 60 / (epsilon * cross_area)):.2f}|
    |10 minute|1.90|{(v_10min := 1.90 * 60 / (epsilon * cross_area)):.2f}|

    From batch experiments, we obtained values for equilibrium isotherms
    
    |Parameter|PFOA|PFBA|
    |:---:|---:|---:|
    |$q_m$ (µg/g)| $1.156$ | $0.461$ | 
    |$b$ (L/µg)| $261.3$ | $97.27$ |
    
    We will try to fit two parameters $k_T$, $q_m$, and fix $b$ to the value obtained from 
    the batch experiments.
    """

    experiments = [
        Experiment(
            name="3 minute + batch",
            contaminant="PFOA",
            setup=ExperimentalSetup(
                length=length,  # cm
                pore_velocity=v_3min,  # cm/h
                rho_p=rho_p,  # g/L
                epsilon=epsilon,  # -
            ),
            btc=BreaktroughData(
                time=data_3min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_3min["PFOA"].to_numpy() / 1000,  # µg/L
                C_0=1.425,  # µg/L
            ),
            parameters=TargetParameters(b=261.3),
        ),
        Experiment(
            name="3 minute + batch",
            contaminant="PFBA",
            setup=ExperimentalSetup(
                length=length,  # cm
                pore_velocity=v_3min,  # cm/h
                rho_p=rho_p,  # g/L
                epsilon=epsilon,  # -
            ),
            btc=BreaktroughData(
                time=data_3min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_3min["PFBA"].to_numpy() / 1000,  # µg/L
                C_0=1.032,  # µg/L
            ),
            parameters=TargetParameters(b=97.27),
        ),
        Experiment(
            name="10 minute + batch",
            contaminant="PFOA",
            setup=ExperimentalSetup(
                length=length,  # cm
                pore_velocity=v_10min,  # cm/h
                rho_p=rho_p,  # g/L
                epsilon=epsilon,  # -
            ),
            btc=BreaktroughData(
                time=data_10min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_10min["PFOA"].to_numpy() / 1000,  # µg/L
                C_0=2.097,  # µg/L
            ),
            parameters=TargetParameters(b=261.3),
        ),
        Experiment(
            name="10 minute + batch",
            contaminant="PFBA",
            setup=ExperimentalSetup(
                length=length,  # cm
                pore_velocity=v_10min,  # cm/h
                rho_p=rho_p,  # g/L
                epsilon=epsilon,  # -
            ),
            btc=BreaktroughData(
                time=data_10min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_10min["PFBA"].to_numpy() / 1000,  # µg/L
                C_0=0.976,  # µg/L
            ),
            parameters=TargetParameters(b=97.27),
        ),
    ]

    config = dict(
        initial_guess=TargetParameters(k_T=1e-2, q_m=40),
        bounds=Bounds(lb=[10e-5, 0.1], ub=[1e1, 200_000]),
    )

    tabs = st.tabs([f"""{exp.name} - {exp.contaminant}""" for exp in experiments])

    for tab, exp in zip(tabs, experiments):
        with tab:
            exp.fit(**config)
            left, right = st.columns(2)
            left.pyplot(exp.plot_btc())
            right.pyplot(exp.plot_relative_btc())
            left, right = st.columns(2)
            left.write(exp.report_fit())
            right.pyplot(exp.plot_relative_btc(with_fit=True))


if __name__ == "__main__":
    data_path_3min = Path(
        "../../.data/ModelFitting/ThomasBessel_EBCT/EBCT3PFOAPFBA.csv"
    )
    data_path_10min = Path(
        "../../.data/ModelFitting/ThomasBessel_EBCT/ebct10pfoapfba.csv"
    )

    data_3min = pd.read_csv(data_path_3min)
    data_10min = pd.read_csv(data_path_10min)

    st.html("""<style>
        table {
        margin-left: auto;
        margin-right: auto;
        }
        </style>
    """)

    pages = [
        st.Page(introduction, title="Introduction"),
        st.Page(fitting_experiment, title="Fitting experiment"),
        st.Page(fitting_while_fixing, title="Fitting with fixed parameters"),
    ]

    nav = st.navigation(pages)
    nav.run()
