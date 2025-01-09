from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import Bounds

from espuma import Case_Directory, Boundary_Probe

from adsorption_model import (
    ThomasModelParameters,
    ThomasExperimentalSetup,
    Experiment,
    BreaktroughData,
)

from plotting import (
    make_single_plot,
    variate_some_parameter,
)

## Constants
C_0 = 2.07  # µg/L
length = 1.13  # cm
rho_p = 1100  # g/L
porosity = 0.4  # -
flow_rate = 1.90  # cm³/min
flow_rate *= 60  # cm³/h
column_diameter = 0.46  # cm
cross_area = 0.25 * np.pi * (column_diameter**2)  # cm²
pore_velocity = flow_rate / (porosity * cross_area)  # cm/h


def introduction():
    st.title("Non-linear adsorption")

    st.header("Governing equations", divider="rainbow")

    r"""
    Considering dispersion and diffusion processes negligible, 
    the mass balance for adsorption through a porous column is given by:

    $$
    \begin{equation}
        \dfrac{\partial C}{\partial t} 
        + v \dfrac{\partial C}{\partial z} 
        + \rho_p \left( \dfrac{1 - n}{n} \right) 
        \dfrac{\partial q}{\partial t} 
        = 0
    \end{equation}
    $$

    Where $C$ is the contaminant concentration in the aqueous phase [µg/L], 
    $q$ is the mass of contaminant adsorbed to the adsorbant [µg/g], 
    $v$ is the fluid velocity [cm/h], 
    $\rho_p$ is the particle density [g/L], 
    $n$ is the porosity [L³/L³],
    $t$ is the time [h], and $z$ is depth [cm]. 

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
    and $Z$ is the total length of the column [cm].
    
    A non-linear isotherm model (Langmuir kinetics) is used to describe the 
    adsorption process as a second-order reaction (with rate $k_T$) and 
    desorption as a first-order reaction (with rate $k_{T,2}$):

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

    Because, at equilibrium ($\partial q/\partial t = 0$), it takes 
    the form of the Langmuir isotherm:  

    $$
    \begin{equation}
        q = q_m \dfrac{b C}{1 + bC}
    \end{equation}
    $$

    Breakthrough data can be fitted to this model by tweaking its three 
    parameters:

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
        \dfrac{J(N/r, nT)}{J(N/r, NT) + \left(1 - J\left(N, NT/r\right)\right)
        \exp{\left(\left(1 - 1/r\right)\left(N - NT\right)\right)}} 
    \end{equation}
    $$

    With the non-dimensional parameters:

    $$
    \begin{align*}
        r &= 1 + bC_0 \\
        N &= \dfrac{\rho_p q_m k_T Z \left(1 - n\right)}{v\,n} \\
        T &= \dfrac{n \left(1/b + C_0\right)}{\rho_p q_m \left(1 - n\right)} 
        \left(\dfrac{v t}{Z} - 1\right)
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

    rf"""
    Consider a column experiment with:
    $$
    \begin{{array}}{{lrcl}}
        \textsf{{Initial concentration:}} 
        & C_0 &=& {C_0} \; \textsf{{µg/L}} \\ 
        \textsf{{Column length:}} 
        & Z &=& {length} \; \textsf{{cm}} \\
        \textsf{{Column diameter:}} 
        & d_{{\textsf{{col}}}} &=& {column_diameter} \; \textsf{{cm}} \\
        \textsf{{Volumetric flowrate:}} 
        & Q &=& {flow_rate:.1f} \; \textsf{{cm³/h}} \\
        \textsf{{Adsorbant dry density:}} 
        & \rho_p &=& {rho_p} \; \textsf{{g/L}} \\
        \textsf{{Porosity:}} 
        & n &=& {porosity}
    \end{{array}}
    $$

    Thus,
    $$
    \begin{{array}}{{lrcl}}
        \textsf{{Cross-sectional area:}}
        & A &=& {cross_area:.3f} \; \textsf{{cm²}} \\
        \textsf{{Pore velocity:}}
        & v = Q/(n \, A)
        &=& {pore_velocity:.1f} \; \textsf{{cm/h}} \\
        \textsf{{Time of 1 pore volume (PV):}}
        & t_{{\textsf{{PV}}}} = Z / v
        &=& {(3600 * length / pore_velocity):.2f} \; \textsf{{s}}\\ 
    \end{{array}}
    $$

    &nbsp;

    """

    setup = ThomasExperimentalSetup(
        C_0=C_0,
        length=length,
        pore_velocity=pore_velocity,
        rho_p=rho_p,
        porosity=porosity,
    )

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

        params = ThomasModelParameters(k_T=k_T, q_m=qm, b=b)

    with cols[1]:
        st.pyplot(make_single_plot(setup, params))

    st.subheader("Summary")

    tab1, tab2, tab3 = st.tabs(["$k_{T}$", "$q_m$", "$b$"])

    with tab1:
        k_T_array = [5e-2, 1e-1, 5e-1, 1e-0]  # L/µg.h
        st.pyplot(variate_some_parameter(setup, params, ("k_T", k_T_array)))

    with tab2:
        qm_array = [25, 40, 55]  # µg/g
        st.pyplot(variate_some_parameter(setup, params, ("q_m", qm_array)))

    with tab3:
        b_array = [1e-1, 1e-0, 1e1, 1e2]  # L/µg
        st.pyplot(variate_some_parameter(setup, params, ("b", b_array)))


def fitting_experiment():
    st.header("Fitting real data", divider="rainbow")

    rf"""
    Consider two column experiments with:
    $$
    \begin{{align*}}
        Z &= {length} \; \textsf{{cm}} \\
        \rho_p &= {rho_p} \; \textsf{{g/L}} \\
        n &= {porosity} \\
        d_\textsf{{cross}} &= {column_diameter} \; \textsf{{cm}} \\
        A_\textsf{{cross}} &= {cross_area:.3f} \; \textsf{{cm²}}
    \end{{align*}}
    $$

    |Experiment|Flow rate [mL/min]|Pore velocity [cm/h]|
    |:---:|:--:|:--:|
    |3 minute|6.25|{(v_3min :=6.25 * 60 / (porosity * cross_area)):.2f}|
    |10 minute|1.90|{(v_10min := 1.90 * 60 / (porosity * cross_area)):.2f}|

    We will try to fit the three model parameters, $k_T$, $q_m$ and $b$, 
    to the experimental data.

    """

    experiments = [
        Experiment(
            name="3 minute",
            contaminant="PFOA",
            setup=ThomasExperimentalSetup(
                C_0=1.425,  # µg/L
                length=length,  # cm
                pore_velocity=v_3min,  # cm/h
                rho_p=rho_p,  # g/L
                porosity=porosity,  # -
            ),
            parameters=ThomasModelParameters(),
            btc=BreaktroughData(
                time=data_3min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_3min["PFOA"].to_numpy() / 1000,  # µg/L
            ),
        ),
        Experiment(
            name="3 minute",
            contaminant="PFBA",
            setup=ThomasExperimentalSetup(
                C_0=1.032,  # µg/L
                length=length,  # cm
                pore_velocity=v_3min,  # cm/h
                rho_p=rho_p,  # g/L
                porosity=porosity,  # -
            ),
            parameters=ThomasModelParameters(),
            btc=BreaktroughData(
                time=data_3min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_3min["PFBA"].to_numpy() / 1000,  # µg/L
            ),
        ),
        Experiment(
            name="10 minute",
            contaminant="PFOA",
            setup=ThomasExperimentalSetup(
                C_0=2.097,  # µg/L
                length=length,  # cm
                pore_velocity=v_10min,  # cm/h
                rho_p=rho_p,  # g/L
                porosity=porosity,  # -
            ),
            parameters=ThomasModelParameters(),
            btc=BreaktroughData(
                time=data_10min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_10min["PFOA"].to_numpy() / 1000,  # µg/L
            ),
        ),
        Experiment(
            name="10 minute",
            contaminant="PFBA",
            setup=ThomasExperimentalSetup(
                C_0=0.976,  # µg/L
                length=length,  # cm
                pore_velocity=v_10min,  # cm/h
                rho_p=rho_p,  # g/L
                porosity=porosity,  # -
            ),
            parameters=ThomasModelParameters(),
            btc=BreaktroughData(
                time=data_10min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_10min["PFBA"].to_numpy() / 1000,  # µg/L
            ),
        ),
    ]

    tabs = st.tabs(
        [f"""{exp.name} - {exp.contaminant}""" for exp in experiments]
    )

    config = dict(
        initial_guess=ThomasModelParameters(k_T=1e-2, q_m=40, b=1),
        bounds=Bounds(lb=[10e-5, 0.1, 1e-4], ub=[1e1, 200_000, 1e3]),
    )

    for tab, exp in zip(tabs, experiments):
        with tab:
            exp.fit(**config)
            left, right = st.columns(2)
            fig = exp.plot_btc()
            left.pyplot(fig)
            fig = exp.plot_relative_btc()
            right.pyplot(fig)
            left, right = st.columns(2)
            left.write(exp.report_fit())
            fig = exp.plot_relative_btc(with_fit=True)
            right.pyplot(fig)


def fitting_while_fixing():
    st.header("Fitting with fixed parameters", divider="rainbow")

    rf"""
    Consider two column experiments with:
    $$
    \begin{{align*}}
        Z &= {length} \; \textsf{{cm}} \\
        \rho_p &= {rho_p} \; \textsf{{g/L}} \\
        n &= {porosity} \\
        d_\textsf{{cross}} &= {column_diameter} \; \textsf{{cm}} \\
        A_\textsf{{cross}} &= {cross_area:.3f} \; \textsf{{cm²}}
    \end{{align*}}
    $$

    |Experiment|Flow rate [mL/min]|Pore velocity [cm/h]|
    |:---:|:--:|:--:|
    |3 minute|6.25|{(v_3min :=6.25 * 60 / (porosity * cross_area)):.2f}|
    |10 minute|1.90|{(v_10min := 1.90 * 60 / (porosity * cross_area)):.2f}|

    From batch experiments, we obtained values for equilibrium isotherms

    |Parameter|PFOA|PFBA|
    |:---:|---:|---:|
    |$q_m$ (µg/g)| $1.156$ | $0.461$ |
    |$b$ (L/µg)| $261.3$ | $97.27$ |

    The values for $q_m$ are pretty distant from the values obtained from 
    the batch experiments, so we will try to fit the parameters $k_T$ and 
    $q_m$ to the experimental data, while fixing $b$ to the value obtained 
    from the batch experiments.
    """

    experiments = [
        Experiment(
            name="3 minute + batch",
            contaminant="PFOA",
            setup=ThomasExperimentalSetup(
                C_0=1.425,  # µg/L
                length=length,  # cm
                pore_velocity=v_3min,  # cm/h
                rho_p=rho_p,  # g/L
                porosity=porosity,  # -
            ),
            btc=BreaktroughData(
                time=data_3min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_3min["PFOA"].to_numpy() / 1000,  # µg/L
            ),
            parameters=ThomasModelParameters(b=261.3),
            # parameters=ThomasModelParameters(q_m=1.156, b=261.3),
        ),
        Experiment(
            name="3 minute + batch",
            contaminant="PFBA",
            setup=ThomasExperimentalSetup(
                C_0=1.032,  # µg/L
                length=length,  # cm
                pore_velocity=v_3min,  # cm/h
                rho_p=rho_p,  # g/L
                porosity=porosity,  # -
            ),
            btc=BreaktroughData(
                time=data_3min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_3min["PFBA"].to_numpy() / 1000,  # µg/L
            ),
            parameters=ThomasModelParameters(b=97.27),
            # parameters=ThomasModelParameters(q_m=0.461, b=97.27),
        ),
        Experiment(
            name="10 minute + batch",
            contaminant="PFOA",
            setup=ThomasExperimentalSetup(
                C_0=2.097,  # µg/L
                length=length,  # cm
                pore_velocity=v_10min,  # cm/h
                rho_p=rho_p,  # g/L
                porosity=porosity,  # -
            ),
            btc=BreaktroughData(
                time=data_10min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_10min["PFOA"].to_numpy() / 1000,  # µg/L
            ),
            parameters=ThomasModelParameters(b=261.3),
            # parameters=ThomasModelParameters(q_m=1.156, b=261.3),
        ),
        Experiment(
            name="10 minute + batch",
            contaminant="PFBA",
            setup=ThomasExperimentalSetup(
                C_0=0.976,  # µg/L
                length=length,  # cm
                pore_velocity=v_10min,  # cm/h
                rho_p=rho_p,  # g/L
                porosity=porosity,  # -
            ),
            btc=BreaktroughData(
                time=data_10min["Time (s)"].to_numpy() / 3600,  # h
                conc=data_10min["PFBA"].to_numpy() / 1000,  # µg/L
            ),
            parameters=ThomasModelParameters(b=97.27),
            # parameters=ThomasModelParameters(q_m=0.461, b=97.27),
        ),
    ]

    config = dict(
        initial_guess=ThomasModelParameters(k_T=1e-2, q_m=40),
        bounds=Bounds(lb=[10e-5, 0.1], ub=[1e1, 200_000]),
    )

    # config = dict(
    #     initial_guess=ThomasModelParameters(k_T=1e-2),
    #     bounds=Bounds(lb=[10e-5], ub=[1e1]),
    # )

    tabs = st.tabs(
        [f"""{exp.name} - {exp.contaminant}""" for exp in experiments]
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


def rewrite_equations():
    st.header("Fitting with fixed parameters", divider="rainbow")

    r"""
    Rewrite the equations to solve using OpenFOAM.

    $$
    \begin{equation}
        \dfrac{\partial C}{\partial t} 
        + v \dfrac{\partial C}{\partial z}
        + \rho_p \left( \dfrac{1 - n}{n} \right)\dfrac{\partial q}{\partial t} 
        = 0
    \end{equation}
    $$

    Where $C$ is the concentration of the adsorbent per water volume, $q$ is 
    the mass of adsorbed, $v$ is the pore water velocity, $\rho_p$ is the 
    particle density (mass per volume of solid), and $n$ is the volumetric 
    porosity.

    We multiply all terms by the porosity ($n$) to have them all in REV units.
    Also, we include a diffusive term to account for diffusion and 
    dispersion processes.
    
    $$
    \begin{equation}
        \dfrac{n \partial C}{\partial t} 
        + \texttt{q} \, \dfrac{\partial C}{\partial z} 
        - D \dfrac{\partial^2 C}{\partial z^2}
        = 
        - \rho_p \left(1 - n\right)\dfrac{\partial q}{\partial t} 
    \end{equation}
    $$

    Where $D$ is a diffusivity coefficient and $\texttt{q} = u \cdot n$ is 
    the Darcy's specific discharge. Rewriting $q$ as mass of adsorbent per 
    REV volume $S$:

    $$
    \begin{equation*}
        S = \rho_p \left(1 - n\right) q
    \end{equation*}
    $$

    And assuming that $\rho_p$ and $n$ are constants,
    
    $$
    \begin{equation}
        n \dfrac{\partial C}{\partial t} 
        + \texttt{q} \, \dfrac{\partial C}{\partial z} 
        - D \dfrac{\partial^2 C}{\partial z^2}
        = 
        - \dfrac{\partial S}{\partial t} 
    \end{equation}
    $$

    The non-linear isotherm model (Langmuir kinetics) is also rewritten 
    in terms of $S$:

    $$
    \begin{equation}
        \dfrac{\partial S}{\partial t} 
        = 
        k_T \left[
            C\left( S_m - S \right)
            - \dfrac{1}{b} \, S
        \right]
    \end{equation}
    $$

    Replacing in the governing equation:

    $$
    \begin{equation}
        \dfrac{n \partial C}{\partial t} 
        + \texttt{q} \, \dfrac{\partial C}{\partial z} 
        - D \dfrac{\partial^2 C}{\partial z^2}
        = 
        - k_T C\left( S_m - S \right)
        + \dfrac{k_T}{b} \, S
    \end{equation}
    $$

    We will rewrite the first term on the right-hand side as a non-dimensional
    *site blocking* term:

    $$
    \begin{equation}
        \dfrac{n \partial C}{\partial t} 
        + \texttt{q} \, \dfrac{\partial C}{\partial z} 
        - D \dfrac{\partial^2 C}{\partial z^2}
        = 
        - k_T S_m C 
        \underbrace{\left( 1 - \dfrac{S}{S_m} \right)}_{\text{site blocking}}
        + \dfrac{k_T}{b} \, S
    \end{equation}
    $$

    Finally, we have the following equation to use in our OpenFOAM solver:
    
    $$
    \begin{equation}
        \dfrac{n \partial C}{\partial t} 
        + \texttt{q} \, \dfrac{\partial C}{\partial z} 
        - D \dfrac{\partial^2 C}{\partial z^2}
        = 
        - n \, k_{\textsf{ads}} C \left( 1 - \dfrac{S}{S_m} \right)
        + k_{\textsf{des}} \, S
    \end{equation}
    $$

    These equations are equivalent to the Thomas model for $D=0$ and a single 
    adsorbate. The parameters $k_T$, $q_m$ and $b$ from the Thomas model are 
    related to these equations parameters $k_{\textsf{ads}}$, 
    $k_{\textsf{des}}$ and $S_m$ as follows:

    $$
    \begin{align*}
        s_m &= \rho_p \left(1 - n\right) q_m \\
        n \, k_{\textsf{ads}} &= k_T \rho_p \left(1 - n\right) q_m \\
        k_{\textsf{des}} &= \dfrac{k_T}{b} \\
    \end{align*}
    $$

    However, for a mixture of adsorbants, transport in the aqueous phase and 
    accumulation onto the adsorbant are represented by:

    $$
    \begin{align}
        \dfrac{n \partial C_i}{\partial t} 
        + \texttt{q}\dfrac{\partial C_i}{\partial z} 
        - D \dfrac{\partial^2 C_i}{\partial z^2}
        &= 
        - n \, k_{\textsf{ads},i} C_i 
        \left( 1 - \dfrac{\sum_i S_i}{S_m} \right)
        + k_{\textsf{des},i} \, S
    \\
        \dfrac{\partial S_i}{\partial t} 
        &= 
        n \, k_{\textsf{ads},i} C_i 
        \left( 1 - \dfrac{\sum_i S_i}{S_m} \right)
        - k_{\textsf{des},i} \, S
    \end{align}
    $$

    Where $C_i$ is the concentration of contaminant $i$ in the aqueous phase 
    and $S_i$ its adsorbed mass per REV volume. $k_{\textsf{ads},i}$ and 
    $k_{\textsf{des},i}$ are contaminant-specific adsorption and desorption 
    rate coefficients, respectively. In contrast, $S_m$ is considered an 
    adsorbant property, thus, its value is the same for all contaminants $i$. 
    The *site-blocking* term is a function of the adsorbed mass of 
    all contaminants $\sum_i S_i$.
    """

    st.subheader("Comparing models", divider="rainbow")

    r"""
    Consider the 3-minute PFOA column experiment with:
    
    $$
    \begin{align*}
        k_T =& 1.00\times 10^{-4} \, \textrm{L/µg.h} \\
        q_m =& 7.57 \times 10^{4} \, \textrm{µg/g} \\
        b =& 8.69 \times 10^{-4} \, \textrm{L/µg} \\
    \end{align*}
    $$

    Those parameters are equivalent to the following values for our
    OpenFOAM model:

    $$
    \begin{align*}
        S_m =& 7.57 \times 10^{4} \; \textrm{kg/m³} \\
        k_{\textsf{ads}} =& 3.47 \; \text{1/s} \\
        k_{\textsf{des}} =& 3.2 \times 10^{-5} \; \text{1/s} \\
    \end{align*}
    $$

    But are those values sensible?
    
    """

    of_case = Case_Directory("../../notebooks/_column_case")
    prb = Boundary_Probe(of_case, of_case.system.boundaryProbes)
    data = prb.array_data

    v = data["pfoa_aq"]
    rel_conc = v.isel(probes=0) / v.isel(probes=1)
    t_hours = data.time / 3600

    exp = Experiment(
        name="3 minute",
        contaminant="PFOA",
        setup=ThomasExperimentalSetup(
            C_0=1.425,  # µg/L
            length=length,  # cm
            pore_velocity=(6.25 * 60) / (porosity * cross_area),  # cm/h
            rho_p=rho_p,  # g/L
            porosity=porosity,  # -
        ),
        parameters=ThomasModelParameters(k_T=1e-4, q_m=7.57e4, b=8.69e-4),
        btc=BreaktroughData(
            time=data_3min["Time (s)"].to_numpy() / 3600,  # h
            conc=data_3min["PFOA"].to_numpy() / 1000,  # µg/L
        ),
    )

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(8, 3),
        sharey=True,
        gridspec_kw={"wspace": 0.05, "width_ratios": [0.5, 1, 1]},
    )

    ax = axs[0]
    ax.plot(data.time, rel_conc, label="PFOA", c="k")
    ax.set_ylabel("Relative concentration")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(left=0, right=3)
    ax.axvline(0.7197, label="1 Pore-volume", ls="dotted", lw=1, c="grey")

    ax = axs[1]
    ax.plot(t_hours, rel_conc, label="PFOA", c="k")
    ax.set_title("OpenFOAM model\nNumerical solution")
    ax.set_xlabel("Time (h)")
    ax.legend()
    # ax.set_xlim(left=-0.0025, right=0.0025)

    ax = axs[2]
    exp.plot_relative_btc(with_fit=True, ax=ax)
    ax.set_title("Thomas model\nAnalytical solution")
    ax.legend()

    for ax in axs:
        ax.axhline(1, color="gray", ls="dotted", lw=0.5, c="k")
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)

    st.pyplot(fig)


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
        st.Page(rewrite_equations, title="Rewrite equations"),
    ]

    nav = st.navigation(pages)
    nav.run()
