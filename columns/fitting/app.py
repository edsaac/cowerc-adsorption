import streamlit as st
from pathlib import Path
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
# from scipy.optimize import curve_fit
from scipy.stats import linregress

def thomas_model(t, a, b):
    return 1/(1 + np.exp(a - b*t))

def log_thomas_model(t, a, b):
    return 1/(1 + np.exp(a - b*np.log(t)))


data_folder = Path("./experimental_data")
metadata_file = data_folder / "metadata.csv"
metadata = read_csv(metadata_file)
list_experiments = [f for f in data_folder.iterdir() if "EBCT" in f.name]

with st.sidebar:
    experiment = st.selectbox("Select experiment", list_experiments, format_func= lambda x: x.name, index=None)

if experiment:

    ## Select experiment
    df = read_csv(experiment)
    keys = df.keys()

    with st.expander("🗃️ Data", expanded=True):
        st.dataframe(df, use_container_width=True, hide_index=True, height=200,
            column_config={k:st.column_config.NumberColumn(format="%.2e") for k in keys if "kg/m3" in k}
        )
        

    time_col = st.sidebar.selectbox("Time column", options=keys, index=2)
    rel_concen_col = st.sidebar.selectbox("Concentration column", options=[k for k in keys if "C0" in k])
    
    concen_col = rel_concen_col.replace("C/C0", "kg/m3")
    
    ## Data to fit
    dfma = df[0 < df[rel_concen_col]]
    time = dfma[time_col]/60  ## seconds to minutes
    rel_conc = dfma[rel_concen_col]
    conc = dfma[concen_col]

    tlin = np.linspace(time.min(), time.max(), 120)

    with st.sidebar:
        "****"
        "**Experiment known parameters**"
        adsorbant_mass = float(metadata["M [g]"][metadata["Experiment"] == experiment.name].iloc[0])
        MASSADSB = st.number_input("Adsorbant mass $M$ [g(s)]", value=adsorbant_mass, format="%.3f")
        
        flowrate = float(metadata["Q [mL/min]"][metadata["Experiment"] == experiment.name].iloc[0])
        FLOWRATE = st.number_input("Flowrate $Q$ [cm³/min]", value=flowrate)
        
        initial_concentration = np.mean(conc/rel_conc)
        INITCONC = st.number_input("Initial conc. $C_0$ [mg(c)/cm³]", value=initial_concentration, format="%.3e")

    
    
    tabs = st.tabs(["Summary", "Thomas model", "log-Thomas model"])
    
    with tabs[1]:
        with st.expander("Details:"):
            R"""
            #### Thomas model:
            $$
                \dfrac{C}{C_0} = \left[ 1 + \exp{\left( \dfrac{k_T q_0 M}{Q} - k_T C_0 t \right)} \right]^{-1}
            $$

            Equivalent to:
            $$
                \dfrac{C}{C_0} = \left[ 1 + \exp{\left( a - b t \right)} \right]^{-1}
            $$

            With:
            $$
                a = \dfrac{k_T q_0 M}{Q} \qquad b = k_T C_0
            $$

            Where $a$ and $b$ are the intercept and slope of a linear regression of:

            $$
                \log{\left(\dfrac{C_0}{C} - 1\right)} = a - bt
            $$ 

            So the model parameters $k_T$ and $q_0$ are found replacing:
            
            $$
                k_T = \dfrac{b}{C_0} \qquad q_0 = \dfrac{a Q}{k_T M}
            $$

            With units

            $$
                k_T = \left[ \dfrac{\mathrm{cm^3}}{\mathrm{mg_{(c)}} \; \mathrm{min}} \right]
                \qquad
                q_0 = \left[ \dfrac{\mathrm{mg_{(c)}}}{\mathrm{g_{(s)}}} \right]
            $$

            *********
            """

        cols = st.columns([2,1])
        with cols[0]:
            fig,ax = plt.subplots(figsize=[4,3])
            ax.set_title("Thomas model", fontsize=10)
            ln_conc = np.log((1/rel_conc) - 1)
            lr = linregress(time, ln_conc)
            a, b = lr.intercept, -lr.slope  
            clin = thomas_model(tlin, a, b)
            ax.scatter(time, ln_conc, label="Data", color='k')
            ax.plot(tlin, a - b * tlin, label="Thomas model")
            ax.set_ylabel(R"$\ln(C_0/C - 1)$")
            ax.ticklabel_format(axis='x', useMathText=True, scilimits=[0,0])
            ax.legend(prop={'size':8})
            ax.set_xlabel("Time $t$ [min]")
            
            st.pyplot(fig)

        with cols[1]:
            
            fR"""
            **Best-fit parameters:**
            
            |     | Value| Unit|
            |:---:|--:|:--|
            |$a$|{a:.2e}|-|
            |$b$|{b:.2e}|-|
            
            &nbsp;
            
            **Then:**
            
            |     | Value| Unit|
            |:---:|--:|:--|
            |$k_T$|{b/INITCONC:.2e}|$\tfrac{{\rm cm^3}}{{\rm mg·min}}$|
            |$q_0$|{a*FLOWRATE/(MASSADSB * b/INITCONC):.2e}|$\tfrac{{\rm mg}}{{\rm g}}$|
            """
            

    with tabs[2]:
        with st.expander("Details:"):
            R"""
                #### log-Thomas model:
                $$
                \dfrac{C}{C_0} = \left[ 1 + \exp{\left( k_T \log{\left(\dfrac{q_0 M}{Q}\right)} - k_T \log{\left(C_0 t\right)} \right)} \right]^{-1}
                $$

                Equivalent to:
                $$
                \dfrac{C}{C_0} = \left[ 1 + \exp{\left( a - b \log{(t)} \right)} \right]^{-1}
                $$

                With:
                $$
                    a = k_T \log{\left(\dfrac{q_0 M}{Q C_0}\right)} \qquad b = k_T
                $$

                Where $a$ and $b$ are the intercept and slope of a linear regression of:

                $$
                    \log{\left(\dfrac{C_0}{C} - 1\right)} = a - b\log(t)
                $$

                So the model parameters $k_T$ and $q_0$ are found replacing:
                
                $$
                    k_T = b \qquad q_0 = \dfrac{Q C_0}{M}\exp{\left(\dfrac{a}{k_T}\right)}
                $$ 
                """
            
        cols = st.columns([2,1])
        with cols[0]:
            fig,ax = plt.subplots(figsize=[4,3])
            ax.set_title("log-Thomas model", fontsize=10)
            # popt, pcov, info, msg, ier = curve_fit(log_thomas_model, time, rel_conc, p0=[1, 1], full_output=True)
            lr = linregress(np.log(time), ln_conc)
            a, b = lr.intercept, -lr.slope
            
            clog = log_thomas_model(tlin, a, b)
            ax.scatter(time, ln_conc, label="Data", color='k')
            ax.plot(tlin, np.log(1/clog -1), label="log-Thomas", c='r')
            ax.set_xscale('log')
            ax.legend(prop={'size':8})
            ax.set_ylabel(R"$\ln(C_0/C - 1)$")
            ax.set_xlabel("Time $t$ [min]")
            st.pyplot(fig)

        with cols[1]:

            fR"""
            **Best-fit parameters:**
            
            |     | Value| Unit|
            |:---:|--:|:--|
            |$a$|{a:.2e}|-|
            |$b$|{b:.2e}|-|
            
            &nbsp;
            
            **Then:**
            
            |     | Value| Unit|
            |:---:|--:|:--|
            |$k_T$|{b:.2e}|$\tfrac{{\rm cm^3}}{{\rm mg·min}}$|
            |$q_0$|{(INITCONC * FLOWRATE)/MASSADSB * np.exp(a/b):.2e}|$\tfrac{{\rm mg}}{{\rm g}}$|
            """

    with tabs[0]:
        fig,ax = plt.subplots(figsize=[4,3])
        ax.scatter(time, rel_conc, label="Data", color='k')
        ax.set_ylabel(rel_concen_col)
        ax.plot(tlin, clin, label="Thomas model")
        ax.plot(tlin, clog, label="log-Thomas", c='r')
        ax.ticklabel_format(axis='x', useMathText=True, scilimits=[0,0])
        ax.legend(prop={'size':8})
        ax.set_xlabel("Time $t$ [min]")

        st.pyplot(fig)

with st.expander("Metadata"):
    st.dataframe(metadata, hide_index=True, use_container_width=True)

