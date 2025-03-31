import streamlit as st
import plotly.graph_objects as go
from data_analysis import load_data, stats_calc, charge_rate, pybamm_plot
import pandas as pd
import numpy as np

def current_rate_calc(data_all, dc_all):
    cycle_status = {key: 'charge' if (np.mean(df['Current']) <= 0) else 'discharge' for key, df in dc_all.items()}

    charge_dict = {k: v for k, v in dc_all.items() if cycle_status[k] == 'charge'}
    discharge_dict = {k: v for k, v in dc_all.items() if cycle_status[k] == 'discharge' and k not in [3, 4]}

    Q_pack = 11
    I_discharge = np.round(np.abs(pd.Series([df['Current'].mean() for df in discharge_dict.values()]).mean())/Q_pack,1)
    I_charge = np.round(abs(charge_rate(charge_dict)/Q_pack),1)

    st.write(stats_calc(charge_dict))
    st.write(stats_calc(discharge_dict))
    st.write(data_all['DateTime'].iloc[-1], data_all['DateTime'].iloc[0])
    time_discharge = stats_calc(discharge_dict)["Duration [s]"].sum()
    time_charge = stats_calc(charge_dict)["Duration [s]"].sum()
    total_time = (data_all['DateTime'].iloc[-1] - data_all['DateTime'].iloc[0]).total_seconds()
    time_off = total_time - time_discharge - time_charge
    time_data = pd.Series({
            'Discharge': time_discharge,
            'Charge': time_charge,
            'Off': time_off
        })

    return I_discharge, I_charge, time_data, total_time

def app():
    st.title('Initial Data Analysis')
    # _, data_all, dc_all = load_data()
    I_discharge, I_charge, time_data, all_time = current_rate_calc(st.session_state.data_all, st.session_state.dc_all)
    
    st.write('### Basic Charge-Discharge Profile')
    st.write('The following pybamm experiment definition will cycle the battery between 0% and 100% SOC at the average C-Rates from the filtered field data. Discharge rate is taken as the average current during discharge and charge rate is calculated as the average current during the constant current portion of charge.')
    simple_exp = [
        "Discharge at "+ str(I_discharge) +"C until 2.5 V",
        "Charge at "+ str(I_charge) +"C until 4.2 V",
        "Hold at 4.2 V until 50 mA"]
    
    formatted_simple = ',\n        '.join(f'"{step}"' for step in simple_exp)
    st.markdown(f"""
        ```python
        pybamm.Experiment([(
            {formatted_simple}
        )])""")
    
    if st.button('Plot basic charge-discharge profile:'):
        pybamm_plot(simple_exp)

    # --------------------------------------------------------------------------------------------------------------------

    st.write('### Distribution of Asset Usage')
    st.write('The pie chart below shows the distribution of power consumption during the drive cycles. This can then be incorporated into the simplified day load profile with a consideration of the time when the asset is not in use.')

    fig_pie = go.Figure(data=[go.Pie(labels=['Discharge', 'Charge', 'Off'], values=time_data)])
    st.plotly_chart(fig_pie, use_container_width=True)

    st.write('The following PyBaMM load profile considers the C-Rates from the filtered field data previously presented. We make the assumption that there is a full discharge and charge cycle with the time distribution. Taking this in to account, the representative off time is determined and the PyBaMM experiment is defined.')
    st.write(time_data['Discharge']/time_data.sum(axis=0))
    discharge_time = 1/I_discharge 
    total_time = np.round(discharge_time / (time_data['Discharge']/time_data.sum(axis=0)),2) - discharge_time - 1/I_charge

    simple_rest = [
        "Discharge at "+ str(I_discharge) +"C until 2.5 V",
        "Charge at "+ str(I_charge) +"C until 4.2 V",
        "Hold at 4.2 V until 50 mA",
        "Rest for "+ str(total_time) +" hours (60 minute period)"]

    formatted_simple_rest = ',\n        '.join(f'"{step}"' for step in simple_rest)
    st.markdown(f"""
        ```python
        pybamm.Experiment([(
            {formatted_simple_rest}
        )])""")
    
    if st.button('Plot basic charge-discharge profile with rest:'):
        pybamm_plot(simple_rest)


if __name__ == "__main__":
    app()
