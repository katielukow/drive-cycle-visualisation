import streamlit as st
import plotly.graph_objects as go
from data_analysis import load_data
import pandas as pd
import numpy as np

@st.cache_data
def prepare_data():
    dc_all_fil, data_all, dc_all = load_data()
    
    # Pre-compute charge/discharge status for each drive cycle
    cycle_status = {key: 'charge' if (df['Current'] >= 0).all() else 'discharge' for key, df in dc_all_fil.items()}
    
    # Pre-compute time in hours for each drive cycle
    for df in dc_all_fil.values():
        df['Time_Hours'] = (df['DateTime'] - df['DateTime'].iloc[0]).dt.total_seconds() / 3600
    
    return dc_all_fil, cycle_status, data_all

def filter_data(dc_all_fil, cycle_status, charge, discharge):
    if charge and discharge:
        return dc_all_fil
    elif charge:
        return {k: v for k, v in dc_all_fil.items() if cycle_status[k] == 'charge'}
    elif discharge:
        return {k: v for k, v in dc_all_fil.items() if cycle_status[k] == 'discharge'}
    else:
        return {}

def create_figure(filtered_dict_V):
    fig = go.Figure()
    for key, df in filtered_dict_V.items():
        fig.add_trace(go.Scatter(x=df['Time_Hours'], y=df['Voltage'], mode='lines', name=f'Drive Cycle {key}'))

    fig.update_layout(
        title='Voltage vs Time for Filtered Drive Cycles',
        xaxis_title='Time (hours)',
        yaxis_title='Voltage',
        legend_title='Drive Cycles',
        height=600,
    )
    fig.update_xaxes(range=[-5, None])
    return fig

def app():
    dc_all_fil, cycle_status, data_all = prepare_data()

    charge = st.sidebar.checkbox("Charge", True, key="page1_charge")
    discharge = st.sidebar.checkbox("Discharge", True, key="page1_discharge")
    
    filtered_dict_V = filter_data(dc_all_fil, cycle_status, charge, discharge)

    if filtered_dict_V:
        fig = create_figure(filtered_dict_V)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Number of filtered drive cycles: {len(filtered_dict_V)}")
    else:
        st.write("No drive cycles meet the specified criteria.")
    


if __name__ == "__main__":
    app()