import streamlit as st
import plotly.graph_objects as go
from data_analysis import load_data
import pandas as pd

# @st.cache_data
# def prepare_data():
#     # Pre-compute cycle status and time in hours for each drive cycle
#     dc_all_fil = st.session_state.dc_all_fil
    
#     # Compute cycle status and time in hours in a single loop


#     return dc_all_fil, cycle_status, st.session_state.data_all

def filter_data(dc_all_fil, cycle_status, include_charge, include_discharge):
    # Use logical checks to filter only necessary items
    if include_charge and include_discharge:
        return dc_all_fil
    elif include_charge:
        return {k: v for k, v in dc_all_fil.items() if cycle_status[k] == 'charge'}
    elif include_discharge:
        return {k: v for k, v in dc_all_fil.items() if cycle_status[k] == 'discharge'}
    else:
        return {}

@st.cache_data
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
    # Prepare data
    # dc_all_fil, cycle_status, _ = prepare_data()
    dc_all_fil = st.session_state.dc_all_fil
    
    cycle_status = {}
    for key, df in dc_all_fil.items():
        cycle_status[key] = 'charge' if (df['Current'] >= 0).all() else 'discharge'
        if 'Time_Hours' not in df.columns:  # Avoid recomputing if already added
            df['Time_Hours'] = (df['DateTime'] - df['DateTime'].iloc[0]).dt.total_seconds() / 3600

    # Sidebar filters
    charge = st.sidebar.checkbox("Charge", True, key="page1_charge")
    discharge = st.sidebar.checkbox("Discharge", True, key="page1_discharge")

    # Filter data based on the selected options
    filtered_dict_V = filter_data(dc_all_fil, cycle_status, charge, discharge)

    # Render output
    if filtered_dict_V:
        fig = create_figure(filtered_dict_V)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Number of filtered drive cycles: {len(filtered_dict_V)}")
    else:
        st.warning("No drive cycles meet the specified criteria.")

if __name__ == "__main__":
    app()
