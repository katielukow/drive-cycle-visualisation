import streamlit as st
import plotly.graph_objects as go
from data_analysis import load_data

def app():
    # Load data (this will retrieve the cached data)
    dc_all_fil = load_data()
    
    # Your page-specific logic here
    # st.write("Page 1")
    # st.write(df_all.head())

    # Ensure unique keys for any widgets
    charge = st.sidebar.checkbox("Charge", True, key="page1_charge")
    discharge = st.sidebar.checkbox("Discharge", True, key="page1_discharge")
    
    if charge and discharge:
        # Include all data if both are selected
        filtered_dict_V = dc_all_fil
    elif charge:
        # Filter for charge data
        filtered_dict_V = {key: df for key, df in dc_all_fil.items() if (df['Current'] >= 0).all()}
    elif discharge:
        # Filter for discharge data
        filtered_dict_V = {key: df for key, df in dc_all_fil.items() if (df['Current'] <= 0).all()}
    else:
        # Handle case when neither is selected (you can customize this as needed)
        filtered_dict_V = {}

    if filtered_dict_V:
        fig = go.Figure()
        for key, df in filtered_dict_V.items():
            fig.add_trace(go.Scatter(x=(df['DateTime'] - df['DateTime'].iloc[0]).dt.total_seconds() / 3600, y=df['Voltage'], mode='lines', name=f'Drive Cycle {key}'))

        fig.update_layout(
            title='Voltage vs Time for Filtered Drive Cycles',
            xaxis_title='Time (hours)',
            yaxis_title='Voltage',
            legend_title='Drive Cycles',
            height=600,
        )

        # Set the lower limit of x-axis (example: start from -5 hours)
        fig.update_xaxes(range=[-5, None])

        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Number of filtered drive cycles: {len(filtered_dict_V)}")
    else:
        st.write("No drive cycles meet the specified criteria.")
