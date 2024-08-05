import streamlit as st
import plotly.graph_objects as go
from data_analysis import load_data
import pandas as pd
import numpy as np


def app():
    st.title('Preconditioning')
    # dc_all_fil, data_all = load_data()

    # fig_all = go.Figure()
    # fig_all.add_trace(go.Scatter(x=data_all['DateTime'], y=data_all['Voltage'], mode='lines'))
    # fig_all.update_layout(
    #     title='Voltage vs Time for All Drive Cycles',
    #     xaxis_title='Time (hours)',
    #     yaxis_title='Voltage',
    #     legend_title='Drive Cycles',
    #     height=600,
    # )

    # st.plotly_chart(fig_all, use_container_width=True)

    # csv = data_all.to_csv(index=True)
    # # Create a download button
    # st.download_button(
    #     label="Download preconditioned data as CSV",
    #     data=csv,
    #     file_name='data_all.csv',
    #     mime='text/csv'
    # )

    


if __name__ == "__main__":
    app()