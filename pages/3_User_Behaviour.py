import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc
import pandas as pd
import numpy as np
from data_analysis import load_data, stats_calc

@st.cache_data
def prepare_data():
    dc_all_fil = load_data()
    for df in dc_all_fil.values():
        df['Power'] = df['Voltage'] * df['Current']

    stats_all = stats_calc(dc_all_fil)
    stats_all['Date'] = pd.to_datetime(stats_all['Date'])
    stats_all['Day of Week'] = stats_all['Date'].dt.day_name()

    day_behaviour = stats_all[["Day of Week", "Drive Cycle ID", "Mean Current [A]"]].copy()
    day_behaviour['ID'] = np.where(day_behaviour['Mean Current [A]'] > 0, 'charge', 'discharge')

    stats_discharge = stats_all[stats_all["Mean Power [W]"] < 0]
    dis_dc = stats_discharge['Drive Cycle ID'].unique()

    df_temps = stats_discharge[['Drive Cycle ID', 'High_P', 'Medium_P', 'Low_P']]
    violin_df = pd.melt(df_temps, id_vars=['Drive Cycle ID'], 
                        value_vars=['High_P', 'Medium_P', 'Low_P'],
                        var_name='Load Type', value_name='Load')

    return day_behaviour, dis_dc, dc_all_fil, violin_df

def app():
    day_behaviour, dis_dc, dc_all_fil, violin_df = prepare_data()

    st.write('## User Behaviour')

    st.write('### Number of Drive Cycles by Day')
    fig1 = go.Figure()
    for id_type in ['charge', 'discharge']:
        subset = day_behaviour[day_behaviour['ID'] == id_type]
        fig1.add_trace(go.Histogram(x=subset['Day of Week'], name=id_type.capitalize(), histfunc='count', nbinsx=7))
    st.plotly_chart(fig1)

    st.write('### User Driving Behaviour based on power drawn')
    fig3 = go.Figure()

    blue_scale = pc.sequential.Blues
    num_cycles = len(dis_dc)

    for i, dc in enumerate(dis_dc):
        color_index = int((i / (num_cycles - 1)) * (len(blue_scale) - 1))
        fig3.add_trace(go.Histogram(
            x=dc_all_fil[dc]['Power'],
            name=f'Drive Cycle {dc}',
            histnorm='percent',
            marker_color=blue_scale[color_index],
            opacity=0.2
        ))
    fig3.update_layout(barmode='overlay')
    st.plotly_chart(fig3)

    fig2 = go.Figure()
    fig2.add_trace(go.Violin(x=violin_df['Load Type'],
                            y=violin_df['Load']
                ))
    fig2.update_traces(meanline_visible=True)
    fig2.update_layout(violinmode='group')
    st.plotly_chart(fig2)

if __name__ == "__main__":
    app()