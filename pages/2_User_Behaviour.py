import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc
import pandas as pd
import numpy as np
from data_analysis import load_data, stats_calc

@st.cache_data
def prepare_data():
    dc_all_fil, data_filtered = load_data()

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

    return day_behaviour, violin_df, stats_all

def weekly_count(stats_all):
    # Convert dates to week numbers
    stats_all['week_number'] = pd.to_datetime(stats_all['Date']).dt.isocalendar().week

    # Create a mapping of weekdays to their corresponding index in days_data
    weekday_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
                'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    # Map weekdays to their corresponding indices
    stats_all['weekday_index'] = stats_all['Day of Week'].map(weekday_map)

    # Initialize the days_data array
    days_data = np.zeros((52, 7), dtype=int)

    # Fill in days_data
    for week_number, group in stats_all.groupby('week_number'):
        unique_indices = group['weekday_index'].unique()
        days_data[week_number - 1, unique_indices] = 1

    # Compute the mean of the sum across rows
    mean_value = np.mean(np.sum(days_data, axis=1))
    return mean_value

def app():
    day_behaviour, violin_df, stats_all = prepare_data()

    st.write('## User Behaviour')

    st.write('### Number of Drive Cycles by Day')
    fig1 = go.Figure()
    for id_type in ['charge', 'discharge']:
        subset = day_behaviour[day_behaviour['ID'] == id_type]
        fig1.add_trace(go.Histogram(x=subset['Day of Week'], name=id_type.capitalize(), histfunc='count', nbinsx=7))
    st.plotly_chart(fig1)

    mean_days = weekly_count(stats_all)

    st.write(f'The average number of drive cycles per week is {mean_days:.2f}')

    st.write('### Power Distribution for Discharge Drive Cycles')
    fig2 = go.Figure()
    fig2.add_trace(go.Violin(x=violin_df['Load Type'],
                            y=violin_df['Load']
                ))
    fig2.update_traces(meanline_visible=True)
    fig2.update_layout(violinmode='group')
    st.plotly_chart(fig2)

    
    mean_behaviour = stats_all[["Mean Current [A]",	"Energy [Wh]",	"Mean Power [W]",	"Capacity [Ah]",	"Max Current [A]",	"Max Power [W]"]].mean()
    st.dataframe(mean_behaviour)

    # # Calculate absolute differences from mean values
    # differences = (data_c1[columns_to_use] - means).abs()
    # normal_diffs = differences / means.abs()



if __name__ == "__main__":
    app()