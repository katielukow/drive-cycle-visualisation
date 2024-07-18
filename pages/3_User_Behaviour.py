import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc
from data_analysis import *

dc_all_fil = load_data()
for i in dc_all_fil:
    dc_all_fil[i]['Power'] = dc_all_fil[i]['Voltage'] * dc_all_fil[i]['Current']

stats_all = stats_calc(dc_all_fil)

stats_all['Date'] = pd.to_datetime(stats_all['Date'])
stats_all['Day of Week'] = stats_all['Date'].dt.day_name()

day_behaviour = stats_all[["Day of Week", "Drive Cycle ID", "Mean Current [A]"]].copy()
day_behaviour['ID'] = day_behaviour['Mean Current [A]'].apply(lambda x: 'charge' if x > 0 else 'discharge')

stats_discharge = stats_all[stats_all["Mean Power [W]"] < 0]
dis_dc = stats_discharge['Drive Cycle ID'].unique()
blue_scale = pc.sequential.Blues
num_cycles = len(dis_dc)



df_temps = stats_discharge[['Drive Cycle ID', 'High_P', 'Medium_P', 'Low_P']]

new_dfs = []
for i, row in df_temps.iterrows():
    # if row['High'] > 0:
        new_dfs.append({
            'Drive Cycle ID': row['Drive Cycle ID'],
            'Load': row['High_P'],
            'Load Type': 'High_P',
            # 'Cluster': row['Cluster']
        })
    # if row['Medium'] > 0:
        new_dfs.append({
            'Drive Cycle ID': row['Drive Cycle ID'],
            'Load': row['Medium_P'],
            'Load Type': 'Medium_P',
            # 'Cluster': row['Cluster']
        })
    # if row['Low'] > 0:
        new_dfs.append({
            'Drive Cycle ID': row['Drive Cycle ID'],
            'Load': row['Low_P'],
            'Load Type': 'Low_P',
            # 'Cluster': row['Cluster']
        })

violin_df = pd.DataFrame(new_dfs)

def app():
    st.write('## User Behaviour')

    st.write('### Number of Drive Cycles by Day')
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=day_behaviour['Day of Week'], name='Charge', histfunc='count', nbinsx=7))
    fig1.add_trace(go.Histogram(x=day_behaviour['Day of Week'], name='Discharge', histfunc='count', nbinsx=7, histnorm='percent'))
    st.plotly_chart(fig1)

    st.write('### User Driving Behaviour based on power drawn')
    fig3 = go.Figure()

    for i, dc in enumerate(dis_dc):
        # Calculate the color index based on the position in the list
        color_index = int((i / (num_cycles - 1)) * (len(blue_scale) - 1))
        
        # Add trace with blue color and transparency
        fig3.add_trace(go.Histogram(
            x=dc_all_fil[dc]['Power'],
            name=f'Drive Cycle {dc}',
            histnorm='percent',
            marker_color=blue_scale[color_index],
            opacity=0.2  # Adjust this value for desired transparency (0.0 to 1.0)
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
