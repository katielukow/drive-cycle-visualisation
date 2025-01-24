import streamlit as st
import plotly.graph_objects as go

import pandas as pd
import numpy as np
from data_analysis import load_data, stats_calc, user_power_division, riding_events_power, charge_rate, user_stat, pybamm_plot


@st.cache_data
def prepare_data(filtered_dict_I, data_filtered, dc_all):
    for df in filtered_dict_I.values():
        df['Power'] = df['Voltage'] * df['Current']

    stats_all = stats_calc(filtered_dict_I)
    user_all = user_stat(dc_all)
    user_all['Date'] = pd.to_datetime(user_all['Date'])
    user_all['Day of Week'] = user_all['Date'].dt.day_name()

    day_behaviour = user_all[["Day of Week", "Drive Cycle ID", "Mean Power [W]"]].copy()
    day_behaviour['ID'] = np.where(day_behaviour['Mean Power [W]'] < 0, 'charge', 'discharge')

    stats_discharge = stats_all[stats_all["Mean Power [W]"] > 0]

    return day_behaviour, stats_all, user_all


def app():
    filtered_dict_I, data_filtered, dc_all = st.session_state.dc_all_fil, st.session_state.data_all, st.session_state.dc_all
    day_behaviour, stats_all, user_all = prepare_data(filtered_dict_I, data_filtered, dc_all)
    
    cycle_status = {key: 'charge' if (np.mean(df['Current']) <= 0) else 'discharge' for key, df in st.session_state.dc_all.items()}

    # Calculate statistics once
    charge_dict = {k: v for k, v in dc_all.items() if cycle_status[k] == 'charge'}
    discharge_dict = {k: v for k, v in dc_all.items() if cycle_status[k] == 'discharge' and k not in [3, 4]}
    # del discharge_dict[3] # remove the drive cycle with ID 3 as it is an outlier
    # del discharge_dict[4] # remove the drive cycle with ID 4 as it is an outlier
    
    Q_pack = 11

    discharge_fil = {key: df for key, df in discharge_dict.items() if ((df['Current'] >= -1)).all()}
    power_data = np.concatenate([discharge_fil[i]['Power'] for i in discharge_fil])
    current_data = np.concatenate([discharge_fil[i]['Current'] for i in discharge_fil])
    power_data = power_data[power_data > 0]
    current_data = current_data[current_data > 0]

    bins_I, hist_I = user_power_division(current_data, False)
    bins_P, hist_P = user_power_division(power_data, False) 

    # pwr_div = []
    pwr_div = [riding_events_power(dc_all[i], bins_P) for i in dc_all]
    dates = [dc_all[i].DateTime.iloc[0].date() for i in dc_all]
    pwr_df = pd.DataFrame(pwr_div, index=dc_all.keys(), columns=['P_high', 'P_mid', 'P_low', 'P_charge', 'P_total'])
    # power_division = pwr_df.sum(axis=0) / pwr_df.sum(axis=0).P_total
    pwr_percent = pwr_df.div(pwr_df['P_total'], axis=0)
    pwr_percent.drop(columns='P_total', inplace=True)

    pwr_discharge = pwr_percent[pwr_percent.index.isin(discharge_dict.keys())].drop(columns='P_charge')
    # pwr_charge = pwr_df[pwr_df.index.isin(charge_dict.keys())]

    stats_all = stats_calc(filtered_dict_I)

    # Preprocess the data
    pwr_df['Date'] = dates
    grouped_data = pwr_df.groupby('Date').sum()[['P_high', 'P_mid', 'P_low', 'P_charge']].reset_index()
    final_data = grouped_data.drop(columns=['Date'])
    final_data['Off'] = 24 * 3600 - final_data.sum(axis=1)

    # Aggregate data for the pie chart
    # discharge_value = sum_data['P_high'] + sum_data['P_mid'] + sum_data['P_low']

    st.write('### Discharge Usage Distribution')
    st.write('We can now begin to investigate the behaviours regarding the charge and discharge profiles closer. The first step of this is to split the power data into three behaviours: High Power, Medium Power, and Low Power. The below histogram presents all of the discharge power data separated into three bins based on the valleys of this data.')
    # Calculate proportions
    # sum_data_per = sum_data / sum_data.sum()

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=power_data, nbinsx=100))
    fig_hist.update_layout(
        title='Power Distribution',
        xaxis_title='Power [W]',
        yaxis_title='Counts',
        height=600,
    )
    # Add vertical line at x = -245
    fig_hist.add_shape(
        type="line",
        x0=bins_P[1][0], x1=bins_P[1][0],  # x-position of the vertical line
        y0=0, y1=100000,  # y-position (from 0 to max y-axis value)
        line=dict(color="red", width=2, dash="dash")  # Customize line appearance
    )
    fig_hist.add_shape(
        type="line",
        x0=bins_P[2][0], x1=bins_P[2][0],  # x-position of the vertical line
        y0=0, y1=100000,  # y-position (from 0 to max y-axis value)
        line=dict(color="red", width=2, dash="dash")  # Customize line appearance
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.write("Utilising these bins we can now evaluate the individual distribtions within each trip.")
    st.write(pwr_discharge)
    fig_pie_dist = go.Figure(data=[go.Pie(labels=['High Power', 'Medium Power', 'Low Power'], values=pwr_discharge.mean(axis=0))])
    fig_pie_dist.update_layout(
        title='Average Power Distribution',
        height=600,
    )
    st.plotly_chart(fig_pie_dist, use_container_width=True)

    # fig_hist = go.Figure()
    # fig_hist.add_trace(go.Histogram(
    #     x=pwr_discharge['P_high'], 
    #     histnorm='probability density',
    #     xbins=dict(
    #         size=0.01  # Bin width of 0.01
    #     ), name='High Power'
    # ))
    # fig_hist.add_trace(go.Histogram(
    #     x=pwr_discharge['P_mid'], 
    #     histnorm='probability density',
    #     xbins=dict(
    #         size=0.01  # Bin width of 0.01
    #     ), name='Medium Power'
    # ))
    # fig_hist.add_trace(go.Histogram(
    #     x=pwr_discharge['P_low'], 
    #     histnorm='probability density',
    #     xbins=dict(
    #         size=0.01  # Bin width of 0.01
    #     ), name='Low Power'
    # ))
    # fig_hist.update_layout(
    #     title='Power Distribution during Discharge',
    #     xaxis_title='Proportional Time in Power Category',
    #     yaxis_title='Probability Density',
    #     height=600,
    #     legend_title='Power Category',

    # )
    # st.plotly_chart(fig_hist, use_container_width=True)


    st.write("Combining this information with the preprocessing data, we can develop a stepped load profile to take in to account this power division. The following pybamm experiment definition presents this load profile.")

    I_charge = np.round(abs(charge_rate(charge_dict)/ Q_pack),1)
    step_per = pwr_discharge.mean(axis=0)
    t_total = 0.5*60*60
    t_h = int(step_per.P_high * t_total)
    t_m = int(step_per.P_mid * t_total)
    t_l = int(step_per.P_low * t_total)

    I_h = np.round(np.abs(np.mean(bins_I[0]))/Q_pack,2)
    I_m = np.round(np.abs(np.mean(bins_I[1]))/Q_pack,2)
    I_l = np.round(np.abs(np.mean(bins_I[2]))/Q_pack,2)

    no_rest_exp = ["Discharge at " + str(I_l) + "C for " + str(t_l) + " seconds or until 3 V",
                            "Discharge at " + str(I_m) + "C for " + str(t_m) + " seconds or until 3 V",
                            "Discharge at " + str(I_h) + "C for " + str(t_h) + " seconds or until 3 V",
                            "Charge at " + str(I_charge) + "C until 4.2 V",
                            "Hold at 4.2 V until 50 mA"
                            ]
    
    formatted_no_rest = ',\n        '.join(f'"{step}"' for step in no_rest_exp)

    st.write("Stepped Load Profile Without Rests:")
    st.markdown(f"""
    ```python
    pybamm.Experiment([(
        {formatted_no_rest}
    )])""")

    if st.button('Run PyBaMM Simulation for Stepped Profile without Rests:'):
        pybamm_plot(no_rest_exp)
    
    st.write("Stepped Load Profile With Rests:")
    t_total_all = (data_filtered['DateTime'].iloc[-1] - data_filtered['DateTime'].iloc[0]).total_seconds()
    t_charge = stats_all[stats_all.index.isin(discharge_dict.keys())]['Duration [s]'].sum()/t_total_all
    t_discharge = (stats_all[stats_all.index.isin(charge_dict.keys())]['Duration [s]'].sum())/t_total_all
    t_off = 1 - t_charge - t_discharge

    t_total = 24 * 3600
    t_dis = t_total * t_discharge
    t_h = int(step_per.P_high * t_dis)
    t_m = int(step_per.P_mid * t_dis)
    t_l = int(step_per.P_low * t_dis)
    t_r = int(t_off * t_total)
    
    rest_exp = [
        "Discharge at " + str(I_l) + "C for " + str(t_l) + "seconds or until 3 V",
        "Discharge at " + str(I_m) + "C for " + str(t_m) + "seconds or until 3 V",
        "Discharge at " + str(I_h) + "C for " + str(t_h) + "seconds or until 3 V",
        "Charge at " + str(I_charge) + "C until 4.2 V",
        "Hold at 4.2 V until 50 mA",
        "Rest for " + str(t_r) + " seconds (60 minute period)"]
    
    formatted_rest = ',\n        '.join(f'"{step}"' for step in rest_exp)
    st.markdown(f"""
        ```python
        pybamm.Experiment([(
            {formatted_rest}
        )])""")
    
    if st.button('Run PyBaMM Simulation for Stepped Profile with Rests:'):
        pybamm_plot(rest_exp)
    


if __name__ == "__main__":
    app()