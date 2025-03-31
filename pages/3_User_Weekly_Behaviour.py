import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pybamm
from plotly.subplots import make_subplots
from data_analysis import load_data, stats_calc, user_power_division, riding_events_power, user_stat, charge_rate

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

    # df_temps = stats_discharge[['Drive Cycle ID', 'High_P', 'Medium_P', 'Low_P']]
    # violin_df = pd.melt(df_temps, id_vars=['Drive Cycle ID'], 
    #                     value_vars=['High_P', 'Medium_P', 'Low_P'],
    #                     var_name='Load Type', value_name='Load')

    return filtered_dict_I, day_behaviour, stats_all, user_all

def weekly_counts(df):
    # df['year'] = pd.to_datetime(df['Date']).dt.year
    unique_dates_per_week = df.groupby(['year', 'week_number'])['Date'].nunique().reset_index(name='unique_date_count')

    min_year = unique_dates_per_week['year'].min()
    max_year = unique_dates_per_week['year'].max()
    min_week = unique_dates_per_week[unique_dates_per_week['year'] == min_year]['week_number'].min()
    max_week = unique_dates_per_week[unique_dates_per_week['year'] == max_year]['week_number'].max()

    # Create a DataFrame with all year-week combinations from the minimum to maximum range
    full_weeks = pd.DataFrame([
        (year, week)
        for year in range(min_year, max_year + 1)
        for week in range(1, 53)  # Weeks 1 to 52
        if not (year == min_year and week < min_week) and not (year == max_year and week > max_week)
    ], columns=['year', 'week_number'])

    # Merge with the unique dates count, filling missing values with 0
    complete_weeks = full_weeks.merge(unique_dates_per_week, on=['year', 'week_number'], how='left').fillna(0)

    # Convert unique_date_count to integer for clarity
    complete_weeks['unique_date_count'] = complete_weeks['unique_date_count'].astype(int)
    if df['Mean Power [W]'].mean() < 0:
        return complete_weeks, int(complete_weeks.unique_date_count.mean())
    else:
        return complete_weeks, int(complete_weeks[complete_weeks['unique_date_count'] != 0].unique_date_count.mean())

def mean_cycles(filtered_dict_I):
    cycle_status = {key: 'charge' if (df['Current'] <= 0).all() else 'discharge' for key, df in filtered_dict_I.items()}
    discharge_dict = {k: v for k, v in filtered_dict_I.items() if cycle_status[k] == 'discharge'}

    stats_discharge = stats_calc(discharge_dict)

    E_discharge = np.mean(stats_discharge["Energy [Wh]"])
    stats_discharge["mean difference"] = (stats_discharge["Energy [Wh]"]-E_discharge)
    discharge_ID = stats_discharge.iloc[(stats_discharge["mean difference"].abs()).idxmin()]["Drive Cycle ID"]

    discharge_df = filtered_dict_I[discharge_ID][["DateTime", "Current", "Voltage"]]
    discharge_df["Time"] = (discharge_df["DateTime"] - discharge_df["DateTime"].iloc[0]).dt.total_seconds()

    return  discharge_df

def create_commute_experiment(commute_days):
    if commute_days > 6:
        raise ValueError("The number of commute days cannot exceed 6")
    # Ensure the number of commute days does not exceed 6
    commute_days = min(commute_days, 6)
    subcycle_commute = ["*subcycle_commute"]
    subcycle_rest = ["*subcycle_rest"]
    subcycle_charge = ["*subcycle_charge"]

    # Start with an empty list to hold the sequence of subcycles
    commute = []

    # Append subcycle_commute and subcycle_rest for each commute day
    if commute_days < 1:
        for _ in range(commute_days):
            commute.extend(subcycle_commute)
            commute.extend(subcycle_rest)
        remaining_days = 7 - commute_days * 2 - 1
        commute.extend(subcycle_rest * remaining_days)

    else:
        for _ in range(commute_days):
            commute.extend(subcycle_commute)
        remaining_days = 7 - commute_days - 1
        commute.extend(subcycle_rest * remaining_days)

    commute.extend(subcycle_charge)

    commute_str = str(commute).replace("[", "(").replace("]", ")").replace("'", "")
    return commute_str

def app():
    st.title('User Behaviour')
    # dc_all_fil, data_all, dc_all = load_data()
    dc_all_fil, data_all, dc_all = st.session_state.dc_all_fil, st.session_state.data_all, st.session_state.dc_all
    cycle_status = {key: 'charge' if (np.mean(df['Current']) <= 0) else 'discharge' for key, df in dc_all.items()}

    charge_dict = {k: v for k, v in dc_all.items() if cycle_status[k] == 'charge'}
    discharge_dict = {k: v for k, v in dc_all.items() if cycle_status[k] == 'discharge' and k not in [3, 4]}
    
    Q_pack = 11

    discharge_fil = {key: df for key, df in discharge_dict.items() if ((df['Current'] >= -1)).all()}
    power_data = np.concatenate([discharge_fil[i]['Power'] for i in discharge_fil])
    current_data = np.concatenate([discharge_fil[i]['Current'] for i in discharge_fil])
    power_data = power_data[power_data > 0]
    current_data = current_data[current_data > 0]

    bins_I, hist_I = user_power_division(current_data, False)
    bins_P, hist_P = user_power_division(power_data, False) 

    pwr_div = []
    pwr_div = [riding_events_power(dc_all[i], bins_P) for i in dc_all]
    dates = [dc_all[i].DateTime.iloc[0].date() for i in dc_all]
    pwr_df = pd.DataFrame(pwr_div, index=dc_all.keys(), columns=['P_high', 'P_mid', 'P_low', 'P_charge', 'P_total'])
    pwr_percent = pwr_df.div(pwr_df['P_total'], axis=0)
    pwr_percent.drop(columns='P_total', inplace=True)

    pwr_discharge = pwr_percent[pwr_percent.index.isin(discharge_dict.keys())].drop(columns='P_charge')

    stats_all = stats_calc(dc_all_fil)

    # Preprocess the data
    pwr_df['Date'] = dates
    grouped_data = pwr_df.groupby('Date').sum()[['P_high', 'P_mid', 'P_low', 'P_charge']].reset_index()
    final_data = grouped_data.drop(columns=['Date'])
    final_data['Off'] = 24 * 3600 - final_data.sum(axis=1)

    st.write('### Discharge Usage Distribution')
    st.write('We can now start to incorporate some user behaviours into the load profile to better represent the real world usage of the system. First we investitage the number of days the asset is used per week in both the discharge and charge areas.')

    stats_all = stats_calc(dc_all)
    stats_all['Date'] = pd.to_datetime(stats_all['Date'])
    stats_all['Day of Week'] = stats_all['Date'].dt.day_name()
    stats_all['week_number'] = pd.to_datetime(stats_all['Date']).dt.isocalendar().week

    # Create a mapping of weekdays to their corresponding index in days_data
    weekday_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
                'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    # Map weekdays to their corresponding indices
    stats_all['weekday_index'] = stats_all['Day of Week'].map(weekday_map)
    stats_all['year'] = pd.to_datetime(stats_all['Date']).dt.year

    user_all_discharge = stats_all[stats_all.index.isin(discharge_dict.keys())]
    discharge_hist, mean_discharge = weekly_counts(user_all_discharge)

    discharge_df = mean_cycles(dc_all)
    st.write("The average number of days the asset is used for discharging is", mean_discharge,". We take the mean only from weeks where the asset is used.")
    commuting_days = st.slider("Number of Commute Days", 1, 6, mean_discharge)
    
    I_charge = np.round(abs(charge_rate(charge_dict)/ Q_pack),1)

    step_per = pwr_discharge.mean(axis=0)
    t_total = 500
    t_h = int(st.session_state.t_h / commuting_days / 2)
    t_m = int(st.session_state.t_m / commuting_days / 2)
    t_l = int(st.session_state.t_l / commuting_days / 2)

    I_h = np.round(np.abs(np.mean(bins_I[2]))/Q_pack,2)
    I_m = np.round(np.abs(np.mean(bins_I[1]))/Q_pack,2)
    I_l = np.round(np.abs(np.mean(bins_I[0]))/Q_pack,2)


    fig_days = go.Figure()
    fig_days.add_trace(go.Histogram(
        x=discharge_hist['unique_date_count'],
        nbinsx=8,
        xbins=dict(start=-0.5, end=7.5, size=1),
        marker=dict(line=dict(color='white', width=3))
    ))
    fig_days.update_layout(
        title='Discharge Days per Week',
        xaxis_title='Number of Discharge Days',
        yaxis_title='Count',  # Update y-axis title to 'Count'
        height=600,
    )

    st.plotly_chart(fig_days, use_container_width=True)

    st.write("The first experiment we create is a weekly representative load profile with the stepped load from before.")

    subcycle_commute = ["Rest for 8 hours (30 minute period)", 
                        "Discharge at " + str(I_l) + "C for " + str(t_l) + " seconds or until 2.5 V",
                        "Discharge at " + str(I_m) + "C for " + str(t_m) + " seconds or until 2.5 V",
                        "Discharge at " + str(I_h) + "C for " + str(t_h) + " seconds or until 2.5 V", 
                        "Rest for 8 hours (30 minute period)", 
                        "Discharge at " + str(I_l) + "C for " + str(t_l) + " seconds or until 2.5 V",
                        "Discharge at " + str(I_m) + "C for " + str(t_m) + " seconds or until 2.5 V",
                        "Discharge at " + str(I_h) + "C for " + str(t_h) + " seconds or until 2.5 V", 
                        "Rest for 8 hours (30 minute period)"
                        ]

    subcycle_charge = ["Rest for 12 hours (30 minute period)", 
                    "Charge at " + str(I_charge) + "C until 4.2 V",
                    "Hold at 4.2 V until 50 mA",
                    "Rest for 10 hours (30 minute period)"
                    ]

    subcycle_rest = ["Rest for 24 hours (60 minute period)",]
    commute = create_commute_experiment(commuting_days)

    st.markdown(f'''
    ```python
    subcycle_commute = {subcycle_commute}

    subcycle_charge = ["Rest for 12 hours (30 minute period)", 
                    "Charge at {I_charge}C until 4.2 V",
                    "Hold at 4.2 V until 50 mA",
                    "Rest for 10 hours (30 minute period)"
                    ]

    subcycle_rest = {subcycle_rest}

    commute = {commute}

    exp = pybamm.Experiment([commute])
    ''')



    # Add a button
    if st.button('Run PyBaMM Simulation'):


        commute = (*subcycle_commute, *subcycle_commute, *subcycle_commute, *subcycle_commute, *subcycle_rest, *subcycle_rest, *subcycle_charge)
        exp = pybamm.Experiment([commute])

        model = pybamm.lithium_ion.SPM()
        parameters = pybamm.ParameterValues("OKane2022")
        sim = pybamm.Simulation(model, parameter_values = parameters, experiment=exp, solver=pybamm.IDAKLUSolver())

        sol = sim.solve()
        time = sol["Time [s]"].entries
        current = sol["Current [A]"].entries
        voltage = sol["Terminal voltage [V]"].entries

        fig = make_subplots(rows=2, cols=1, subplot_titles=('Current Over Time', 'Voltage Over Time'))

        # Left subplot: Current
        fig.add_trace(go.Scatter(x=time, y=current, mode='lines', name='Current [A]', line=dict(color='blue')), row=1, col=1)

        # Right subplot: Voltage
        fig.add_trace(go.Scatter(x=time, y=voltage, mode='lines', name='Voltage [V]', line=dict(color='red')), row=2, col=1)

        # Update layout for the figure
        fig.update_layout(
            title_text='Current and Voltage Over Time',
            xaxis_title_text='Time [s]',
            height=600,  # Adjust the height to fit the subplots
            showlegend=False
        )

        # Update axis labels for the individual subplots
        fig.update_xaxes(title_text="Time [s]", row=1, col=1)
        fig.update_yaxes(title_text="Current [A]", row=1, col=1)
        fig.update_xaxes(title_text="Time [s]", row=2, col=1)
        fig.update_yaxes(title_text="Voltage [V]", row=2, col=1)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    
    st.write("The load profile can be taken one step further where a dynamic load profile can be extracted from the raw data and replace the stepped profile. Initially, a load profile is taken from the selection as representative through the energy used. Mean energy is calculated and the drive cycle with a energy clsoest to the mean is selected. The next page will further investigate the selection of this drive cycle and other methods to determine the representivity of the load profile.")
    
    discharge_df['Power'] = (discharge_df['Voltage']/10)*(discharge_df['Current']/4) # Power = V*I (W). Negative sign to match pybamm syntax. 
    data_temp = discharge_df[['Time', 'Current']]
    # data_temp['Power'].iloc[0]  = 0

    dT = 2.3
    normal_time = np.arange(0, data_temp['Time'].max(), dT)
    power_interp = np.interp(normal_time, data_temp['Time'], data_temp['Current']/4/2.5*2) # multiply power by two to account for the used cell for modelling having being 5 Ah instead of 2.5 Ah
    normalized_current = pd.DataFrame({'Time': normal_time, 'Current': power_interp})
    # Convert DataFrame directly to CSV format
    csv_data = normalized_current.to_csv(index=False)

    # Create the download button
    st.download_button(
        label="Download Normalized Current Data",
        data=csv_data,
        file_name='normalized_current.csv',
        mime='text/csv'
    )

    step_P = pybamm.step.current(value=normalized_current.values, duration=902, termination="3.0 V")

    subcycle_commute = ["Rest for 8 hours (30 minute period)", 
                        step_P, 
                        "Rest for 8 hours (30 minute period)", 
                        step_P,
                        "Rest for 8 hours (30 minute period)"
                        ]

    st.markdown(f'''
    ```python
    normalized_power = pd.read_csv("normalized_power.csv")
      
    step_P = pybamm.step.power(value=normalized_power.values, duration="720 seconds", termination="3.0 V")
    
    subcycle_commute = ["Rest for 8 hours (30 minute period)", 
                        step_P, 
                        "Rest for 8 hours (30 minute period)", 
                        step_P,
                        "Rest for 8 hours (30 minute period)"
                        ]

    subcycle_charge = {subcycle_charge}

    subcycle_rest = {subcycle_rest}

    commute = {commute}

    exp = pybamm.Experiment([commute])
    ''')

    if st.button('Run PyBaMM Simulation with Normalized Power Data'):

        model = pybamm.lithium_ion.SPM()
        parameters = pybamm.ParameterValues("OKane2022")
        sim = pybamm.Simulation(model=model, parameter_values=parameters, experiment=pybamm.Experiment(subcycle_commute*commuting_days + subcycle_rest * (7-commuting_days-1)+subcycle_charge), solver=pybamm.IDAKLUSolver())

        sol = sim.solve()
        time = sol["Time [s]"].entries
        current = sol["Current [A]"].entries
        voltage = sol["Terminal voltage [V]"].entries

        fig = make_subplots(rows=2, cols=1, subplot_titles=('Current Over Time', 'Voltage Over Time'))

        # Left subplot: Current
        fig.add_trace(go.Scatter(x=time, y=current, mode='lines', name='Current [A]', line=dict(color='blue')), row=1, col=1)

        # Right subplot: Voltage
        fig.add_trace(go.Scatter(x=time, y=voltage, mode='lines', name='Voltage [V]', line=dict(color='red')), row=2, col=1)

        # Update layout for the figure
        fig.update_layout(
            title_text='Current and Voltage Over Time',
            xaxis_title_text='Time [s]',
            height=600,  # Adjust the height to fit the subplots
            showlegend=False
        )

        # Update axis labels for the individual subplots
        fig.update_xaxes(title_text="Time [s]", row=1, col=1)
        fig.update_yaxes(title_text="Current [A]", row=1, col=1)
        fig.update_xaxes(title_text="Time [s]", row=2, col=1)
        fig.update_yaxes(title_text="Voltage [V]", row=2, col=1)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    app()