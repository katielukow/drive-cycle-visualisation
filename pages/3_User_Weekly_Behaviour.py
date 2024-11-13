import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pybamm
from plotly.subplots import make_subplots
from data_analysis import load_data, stats_calc, user_power_division, riding_events_power, user_stat, charge_rate

@st.cache_data
def prepare_data():
    filtered_dict_I, data_filtered, dc_all = load_data()

    for df in filtered_dict_I.values():
        df['Power'] = df['Voltage'] * df['Current']

    stats_all = stats_calc(filtered_dict_I)
    # stats_all['Date'] = pd.to_datetime(stats_all['Date'])
    # stats_all['Day of Week'] = stats_all['Date'].dt.day_name()

    user_all = user_stat(dc_all)
    user_all['Date'] = pd.to_datetime(user_all['Date'])
    user_all['Day of Week'] = user_all['Date'].dt.day_name()

    day_behaviour = user_all[["Day of Week", "Drive Cycle ID", "Mean Power [W]"]].copy()
    day_behaviour['ID'] = np.where(day_behaviour['Mean Power [W]'] < 0, 'charge', 'discharge')

    stats_discharge = stats_all[stats_all["Mean Power [W]"] > 0]

    df_temps = stats_discharge[['Drive Cycle ID', 'High_P', 'Medium_P', 'Low_P']]
    violin_df = pd.melt(df_temps, id_vars=['Drive Cycle ID'], 
                        value_vars=['High_P', 'Medium_P', 'Low_P'],
                        var_name='Load Type', value_name='Load')

    return filtered_dict_I, day_behaviour, violin_df, stats_all, user_all

def weekly_count(user_all):
    # Convert dates to week numbers
    user_all['week_number'] = pd.to_datetime(user_all['Date']).dt.isocalendar().week

    # Create a mapping of weekdays to their corresponding index in days_data
    weekday_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
                'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    # Map weekdays to their corresponding indices
    user_all['weekday_index'] = user_all['Day of Week'].map(weekday_map)

    user_all_charge = user_all[user_all['Status'] == 1]
    user_all_discharge = user_all[user_all['Status'] == 0]

    # Initialize the days_data array
    days_data_charge = np.zeros((52, 7), dtype=int)
    days_data_discharge = np.zeros((52,7), dtype=int)

    # Fill in days_data 
    for week_number, group in user_all_charge.groupby('week_number'):
        unique_indices = group['weekday_index'].unique()
        days_data_charge[week_number - 1, unique_indices] = 1

    for week_number, group in user_all_discharge.groupby('week_number'):
        unique_indices = group['weekday_index'].unique()
        days_data_discharge[week_number - 1, unique_indices] = 1

    # Compute the mean of the sum across rows
    mean_charge = round(np.mean(np.sum(days_data_charge, axis=1)))
    mean_discharge = round(np.mean(np.sum(days_data_discharge, axis=1)))

    return mean_charge, mean_discharge

def mean_cycles(filtered_dict_I):
    cycle_status = {key: 'charge' if (df['Current'] <= 0).all() else 'discharge' for key, df in filtered_dict_I.items()}
    charge_dict = {k: v for k, v in filtered_dict_I.items() if cycle_status[k] == 'charge'}
    discharge_dict = {k: v for k, v in filtered_dict_I.items() if cycle_status[k] == 'discharge'}
    stats_charge = stats_calc(charge_dict)
    stats_discharge = stats_calc(discharge_dict)

    E_charge = np.mean(stats_charge["Energy [Wh]"])
    stats_charge["mean difference"] = (stats_charge["Energy [Wh]"]-E_charge)
    charge_ID = stats_charge.iloc[(stats_charge["mean difference"].abs()).idxmin()]["Drive Cycle ID"]

    charge_df = filtered_dict_I[charge_ID][["DateTime", "Current"]]
    charge_df["Time"] = (charge_df["DateTime"] - charge_df["DateTime"].iloc[0]).dt.total_seconds()

    E_discharge = np.mean(stats_discharge["Energy [Wh]"])
    stats_discharge["mean difference"] = (stats_discharge["Energy [Wh]"]-E_discharge)
    discharge_ID = stats_discharge.iloc[(stats_discharge["mean difference"].abs()).idxmin()]["Drive Cycle ID"]

    discharge_df = filtered_dict_I[discharge_ID][["DateTime", "Current"]]
    discharge_df["Time"] = (discharge_df["DateTime"] - discharge_df["DateTime"].iloc[0]).dt.total_seconds()

    return charge_df, discharge_df

def create_commute_experiment(commute_days):
    if commute_days > 6:
        raise ValueError("The number of commute days cannot exceed 6")
    # Ensure the number of commute days does not exceed 6
    commute_days = min(commute_days, 6)
    subcycle_commuteP = ["*subcycle_commuteP"]
    subcycle_rest = ["*subcycle_rest"]
    subcycle_charge = ["*subcycle_charge"]

    # Start with an empty list to hold the sequence of subcycles
    commuteP = []

    # Append subcycle_commuteP and subcycle_rest for each commute day
    if commute_days < 4:
        for _ in range(commute_days):
            commuteP.extend(subcycle_commuteP)
            commuteP.extend(subcycle_rest)
        remaining_days = 7 - commute_days * 2 - 1
        commuteP.extend(subcycle_rest * remaining_days)

    else:
        for _ in range(commute_days):
            commuteP.extend(subcycle_commuteP)
        remaining_days = 7 - commute_days - 1
        commuteP.extend(subcycle_rest * remaining_days)
    
    commuteP.extend(subcycle_charge)

    commuteP_str = str(commuteP).replace("[", "(").replace("]", ")").replace("'", "")
    return commuteP_str

def load_profile(charge_df, discharge_df, mean_charge, mean_discharge):
    # Define the length of a day in seconds
    seconds_in_day = 24 * 3600

    # User-defined values
    charge_days = mean_charge
    discharge_days = mean_discharge

    # Calculate the number of rest days
    rest_days = 7 - (charge_days + discharge_days)

    # Generate time and current for rest day
    rest_time = np.arange(0, seconds_in_day, 60)
    rest_current = np.zeros(len(rest_time))
    rest_day = np.column_stack((rest_time, rest_current))

    # Generate time and current for discharge day
    discharge = discharge_df[["Time", "Current"]].values
    discharge_time = np.arange(np.ceil(discharge[-1][0]), seconds_in_day, 60)
    discharge_rest = np.zeros(len(discharge_time))
    discharge_combined_time = np.concatenate((discharge[:, 0], discharge_time))
    discharge_combined_current = np.concatenate((discharge[:, 1], discharge_rest))
    discharge_day = np.column_stack((discharge_combined_time, discharge_combined_current))

    # Generate time and current for charge day
    charge = charge_df[["Time", "Current"]].values
    charge_time = np.arange(np.ceil(charge[-1][0]), seconds_in_day, 60)
    charge_rest = np.zeros(len(charge_time))
    charge_combined_time = np.concatenate((charge[:, 0], charge_time))
    charge_combined_current = np.concatenate((charge[:, 1], charge_rest))
    charge_day = np.column_stack((charge_combined_time, charge_combined_current))

    # Initialize an empty list to store all days
    all_days = []

    # Add discharge days to the list
    for i in range(discharge_days):
        discharge_copy = discharge_day.copy()
        discharge_copy[:, 0] += i * seconds_in_day
        all_days.append(discharge_copy)

    # Add charge days to the list
    for i in range(charge_days):
        charge_copy = charge_day.copy()
        charge_copy[:, 0] += (discharge_days + i) * seconds_in_day
        all_days.append(charge_copy)

    # Add rest days to the list
    for i in range(rest_days):
        rest_copy = rest_day.copy()
        rest_copy[:, 0] += (discharge_days + charge_days + i) * seconds_in_day
        all_days.append(rest_copy)

    # Combine all days into a single array
    combined_days = np.vstack(all_days)

    # Convert to DataFrame
    combined_df = pd.DataFrame(combined_days, columns=["Time", "Current"])

    return combined_df


def app():
    st.title('User Behaviour')
    dc_all_fil, data_all, dc_all = load_data()
    cycle_status = {key: 'charge' if (np.mean(df['Current']) <= 0) else 'discharge' for key, df in dc_all.items()}

    # Calculate statistics once
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
    power_division = pwr_df.sum(axis=0) / pwr_df.sum(axis=0).P_total
    pwr_percent = pwr_df.div(pwr_df['P_total'], axis=0)
    pwr_percent.drop(columns='P_total', inplace=True)

    pwr_discharge = pwr_percent[pwr_percent.index.isin(discharge_dict.keys())].drop(columns='P_charge')
    pwr_charge = pwr_df[pwr_df.index.isin(charge_dict.keys())]

    stats_all = stats_calc(dc_all_fil)

    # Preprocess the data
    pwr_df['Date'] = dates
    grouped_data = pwr_df.groupby('Date').sum()[['P_high', 'P_mid', 'P_low', 'P_charge']].reset_index()
    final_data = grouped_data.drop(columns=['Date'])
    final_data['Off'] = 24 * 3600 - final_data.sum(axis=1)

    sum_data = final_data.sum(axis=0)

    # Aggregate data for the pie chart
    discharge_value = sum_data['P_high'] + sum_data['P_mid'] + sum_data['P_low']

    st.write('### Discharge Usage Distribution')
    st.write('We can now start to incorporate some user behaviours into the load profile to better represent the real world usage of the system. First we investitage the number of days the asset is used per week in both the discharge and charge areas.')
    # Calculate proportions
    sum_data_per = sum_data / sum_data.sum()

    stats_all = stats_calc(dc_all)
    stats_all['Date'] = pd.to_datetime(stats_all['Date'])
    stats_all['Day of Week'] = stats_all['Date'].dt.day_name()
    stats_all['week_number'] = pd.to_datetime(stats_all['Date']).dt.isocalendar().week

    # Create a mapping of weekdays to their corresponding index in days_data
    weekday_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
                'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    # Map weekdays to their corresponding indices
    stats_all['weekday_index'] = stats_all['Day of Week'].map(weekday_map)

    user_all_charge = stats_all[stats_all.index.isin(charge_dict.keys())]
    user_all_discharge = stats_all[stats_all.index.isin(discharge_dict.keys())]

    start_date = pd.Timestamp('2023-10-01').week
    end_date = pd.Timestamp('2023-12-01').week
    days_data_charge = np.zeros((52, 7), dtype=int)
    days_data_discharge = np.zeros((52,7), dtype=int)

    # Fill in days_data
    for week_number, group in user_all_charge.groupby('week_number'):
        unique_indices = group['weekday_index'].unique()
        days_data_charge[week_number - 1, unique_indices] = 1

    for week_number, group in user_all_discharge.groupby('week_number'):
        unique_indices = group['weekday_index'].unique()
        days_data_discharge[week_number - 1, unique_indices] = 1

    days_data_discharge = days_data_discharge[start_date:end_date]
    days_data_charge = days_data_charge[start_date:end_date]
    days_discharge = np.sum(days_data_discharge, axis=1)
    mean_discharge = round(np.mean(days_discharge[days_discharge != 0]))

    stats_charge = stats_calc(charge_dict)
    stats_discharge = stats_calc(discharge_dict)

    E_charge = np.mean(stats_charge["Energy [Wh]"])
    stats_charge["mean difference"] = (stats_charge["Energy [Wh]"]-E_charge)
    charge_ID = stats_charge.iloc[(stats_charge["mean difference"].abs()).idxmin()]["Drive Cycle ID"]

    charge_df = dc_all[charge_ID][["DateTime", "Current"]]
    charge_df["Time"] = (charge_df["DateTime"] - charge_df["DateTime"].iloc[0]).dt.total_seconds()

    E_discharge = np.mean(stats_discharge["Energy [Wh]"])
    stats_discharge["mean difference"] = (stats_discharge["Energy [Wh]"]-E_discharge)
    discharge_ID = stats_discharge.iloc[(stats_discharge["mean difference"].abs()).idxmin()]["Drive Cycle ID"]

    discharge_df = dc_all[discharge_ID][["DateTime", "Current", "Voltage"]]
    discharge_df["Time"] = (discharge_df["DateTime"] - discharge_df["DateTime"].iloc[0]).dt.total_seconds()

    st.write("The average number of days the asset is used for discharging is", mean_discharge,". We take the mean only from weeks where the asset is used.")
    commuting_days = st.slider("Number of Commute Days", 1, 6, mean_discharge)
    


    I_charge = np.round(abs(charge_rate(charge_dict)/ Q_pack),1)

    step_per = pwr_discharge.mean(axis=0)
    t_total = 0.5*60*60
    t_h = int(step_per.P_high * t_total)
    t_m = int(step_per.P_mid * t_total)
    t_l = int(step_per.P_low * t_total)

    I_h = np.round(np.abs(np.mean(bins_I[0]))/Q_pack,2)
    I_m = np.round(np.abs(np.mean(bins_I[1]))/Q_pack,2)
    I_l = np.round(np.abs(np.mean(bins_I[2]))/Q_pack,2)
    discharge_days = np.sum(days_data_discharge, axis=1)

    fig_days = go.Figure()
    fig_days.add_trace(go.Histogram(
        x=discharge_days,
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
                        "Discharge at " + str(I_l) + "C for " + str(t_l) + " seconds or until 3 V",
                        "Discharge at " + str(I_m) + "C for " + str(t_m) + " seconds or until 3 V",
                        "Discharge at " + str(I_h) + "C for " + str(t_h) + " seconds or until 3 V", 
                        "Rest for 8 hours (30 minute period)", 
                        "Discharge at " + str(I_l) + "C for " + str(t_l) + " seconds or until 3 V",
                        "Discharge at " + str(I_m) + "C for " + str(t_m) + " seconds or until 3 V",
                        "Discharge at " + str(I_h) + "C for " + str(t_h) + " seconds or until 3 V", 
                        "Rest for 8 hours (30 minute period)"
                        ]

    subcycle_charge = ["Rest for 12 hours (30 minute period)", 
                    "Charge at " + str(I_charge) + "C until 4.2 V",
                    "Hold at 4.2 V until 50 mA",
                    "Rest for 10 hours (30 minute period)"
                    ]

    subcycle_rest = ["Rest for 24 hours (60 minute period)",]
    commuteP = create_commute_experiment(commuting_days)
    


    st.markdown(f'''
    ```python
    subcycle_commute = {subcycle_commute}

    subcycle_charge = {subcycle_charge}

    subcycle_rest = {subcycle_rest}

    commuteP = {commuteP}

    exp = pybamm.Experiment([commuteP])
    ''')



    # Add a button
    if st.button('Run PyBaMM Simulation'):

        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=pybamm.Experiment([(*subcycle_commute, *subcycle_commute, *subcycle_commute, *subcycle_commute, *subcycle_rest, *subcycle_rest, *subcycle_charge)]), solver=pybamm.IDAKLUSolver())

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
    data_temp = discharge_df[['Time', 'Power']]
    data_temp['Power'].iloc[0]  = 0

    dT = 2.3
    normal_time = np.arange(0, data_temp['Time'].max(), dT)
    power_interp = np.interp(normal_time, data_temp['Time'], data_temp['Power']*.8)
    normalized_power = pd.DataFrame({'Time': normal_time, 'Power': power_interp})
    # Convert DataFrame directly to CSV format
    csv_data = normalized_power.to_csv(index=False)

    # Create the download button
    st.download_button(
        label="Download Normalized Power Data",
        data=csv_data,
        file_name='normalized_power.csv',
        mime='text/csv'
    )

    step_P = pybamm.step.power(value=normalized_power.values, duration="720 seconds", termination="3.0 V")

    subcycle_commuteP = ["Rest for 8 hours (30 minute period)", 
                        step_P, 
                        "Rest for 8 hours (30 minute period)", 
                        step_P,
                        "Rest for 8 hours (30 minute period)"
                        ]

    st.markdown(f'''
    ```python
    normalized_power = pd.read_csv("normalized_power.csv")
      
    step_P = pybamm.step.power(value=normalized_power.values, duration="720 seconds", termination="3.0 V")
    
    subcycle_commuteP = ["Rest for 8 hours (30 minute period)", 
                        step_P, 
                        "Rest for 8 hours (30 minute period)", 
                        step_P,
                        "Rest for 8 hours (30 minute period)"
                        ]

    subcycle_charge = {subcycle_charge}

    subcycle_rest = {subcycle_rest}

    commuteP = {commuteP}

    exp = pybamm.Experiment([commuteP])
    ''')

    if st.button('Run PyBaMM Simulation with Normalized Power Data'):

        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=pybamm.Experiment([(*subcycle_commuteP, *subcycle_commuteP, *subcycle_commuteP, *subcycle_commuteP, *subcycle_rest, *subcycle_rest, *subcycle_charge)]), solver=pybamm.IDAKLUSolver())

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