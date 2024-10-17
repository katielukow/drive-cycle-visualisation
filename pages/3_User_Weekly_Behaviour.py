import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc
import pandas as pd
import numpy as np
import pybamm
import matplotlib.pyplot as plt
from data_analysis import load_data, stats_calc, user_power_division, riding_events_power

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
    day_behaviour['ID'] = np.where(day_behaviour['Mean Power [W]'] > 0, 'charge', 'discharge')

    stats_discharge = stats_all[stats_all["Mean Power [W]"] < 0]

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
    cycle_status = {key: 'charge' if (df['Current'] >= 0).all() else 'discharge' for key, df in filtered_dict_I.items()}
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
    else:
        for _ in range(commute_days):
            commuteP.extend(subcycle_commuteP)


    # If the total cycle length is less than 7, fill remaining days with rest
    total_cycles = commute_days + 1  # Commute days + 1 charge day
    if total_cycles < 7:
        remaining_days =  7 - total_cycles
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
    cycle_status = {key: 'charge' if (np.mean(df['Current']) >= 0) else 'discharge' for key, df in dc_all.items()}

    # Calculate statistics once
    charge_dict = {k: v for k, v in dc_all.items() if cycle_status[k] == 'charge'}
    discharge_dict = {k: v for k, v in dc_all.items() if cycle_status[k] == 'discharge'}
    del discharge_dict[3] # remove the drive cycle with ID 3 as it is an outlier
    del discharge_dict[4] # remove the drive cycle with ID 4 as it is an outlier
    
    Q_pack = 11

    discharge_fil = {key: df for key, df in discharge_dict.items() if ((df['Current'] <= 1)).all()}
    power_data = np.concatenate([discharge_fil[i]['Power'] for i in discharge_fil])
    current_data = np.concatenate([discharge_fil[i]['Current'] for i in discharge_fil])
    power_data = power_data[power_data < 0]
    current_data = current_data[current_data < 0]

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

    # fig_hist = go.Figure()
    # fig_hist.add_trace(go.Histogram(x=power_data, nbinsx=100, histnorm='probability density'))
    # fig_hist.update_layout(
    #     title='Power Distribution',
    #     xaxis_title='Power [W]',
    #     yaxis_title='Probability Density',
    #     height=600,
    # )



    # User-set values for current categories
    # I_values = {
    #     'High_I': I_bins[0].mean(),
    #     'Medium_I': I_bins[1].mean(),
    #     'Low_I': I_bins[2].mean(),
    #     'Charge_I': 4
    # }
    # fig = go.Figure(data=[go.Pie(labels=['High', 'Medium', 'Low', 'Charge','Off'], values=sum_data[['P_high', 'P_mid', 'P_low', 'P_charge','Off']])])
    # st.plotly_chart(fig, use_container_width=True)

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

    st.write("The average number of days the asset is used for discharging is", mean_discharge)
    commuting_days = st.slider("Number of Commute Days", 1, 6, mean_discharge)
    commuteP = create_commute_experiment(commuting_days)


    I_charge = pd.Series([df['Current'].mean() for df in charge_dict.values()]).mean()
    I_charge = np.round(I_charge / Q_pack,1)

    step_per = pwr_discharge.mean(axis=0)
    t_total = 0.5*60*60
    t_h = step_per.P_high * t_total
    t_m = step_per.P_mid * t_total
    t_l = step_per.P_low * t_total

    I_h = np.abs(np.mean(bins_I[0]))/Q_pack
    I_m = np.abs(np.mean(bins_I[1]))/Q_pack
    I_l = np.abs(np.mean(bins_I[2]))/Q_pack

    discharge_days = np.sum(days_data_discharge, axis=1)

    fig_days = go.Figure()
    fig_days.add_trace(go.Histogram(x=discharge_days, histnorm='probability density'))
    fig_days.update_layout(
        title='Discharge Days per Week',
        xaxis_title='Number of Discharge Days',
        yaxis_title='Probability Density',
        height=600,
    )
    st.plotly_chart(fig_days, use_container_width=True)

    st.write("The first experiment we create is a weekly representative load profile with the stepped load from before.")


    st.markdown(f'''
    ```python
    subcycle_commuteP = ["Rest for 8 hours (30 minute period)", 
                        "Discharge at {I_l:.2f} C for {t_l:.0f} seconds or until 2.5 V",
                        "Discharge at {I_m:.2f} C for {t_m:.0f} seconds or until 2.5 V",
                        "Discharge at {I_h:.2f} C for {t_h:.0f} seconds or until 2.5 V", 
                        "Rest for 8 hours (30 minute period)", 
                        "Discharge at {I_l:.2f} C for {t_l:.0f} seconds or until 2.5 V",
                        "Discharge at {I_m:.2f} C for {t_m:.0f} seconds or until 2.5 V",
                        "Discharge at {I_h:.2f} C for {t_h:.0f} seconds or until 2.5 V",
                        "Rest for 8 hours (30 minute period)"
                        ]

    subcycle_charge = ["Rest for 12 hours (30 minute period)", 
                    "Charge at {I_charge:.2f}C for 2 hours or until 4.2 V",
                    "Hold at 4.2 V until 50 mA",
                    "Rest for 10 hours (30 minute period)"
                    ]

    subcycle_rest = ["Rest for 24 hours (60 minute period)",]

    subcycle_short_rest = ["Rest for 8 hours (30 minute period)",]

    commuteP = {commuteP}

    exp = pybamm.Experiment([commuteP] * subcycle_number)''', unsafe_allow_html=True)


    # Add a button
    if st.button('Run PyBaMM Simulation'):
        # PyBaMM script that runs when the button is pressed
        st.write(I_charge)
        model = pybamm.lithium_ion.SPM()  # You can replace this with your specific PyBaMM model
        subcycle_commuteP = ["Rest for 8 hours (30 minute period)", 
                            "Discharge at " + str(I_l) + "C for " + str(t_l) + "seconds or until 2.5 V",
                            "Discharge at " + str(I_m) + "C for " + str(t_m) + "seconds or until 2.5 V",
                            "Discharge at " + str(I_h) + "C for " + str(t_h) + "seconds or until 2.5 V", 
                            "Rest for 8 hours (30 minute period)", 
                            "Discharge at " + str(I_l) + "C for " + str(t_l) + "seconds or until 2.5 V",
                            "Discharge at " + str(I_m) + "C for " + str(t_m) + "seconds or until 2.5 V",
                            "Discharge at " + str(I_h) + "C for " + str(t_h) + "seconds or until 2.5 V", 
                            "Rest for 8 hours (30 minute period)"
                            ]

        subcycle_charge = ["Rest for 12 hours (30 minute period)", 
                        "Charge at " + str(I_charge) + "C for 2 hours or until 4.2 V",
                        "Hold at 4.2 V until 50 mA",
                        "Rest for 10 hours (30 minute period)"
                        ]

        subcycle_rest = ["Rest for 24 hours (60 minute period)",]

        commute = (*subcycle_commuteP, *subcycle_rest, *subcycle_commuteP, *subcycle_rest, *subcycle_commuteP, *subcycle_rest, *subcycle_charge)

        exp = pybamm.Experiment([commute])

        # Create the simulation
        sim = pybamm.Simulation(model, experiment=exp)

        # Solve the simulation
        sol = sim.solve()

    # Step 4: Extract voltage and current
        time = sol["Time [s]"].entries
        current = sol["Current [A]"].entries
        voltage = sol["Terminal voltage [V]"].entries

        # Step 5: Create a Plotly figure with dual y-axes
        fig = go.Figure()

        # Left y-axis: Current
        fig.add_trace(go.Scatter(x=time, y=current, mode='lines', name='Current [A]', yaxis='y1', line=dict(color='blue')))

        # Right y-axis: Voltage
        fig.add_trace(go.Scatter(x=time, y=voltage, mode='lines', name='Voltage [V]', yaxis='y2', line=dict(color='red')))

        # Step 6: Set up the layout with two y-axes
        fig.update_layout(
            title='Current and Voltage Over Time',
            xaxis=dict(title='Time [s]'),
            yaxis=dict(title='Current [A]', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
            yaxis2=dict(title='Voltage [V]', titlefont=dict(color='red'), tickfont=dict(color='red'), anchor='x', overlaying='y', side='right'),
            legend=dict(x=0.1, y=0.9)
        )

        # Step 7: Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)





    


if __name__ == "__main__":
    app()