import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc
import pandas as pd
import numpy as np
from data_analysis import load_data, drive_cycle_id, stats_calc, user_stat 

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
    filtered_dict_I, day_behaviour, violin_df, stats_all, user_all = prepare_data()

    st.write('## User Behaviour')

    st.write('### Number of Drive Cycles by Day')
    fig1 = go.Figure()
    for id_type in ['charge', 'discharge']:
        subset = day_behaviour[day_behaviour['ID'] == id_type]
        fig1.add_trace(go.Histogram(x=subset['Day of Week'], name=id_type.capitalize(), histfunc='count', nbinsx=7))
    st.plotly_chart(fig1)

    mean_days_charge, mean_days_discharge = weekly_count(user_all)

    st.write(f'The average number of discharge drive cycles per week is {mean_days_discharge}')
    st.write(f'The average number of charge drive cycles per week is {mean_days_charge}')

    charge_df, discharge_df = mean_cycles(filtered_dict_I)

    week_load_profile = load_profile(charge_df, discharge_df, mean_days_charge, mean_days_discharge)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=week_load_profile['Time'] / 3600, y=week_load_profile['Current'], mode='lines'))
    fig.update_layout(
        title='Current vs Time for a Week',
        xaxis_title='Time (hours)',
        yaxis_title='Current (A)',
    )
    st.plotly_chart(fig)

    # Download button for the CSV file
    st.download_button(
        label="Download weekly load profile as CSV",
        data=week_load_profile.to_csv(index=False),
        file_name='data_all.csv',
        mime='text/csv'
    )

    # user_all_discharge = user_all[user_all['Status'] == 0]
    # tod_count = user_all_discharge['TOD'].value_counts()
    # fig_pie = go.Figure(data=[go.Pie(labels=tod_count.index, values=tod_count)])
    # st.plotly_chart(fig_pie, use_container_width=True)
    
    # mean_behaviour = stats_all[["Mean Current [A]",	"Energy [Wh]",	"Mean Power [W]",	"Capacity [Ah]",	"Max Current [A]",	"Max Power [W]"]].mean()
    # st.dataframe(mean_behaviour)


    st.write('### Power Distribution for Discharge Drive Cycles')
    fig2 = go.Figure()
    fig2.add_trace(go.Violin(x=violin_df['Load Type'],
                            y=violin_df['Load']
                ))
    fig2.update_traces(meanline_visible=True)
    fig2.update_layout(violinmode='group')
    st.plotly_chart(fig2)

    # # Calculate absolute differences from mean values
    # differences = (data_c1[columns_to_use] - means).abs()
    # normal_diffs = differences / means.abs()



if __name__ == "__main__":
    app()