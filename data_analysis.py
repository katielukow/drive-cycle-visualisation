import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import calendar
from scipy.signal import find_peaks
from plotly.subplots import make_subplots
import pybamm
import plotly.graph_objects as go

file_path = '../data/20240122-Data.parquet'

# Fix date-time formatting (this should be fixed in the arduino code...)
def data_init(df):
    df.rename(columns={"Time": "DateTime"}, inplace=True)
    df['dI'] = df.Current.diff()
    df['dV'] = df.Voltage.diff()

    return df

def datetime_corr(df):
    df['DateTime'] = df['DateTime'].str.replace('T',' ') 
    df['DateTime'] = df['DateTime'].str.replace('Z','')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Date'] = pd.to_datetime(df['DateTime']).dt.date
    df['Date'] = pd.to_datetime(df['Date'])
    # df['dt'] = df.DateTime.diff()  /  pd.Timedelta(minutes=1) * 60
    # df['Time'] = pd.to_datetime(df['DateTime']).dt.time
    # # df['Time'] = pd.to_datetime(df['Time'])
    return df

def coulomb_calc(data):
    for i in data:
        data[i]['dQ [A.h]'] = data[i]['Current'] * (data[i]['dt']) / 3600
        data[i]['Q [A.h]'] = data[i]['dQ [A.h]'].cumsum()
   
    return data

@st.cache_data
def load_data():
    nom_Vp = 36 # nominal voltage of the pack
    max_Vp = 42 # assumed max operating voltage of the pack
    min_Vp = 25 # assumed min operating voltage of the pack
    Q_pack = 11 # capacity of the pack in Ah
    I_max = 30 # max current of the pack in A
    I_min = -6 # min current of the pack in A
    csv = pd.read_parquet(file_path)
    df_init = data_init(csv)
    df_init = df_init[(df_init['DateTime'] > '2023-10-01 00:00:00') & (df_init['DateTime'] < '2023-12-01 00:00:00')].copy()
    # df_all = datetime_corr(df_init)
    df_init['Time_of_Day'] = df_init['DateTime'].dt.strftime('%p')
    df_init['Current'] = df_init['Current'] * -1
    df_init['Power'] = df_init['Current'] * df_init['Voltage']

    # df_temp = df_init[(df_init['DateTime'] > '2023-10-01 00:00:00') & (df_init['DateTime'] < '2023-11-01 00:00:00')].copy()
    # dc_all = drive_cycle_id(df_init, 60) 

    data_filtered = df_init[(df_init["Current"] > I_min) & (df_init["Current"] < I_max) & (df_init['Voltage'] >= min_Vp)]
    # data_filtered["Power"] = data_filtered['Current'] * data_filtered['Voltage']
    dc_all = drive_cycle_id(data_filtered, 60) 

    filtered_dict_V = {key: df for key, df in dc_all.items() if (df['Voltage'] >= min_Vp).all()} 
    dc_all_fil = {key: df for key, df in filtered_dict_V.items() if ((df['Current'] >= I_min)&(df['Current'] <= I_max)).all()} 


    return dc_all_fil, data_filtered, dc_all

# Identify charge cycles - slightly hard coded at the moment considers the following
# Nominal Charge Current 
# Current is relatively constant - -0.01 < dIdt < 0.01
# Time step is less than 1 second
# Time step is positive
# Charge cycle is greater than 15000 data points

@st.cache_data
def load_and_process_data():
    dc_all_fil, data_filtered, dc_all = load_data()
    stats_all = stats_calc(dc_all_fil)
    return stats_all[stats_all["Mean Power [W]"] < 0].dropna()


def charge_id(df, I_charge, dIdt):
    # Add a column for the change in current if it doesn't exist
    if 'dI' not in df.columns:
        df['dI'] = df['Current'].diff()

    # Find the rows where the current is within the specified range
    within_range = (df['Current'] > I_charge - 1) & (df['Current'] < I_charge + 1)

    # Add a column for time differences
    df['dt'] = df['DateTime'].diff() / pd.Timedelta(minutes=1) * 60

    # Filter out data points based on the change in current (dI) and time difference (dt)
    df['keep'] = (df['dI'] < dIdt) & (df['dI'] > -dIdt) & (df['dt'] > 0) & (df['dt'] < 1)

    # Apply the initial filter for the current range
    df_filtered = df[within_range & df['keep']].copy()

    # Find unique dates, assuming that there is never more than one charge a day
    unique_dates = df_filtered['Date'].unique()
    charge_cycles = {}

    for date in unique_dates:
        # Create a temporary dataframe for each unique date
        temp = df_filtered[df_filtered['Date'] == pd.to_datetime(str(date))]

        # Only consider charge cycles with a sufficient number of data points
        if temp.shape[0] > 15000:
            # Find the index where the filtered data starts
            start_idx = temp.index[0]

            # Find the index where current first drops to zero or below after the start index
            end_idx = df[(df.index >= start_idx) & (df['Current'] <= 0.43)].index[0]

            # Select the data from the start index to the end index
            full_charge_cycle = df.loc[start_idx:end_idx]

            # Store the result in the dictionary
            charge_cycles[str(date.date())] = full_charge_cycle
            charge_cycles[str(date.date())]['Time'] = np.cumsum(charge_cycles[str(date.date())]['dt'])

    coulomb_calc(charge_cycles)

    return charge_cycles



def time_div(data):
    def sign(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0
    
    data['dt'] = data.DateTime.diff()  /  pd.Timedelta(minutes=1) * 60
    data['group'] = data.Current.apply(sign)
    data.dt = data.dt.shift(-1)
    data.dt = data.dt.fillna(0)

    charge = np.cumsum(data[(data.group == 1)].copy().dt).iloc[-1]
    discharge = np.cumsum(data[(data.group == -1)].copy().dt).iloc[-1]
    rest = np.cumsum(data[(data.group == 0)].copy().dt).iloc[-1]
    total = (data.DateTime.iloc[-1] - data.DateTime.iloc[0]) / pd.Timedelta(minutes=1) * 60

    labels = 'Charge', 'Discharge', 'Rest'
    sizes = [charge/total, discharge/total, rest/total]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    

    print('The charge time is:', np.round(charge/3600, decimals=3), 'hours. \nThe discharge time is:', np.round(discharge/3600, decimals=3), 'hours. \nThe rest time is:', np.round(rest/3600, decimals=3), " hours. \n")
    print("The total time is: ", np.round(total/3600, decimals=3), 'hours.')
    
    plt.show()
    
    return data

def drive_cycle_id(data, t_delta):
#     t_delta = 60 # seconds - set based on histogram for mask, will determine 'rest' period
    delta_t = data.DateTime.diff()/pd.Timedelta(minutes=1)*60
    mask = delta_t > t_delta
    mask_idx = delta_t[mask].index
    dc = {}
    for i, idx in enumerate(mask_idx):
        if i == 0:
            dc[i] = data.loc[0:idx-1]
        else:
            dc[i] = data.loc[mask_idx[i-1]:idx-1]
    
    return dc

def riding_events(data, bins):
    accel = [bins[2][0], bins[2][1]]
    coast = [bins[1][0], bins[1][1]]
    idle = [bins[0][0], bins[0][1]]


    conditions = [data.Current < 0, 
                np.logical_and(idle[0] < data.Current, data.Current < idle[1]), 
                np.logical_and(coast[0] < data.Current, data.Current < coast[1]), 
                data.Current >= accel[0]]
    values = [0, 1, 2, 3]

    time = np.asarray(data.DateTime.diff().dt.total_seconds().copy()) # time in seconds
    time[0] = 0
    # Create an array of event indices with the same size as the data index
    event_idx = np.select(conditions, values)

    df = pd.DataFrame({'event': event_idx, 'dt': time})

    accel_time = np.cumsum(df[df.event == 3].dt).iloc[-1] if (df['event'] == 3).sum() > 0 else 0
    coast_time = np.cumsum(df[df.event == 2].dt).iloc[-1] if (df['event'] == 2).sum() > 0 else 0
    idle_time = np.cumsum(df[df.event == 1].dt).iloc[-1] if (df['event'] == 1).sum() > 0 else 0
    charge_time = np.cumsum(df[df.event == 0].dt).iloc[-1] if (df['event'] == 0).sum() > 0 else 0
    total_time = np.cumsum(df.dt).iloc[-1]

    # print('Acceleration Time: ', accel_time/60)
    # print('Coast Time: ', coast_time/60)
    # print('Idle Time: ', idle_time/60)
    # print('Charge Time: ', charge_time/60)
    # print('Total Time Powered on: ', total_time/60)

    return [accel_time, coast_time, idle_time, total_time, charge_time]

def riding_events_power(data,  bins):
    accel = [bins[2][0], bins[2][1]]
    coast = [bins[1][0], bins[1][1]]
    idle = [bins[0][0], bins[0][1]]

    power = data['Current'] * data['Voltage']
    conditions = [power < 0, 
                np.logical_and(idle[0] < power, power < idle[1]), 
                np.logical_and(coast[0] < power, power < coast[1]), 
                power >= accel[0]]
    values = [0, 1, 2, 3]

    time = np.asarray(data.DateTime.diff().dt.total_seconds().copy()) # time in seconds
    time[0] = 0
    # Create an array of event indices with the same size as the data index
    event_idx = np.select(conditions, values)

    df = pd.DataFrame({'event': event_idx, 'dt': time})

    accel_time = np.cumsum(df[df.event == 3].dt).iloc[-1] if (df['event'] == 3).sum() > 0 else 0
    coast_time = np.cumsum(df[df.event == 2].dt).iloc[-1] if (df['event'] == 2).sum() > 0 else 0
    idle_time = np.cumsum(df[df.event == 1].dt).iloc[-1] if (df['event'] == 1).sum() > 0 else 0
    charge_time = np.cumsum(df[df.event == 0].dt).iloc[-1] if (df['event'] == 0).sum() > 0 else 0
    total_time = np.cumsum(df.dt).iloc[-1]

    # print('Acceleration Time: ', accel_time/60)
    # print('Coast Time: ', coast_time/60)
    # print('Idle Time: ', idle_time/60)
    # print('Charge Time: ', charge_time/60)
    # print('Total Time Powered on: ', total_time/60)

    return [accel_time, coast_time, idle_time, charge_time, total_time]
#     return new_df
def count_days(start_date, end_date):
    
    """
    Count the number of Fridays between two dates (inclusive).
    
    Args:
        start_date (str or datetime.date or datetime.datetime): The start date.
        end_date (str or datetime.date or datetime.datetime): The end date.
        
    Returns:
        int: The number of Fridays between the start and end dates (inclusive).
    """
    # Convert input to datetime objects if necessary
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # Ensure start_date is not after end_date
    start_date, end_date = sorted([start_date, end_date])
    
    # Initialize the count
    
    day_counts = {day: 0 for day in calendar.day_name}
    
    # Iterate over the date range
    current_date = start_date
    while current_date <= end_date:
        day_name = calendar.day_name[current_date.weekday()]
        day_counts[day_name] += 1
        current_date += timedelta(days=1)
    
    return day_counts

def energy_calc(data):
    df = data.copy()
    time = df.DateTime.diff().dt.total_seconds().copy() # time in seconds
    time.iloc[0] = 0

    df.loc[:,'dt'] = time

    temp = df[df.dt >= 0]

    power = temp['Current'] * temp['Voltage'] # power in watts
    energy = np.trapz(power, np.cumsum(temp.dt)) / 3600 # energy in kilojoules
    capacity = np.trapz(temp['Current'], np.cumsum(temp.dt)) / 3600 # capacity in amp hours
        
    return energy, capacity

def stats_calc(data_input):
    Pmeans = [(data_input[i]['Current'] * data_input[i]['Voltage']).mean() for i in data_input]

    drive_cycle_id = [i for i in data_input]

    duration = []
    Emeans = []

    for i in data_input:
        time = np.cumsum(data_input[i].DateTime.diff().dt.total_seconds())
        d = time.iloc[-1]
        duration.append(d)
        energy, capacity = energy_calc(data_input[i])
        Emeans.append(energy)

    dates = [data_input[i].DateTime.iloc[0].date() for i in data_input]

    
    df = pd.DataFrame({"Drive Cycle ID": drive_cycle_id,
                       "Duration [s]": duration, 
                       "Date": dates, 
                       "Energy [Wh]":Emeans, 
                       "Mean Power [W]":Pmeans, 
                       }) 

    return df

def user_stat(data_input):

    drive_cycle_id = [i for i in data_input]
    duration = []
    for i in data_input:
        time = np.cumsum(data_input[i].DateTime.diff().dt.total_seconds())
        d = time.iloc[-1]
        duration.append(d)
    V0 = [data_input[i]['Voltage'].iloc[0] for i in data_input]
    dates = [data_input[i].DateTime.iloc[0].date() for i in data_input]
    TOD = [data_input[i].DateTime.iloc[0].strftime('%p') for i in data_input]
    Pmeans = [(data_input[i]['Current'] * data_input[i]['Voltage']).mean() for i in data_input]

    df = pd.DataFrame({"Drive Cycle ID": drive_cycle_id,
                       "Duration [s]": duration, 
                       "Date": dates, 
                       "Initial Voltage [V]":V0, 
                       "TOD": TOD, 
                        "Mean Power [W]":Pmeans,

                       }) 
    
    df['Status'] = (df['Mean Power [W]'] > 0).astype(int)
    # df['Time of Day'] = (df['TOD'] == 'PM').astype(int)

    return df

def user_power_division(data, plot):
    # Create a histogram (50 bins)
    
    counts, bin_edges = np.histogram(data, bins=50)

    # Invert the counts to find valleys
    inverted_counts = -counts

    # Find all valleys (peaks in the inverted histogram) and their prominences
    valleys, properties = find_peaks(inverted_counts, prominence=1)  # Adjust prominence threshold if needed

    # Sort valleys by prominence and select the top 2
    sorted_valleys = np.argsort(properties['prominences'])[::-1]  # Sort by prominence (descending)
    top_2_valleys = valleys[sorted_valleys[:2]]  # Take the 2 most prominent valleys

    # Get the positions of the top 2 valleys in terms of bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate the bin centers
    valley_positions = bin_centers[top_2_valleys]

    # Sort the valley positions to ensure proper boundary calculation
    valley_positions.sort()

    # Add extreme boundaries to cover all data (min and max data)
    min_data = np.min(data)
    max_data = np.max(data)
    boundaries = [min_data] + valley_positions.tolist() + [max_data]

    # Assign data to bins using np.digitize
    bin_indices = np.digitize(data, boundaries)
    if plot:
        # Plot histogram and mark the valleys
        plt.hist(data, bins=50, alpha=0.6, label='Data')
        plt.plot(bin_centers[top_2_valleys], counts[top_2_valleys], 'bx', label='Top 2 Valleys', markersize=10)

        # Plot vertical lines at the boundaries
        for boundary in boundaries:
            plt.axvline(x=boundary, color='blue', linestyle='--', label='Boundary')

        plt.xlabel('Data')
        plt.ylabel('Frequency')
        # plt.legend()
        plt.show()

    # Separate data into bins based on valley boundaries
    bin_1 = data[bin_indices == 1]
    bin_2 = data[bin_indices == 2]
    bin_3 = data[bin_indices == 3]

    bins = [(min(bin_1), max(bin_1)), (min(bin_2), max(bin_2)), (min(bin_3), max(bin_3))]

    hist_data = np.column_stack((bin_centers, counts))

    return bins, hist_data
    
def charge_rate(charge_dict):
    # Calculate stats for charge_dict
    c = stats_calc(charge_dict)

    # Calculate deltaV as the difference between the last and first 'Voltage' values for each entry
    deltaV = {key: df['Voltage'].iloc[-1] - df['Voltage'].iloc[0] for key, df in charge_dict.items()}

    # Add deltaV to the 'c' DataFrame
    c['dV'] = pd.Series(deltaV).reset_index()[0]

    # Filter out entries that meet the condition of dV > 5 and Duration > 600
    charge_list = c.query('dV > 5 and `Duration [s]` > 600')["Drive Cycle ID"].to_list()

    # Create the filtered dictionary by iterating over charge_list
    filtered_dict_new = {}

    # Loop through each item in charge_list to filter based on dI values starting from the second row
    for i in charge_list:
        # Start checking from the second row (index 1) to avoid modifying the first row
        df_subset = charge_dict[i].iloc[1:]  # This excludes the first row
        
        # Find the first occurrence where 'dI' is outside the range (-0.026, 0.026)
        first_occurrence_index = df_subset[(df_subset['dI'] < -0.026) | (df_subset['dI'] > 0.026)].index.min()

        # If an occurrence is found, filter the DataFrame up to that index
        if pd.notna(first_occurrence_index):
            df_filtered = charge_dict[i].loc[:first_occurrence_index-1]
        else:
            df_filtered = charge_dict[i]  # If no occurrence, keep the original DataFrame

        # Store the filtered DataFrame in the new dictionary
        filtered_dict_new[i] = df_filtered

    # Calculate the mean current from the filtered DataFrames
    c_mean = {key: df['Current'].mean() for key, df in filtered_dict_new.items()}

    # Calculate the overall mean charge current and normalize by Q_pack
    return np.mean(list(c_mean.values()))

def pybamm_plot(experiment):
    param = pybamm.ParameterValues("Chen2020")
    sim = pybamm.Simulation(model=pybamm.lithium_ion.SPM(), parameter_values=param, experiment=pybamm.Experiment(experiment), solver = pybamm.IDAKLUSolver())
    sol = sim.solve(initial_soc=1)

    time = sol["Time [s]"].entries
    current = sol["Current [A]"].entries / param['Nominal cell capacity [A.h]']
    voltage = sol["Terminal voltage [V]"].entries

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Current Over Time', 'Voltage Over Time'))

    # Left subplot: Current
    fig.add_trace(go.Scatter(x=time, y=current, mode='lines', name='C-Rate', line=dict(color='blue')), row=1, col=1)

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
    fig.update_yaxes(title_text="C-Rate", row=1, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_yaxes(title_text="Voltage [V]", row=2, col=1)

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
