import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
import calendar

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

# Identify charge cycles - slightly hard coded at the moment considers the following
# Nominal Charge Current 
# Current is relatively constant - -0.01 < dIdt < 0.01
# Time step is less than 1 second
# Time step is positive
# Charge cycle is greater than 15000 data points


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

def resistance(df, I_pulse, t_pulse):
    df = df[(df.Current > I_pulse - .2) & (df.Current < I_pulse + .2)].copy()

    times = df['DateTime'].values
    diffs = np.diff(times.astype('timedelta64[ns]'))

    diffs = np.pad(diffs, (0,1), 'constant', constant_values=(pd.Timedelta(0))) # pads the array to the original length to account for diff

    df['dt'] = diffs /  pd.Timedelta(minutes=1) * 60
    mask = diffs > np.timedelta64(1_000_000_000, 'ns')
    groups = np.cumsum(mask)

    df['group'] = groups
    
    dicts = {}

    for i in enumerate(df.group):
        temp = df[(df.group == i[1])].copy()

        if (len(temp.dt) > 5):
            # temp['Time'] = np.cumsum(temp.dt)
            dicts[i[1]] = temp

    # # breakpoint()
    R = pd.DataFrame({'DateTime':[], 'R':[], 'V':[], 'I':[]})

    for i in dicts: 
        new_dt = dicts[i].DateTime.diff()  /  pd.Timedelta(minutes=1) * 60

        dicts[i]['new_dt'] = new_dt.fillna(0)

        if (dicts[i].Voltage.iloc[0] - dicts[i].Voltage.iloc[-1] < 0) & (np.cumsum(dicts[i].new_dt).iloc[-1] > t_pulse):
            r = (dicts[i].Voltage.iloc[0] - dicts[i].Voltage.iloc[-1]) / np.mean(dicts[i].Current)
            
            # print(i, ": ", np.round(dicts[i].Voltage.iloc[0] - dicts[i].Voltage.iloc[-1], decimals = 4), "V, ", np.round(r, decimals = 4), "ohm"," \n")

            t = [dicts[i].DateTime.iloc[0], r, dicts[i].Voltage.iloc[0] - dicts[i].Voltage.iloc[-1], np.mean(dicts[i].Current)]

            R.loc[len(R)] = t
    
    return dicts, R

def lean_fil(data, tolerance, capacity):
    n = data.Voltage.size
    dV = 0.0013922872340472736  * 2
    V_min = data.Voltage.min() - tolerance # Minimum voltage for the counting bucket
    V_max = data.Voltage.max() + tolerance # Maximum voltage for the counting bucket
    V = np.arange(V_min, V_max, dV) # Vector V for counting and plotting
    m = round((V_max - V_min)/dV) # Number of voltage bins (length of V)
    N_V = np.full(n, np.nan)
    
    for k in np.linspace(0, m-1, m, dtype='int'):
        N_V[k] = ((data.Voltage >= V[k]) & (data.Voltage < V[k] + dV)).sum()
    
    N_V = N_V[np.isfinite(N_V)]

    dqdvdict = {}

    dQdV_Q = (np.array(N_V) / n / dV) # LEAN dQdV
    dQdV = [i * capacity for i in dQdV_Q] # LEAN dQdV
    dQdV_fil = savgol_filter(dQdV, 1000, 5, mode='nearest')

    dqdvdict['dQdV'] = dQdV
    dqdvdict['V'] = V
    dqdvdict['dQdV_fil'] = dQdV_fil

    return dqdvdict

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

def riding_events(data):
    idle = [-1,0]
    coast = [-10, -1]
    accel = [-20, -10]

    # data = dc[0]
    conditions = [data.Current > 0, 
                np.logical_and(idle[0] <= data.Current, data.Current < idle[1]), 
                np.logical_and(coast[0] <= data.Current, data.Current < coast[1]), 
                data.Current <= accel[1]]
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
