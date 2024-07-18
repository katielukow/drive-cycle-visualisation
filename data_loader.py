import streamlit as st
import pandas as pd
from data_analysis import stats_calc

@st.cache_data

def load_data():


    min_Vp = 25 # assumed min operating voltage of the pack
    I_max = 6 # max current of the pack in A
    I_min = -30 # min current of the pack in A
    df = pd.read_parquet('../data/20240122-Data.parquet')
    # df_init = data_init(csv)
    # df_all = datetime_corr(df_init)
    df.rename(columns={"Time": "DateTime"}, inplace=True)
    df['dI'] = df.Current.diff()
    df['dV'] = df.Voltage.diff()
    df['Time_of_Day'] = df['DateTime'].dt.strftime('%p')

    df_temp = df[(df['DateTime'] > '2023-10-01 00:00:00') & (df['DateTime'] < '2023-11-01 00:00:00')].copy()
    dc_all = drive_cycle_id(df_temp, 60) 

    filtered_dict_V = {key: df for key, df in dc_all.items() if (df['Voltage'] >= min_Vp).all()} 
    dc_all_fil = {key: df for key, df in filtered_dict_V.items() if ((df['Current'] >= I_min)&(df['Current'] <= I_max)).all()} 


    return dc_all_fil

def load_and_process_data():
    if 'processed_data' not in st.session_state:
        dc_all_fil = load_data()
        stats_all = stats_calc(dc_all_fil)
        stats_discharge = stats_all[stats_all["Mean Power [W]"] < 0].dropna()
        
        st.session_state['processed_data'] = {
            'dc_all_fil': dc_all_fil,
            'stats_all': stats_all,
            'stats_discharge': stats_discharge
        }
    
    return st.session_state['processed_data']

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
