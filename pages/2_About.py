import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_analysis import * 


csv = pd.read_csv("../data/20240122-Data.csv")
df_init = data_init(csv)
df_all = datetime_corr(df_init)
df_all['Time_of_Day'] = df_all['DateTime'].dt.strftime('%p')

data = df_all[(df_all.Date > '2023-10-01') & (df_all.Date < '2023-11-01')].copy() # Filter for just October for development purposes
dc_all = drive_cycle_id(df_all, 60) 

# nom_Vp = 36 # nominal voltage of the pack
# max_Vp = 42 # assumed max operating voltage of the pack
min_Vp = 25 # assumed min operating voltage of the pack
# Q_pack = 11 # capacity of the pack in Ah
I_max = 6 # max current of the pack in A
I_min = -30 # min current of the pack in A

filtered_dict_V = {key: df for key, df in dc_all.items() if (df['Voltage'] >= min_Vp).all()}
dc_all_fil = {key: df for key, df in filtered_dict_V.items() if ((df['Current'] >= I_min) & (df['Current'] <= I_max)).all()}
# # Filter out the drive cycles that have any measurements below the operating voltage defined above
# filtered_dict_V = {key: df for key, df in dc_all.items() if (df['Voltage'] >= min_Vp).all()} 
# dc_all_fil = {key: df for key, df in filtered_dict_V.items() if ((df['Current'] >= I_min)&(df['Current'] <= I_max)).all()} 
def app():
    st.title('Drive Cycle Identification')

    # Input constants
    st.sidebar.header("Constants")
    # nom_Vp = st.sidebar.number_input("Nominal voltage of the pack", value=36)
    # max_Vp = st.sidebar.number_input("Max operating voltage of the pack", value=42)
    # Q_pack = st.sidebar.number_input("Capacity of the pack (Ah)", value=11)
    # I_max = st.sidebar.number_input("Max current of the pack (A)", value=6)
    # I_min = st.sidebar.number_input("Min current of the pack (A)", value=-30)

    # Filter drive cycles

    charge = st.sidebar.checkbox("Charge", True)
    discharge = st.sidebar.checkbox("Discharge", True)
    if charge and discharge:
        filtered_dict_V = dc_all_fil
    elif charge:
        filtered_dict_V = {key: df for key, df in dc_all_fil.items() if (df['Current'] >= 0).all()}
    elif discharge:
        filtered_dict_V = {key: df for key, df in dc_all_fil.items() if (df['Current'] <= 0).all()}
    else:
        filtered_dict_V = False
    # Plot results
    st.header("Drive Cycle Results")

    if filtered_dict_V:
        fig = go.Figure()
        for key, df in filtered_dict_V.items():
            fig.add_trace(go.Scatter(x=(df['DateTime']-df['DateTime'].iloc[0]).dt.total_seconds()/3600, y=df['Voltage'], mode='lines', name=f'Drive Cycle {key}'))
        
        fig.update_layout(
            title='Voltage vs Time for Filtered Drive Cycles',
            xaxis_title='Time',
            yaxis_title='Voltage',
            legend_title='Drive Cycles',
            height=600,
        )
        fig.update_xaxes(range=[-0.1, None])
        
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Number of filtered drive cycles: {len(dc_all_fil)}")
    else:
        st.write("No drive cycles meet the specified criteria.")

# # Create a sidebar for filtering
# st.sidebar.title('Filters')
# column_to_filter = st.sidebar.selectbox('Select a column to filter', data.columns)
# filter_value = st.sidebar.text_input('Enter a value to filter', '')

# # Filter the data based on user input
# if filter_value:
#     filtered_data = data[data[column_to_filter].str.contains(filter_value, case=False)]
# else:
#     filtered_data = data

# # Display the filtered data
# st.write(filtered_data)