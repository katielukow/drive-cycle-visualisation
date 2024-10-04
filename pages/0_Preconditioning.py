import streamlit as st
import plotly.graph_objects as go
from data_analysis import load_data, stats_calc, user_power_division, riding_events_power
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def hist_data(data):
    counts, bin_edges = np.histogram(data, bins=50)

    # Find all peaks and their prominences
    peaks, properties = find_peaks(counts, prominence=1)  # Adjust prominence threshold if needed
    sorted_peaks = np.argsort(properties['prominences'])[::-1]  # Sort by prominence (descending)
    top_3_peaks = peaks[sorted_peaks[:3]]  # Take the 3 most prominent peaks
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate the bin centers
    peak_positions = bin_centers[top_3_peaks]   # Get the positions of the peaks
    peak_positions.sort()

    inverted_counts = -counts
    valleys, properties = find_peaks(inverted_counts, prominence=1) 
    sorted_valleys = np.argsort(properties['prominences'])[::-1]
    top_2_valleys = valleys[sorted_valleys[:2]]

    valley_positions = np.sort(bin_centers[top_2_valleys])

    # Calculate boundaries (midpoints) between the three peak positions
    boundaries_peaks = [(peak_positions[i] + peak_positions[i+1]) / 2 for i in range(len(peak_positions) - 1)]
    min_data = np.min(data)
    max_data = np.max(data)
    boundaries_peaks = [min_data] + boundaries_peaks + [max_data]
    boundaries_valleys = [min_data, valley_positions[0], valley_positions[1], max_data]

    # Assign data to bins using np.digitize
    bin_indices = np.digitize(data, boundaries_valleys)

    bin_1 = data[bin_indices == 1]
    bin_2 = data[bin_indices == 2]
    bin_3 = data[bin_indices == 3]

    return [(min(bin_1), max(bin_1)), (min(bin_2), max(bin_2)), (min(bin_3), max(bin_3))], [bin_1, bin_2, bin_3]

def basic_load_profile(dc_all, discharge_dict, charge_dict):
    Q_pack = 11

    st.write('### Basic Charge-Discharge Profile')
    st.write('The following pybamm experiment definition will cycle the battery between 0% and 100% SOC at the average C-Rates from the filtered field data.')

    I_discharge = pd.Series([df['Current'].mean() for df in discharge_dict.values()]).mean()
    I_charge = pd.Series([df['Current'].mean() for df in charge_dict.values()]).mean()

    # convert to single cell P = V * I 
    n_series = 10
    n_parallel = 4

    I_charge = np.round(I_charge / Q_pack,1)
    I_discharge = np.round(abs(I_discharge) / Q_pack,1)


    st.write("pybamm.Experiment("+"[(" + ", ".join(['"{}"'.format(item) for item in [
        "Discharge at %s C until 2.5 V" % I_discharge, 
        "Charge at %s C until 4.2 V" % I_charge, 
        "Hold at 4.2 V until 50 mA"]]) + ")])")


def app():
    st.title('Initial Data Analysis')
    dc_all_fil, data_all, dc_all = load_data()
    cycle_status = {key: 'charge' if (np.mean(df['Current']) >= 0) else 'discharge' for key, df in dc_all.items()}

    # Calculate statistics once
    charge_dict = {k: v for k, v in dc_all.items() if cycle_status[k] == 'charge'}
    discharge_dict = {k: v for k, v in dc_all.items() if cycle_status[k] == 'discharge'}
    del discharge_dict[3] # remove the drive cycle with ID 3 as it is an outlier
    del discharge_dict[4] # remove the drive cycle with ID 4 as it is an outlier
    
    Q_pack = 11

    st.write('### Basic Charge-Discharge Profile')
    st.write('The following pybamm experiment definition will cycle the battery between 0% and 100% SOC at the average C-Rates from the filtered field data.')

    I_discharge = pd.Series([df['Current'].mean() for df in discharge_dict.values()]).mean()
    I_charge = pd.Series([df['Current'].mean() for df in charge_dict.values()]).mean()

    # convert to single cell P = V * I 
    n_series = 10
    n_parallel = 4

    I_charge = np.round(I_charge / Q_pack,1)
    I_discharge = np.round(abs(I_discharge) / Q_pack,1)


    st.markdown(
        "**pybamm.Experiment(" + "[(" + ", ".join(
            ['"{}"'.format(item) for item in [
                "Discharge at %s C until 2.5 V" % I_discharge, 
                "Charge at %s C until 4.2 V" % I_charge, 
                "Hold at 4.2 V until 50 mA"]]) + ")])**"
    )

    
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

    # Create a new Series with the combined 'Discharge' and original 'Charge_P' and 'Off'
    new_data = pd.Series({
        'Discharge': discharge_value,
        'Charge': sum_data['P_charge'],
        'Off': sum_data['Off']
    })
    
    # labels = ['High', 'Medium', 'Low', 'Charge', 'Off']
    labels = ['Discharge', 'Charge', 'Off']
    
    # Plot the pie chart
    st.write('### Distribution of Asset Usage')
    st.write('The pie chart below shows the distribution of power consumption during the drive cycles. This can then be incorporated into the simplified day load profile with a consideration of the time when the asset is not in use.')
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=new_data)])
    st.plotly_chart(fig_pie, use_container_width=True)

    st.write('The following PyBaMM load profile considers the C-Rates from the filtered field data previously presented. The load profile is scaled to a 24 hour period that can be repeated with the rest being proportaional to usage time as presented above.')

    t_total = 24 * 3600
    t_r = new_data['Off'] / new_data.sum(axis=0) * t_total
    t_d = new_data['Discharge'] / new_data.sum(axis=0) * t_total
    t_c = new_data['Charge'] / new_data.sum(axis=0) * t_total

    st.markdown("**pybamm.Experiment("+"[(" + ", ".join(['"{}"'.format(item) for item in [
        "Discharge at {:.2f} C for {:.0f} seconds or until 2.5 V".format(I_discharge, t_d),
        "Charge at {:.2f} C for {:.0f} seconds or until 4.2 V".format(I_charge, t_c),
        # "Hold at 4.2 V until 50 mA", 
        "Rest for {:.0f} seconds".format(t_r)
    ]]) + ")])**")

    st.write('### Discharge Usage Distribution')
    st.write('The final step of the initial data analysis is to consider the users distribution of power consumption during discharge trips. The following plot shows a histogram of the power distribution during all discharge trips. This data is then separated into three bins defined by the valleys in the distributions. These bins are then used to create a stepped simplified load profile that takes the proportional time spent in each bin into consideration.')
    # Calculate proportions
    sum_data_per = sum_data / sum_data.sum()

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=power_data, nbinsx=100, histnorm='probability density'))
    fig_hist.update_layout(
        title='Power Distribution',
        xaxis_title='Power [W]',
        yaxis_title='Probability Density',
        height=600,
    )

    # Add vertical line at x = -245
    fig_hist.add_shape(
        type="line",
        x0=bins_P[1][0], x1=bins_P[1][0],  # x-position of the vertical line
        y0=0, y1=0.055,  # y-position (from 0 to max y-axis value)
        line=dict(color="red", width=2, dash="dash")  # Customize line appearance
    )

    fig_hist.add_shape(
        type="line",
        x0=bins_P[2][0], x1=bins_P[2][0],  # x-position of the vertical line
        y0=0, y1=0.055,  # y-position (from 0 to max y-axis value)
        line=dict(color="red", width=2, dash="dash")  # Customize line appearance
    )

    st.plotly_chart(fig_hist, use_container_width=True)


    # User-set values for current categories
    # I_values = {
    #     'High_I': I_bins[0].mean(),
    #     'Medium_I': I_bins[1].mean(),
    #     'Low_I': I_bins[2].mean(),
    #     'Charge_I': 4
    # }
    # fig = go.Figure(data=[go.Pie(labels=['High', 'Medium', 'Low', 'Charge','Off'], values=sum_data[['P_high', 'P_mid', 'P_low', 'P_charge','Off']])])
    # st.plotly_chart(fig, use_container_width=True)

    fig_hist = go.Figure()

    # Add the first trace with bins of size 0.01
    fig_hist.add_trace(go.Histogram(
        x=pwr_discharge['P_high'], 
        histnorm='probability density',
        xbins=dict(
            size=0.01  # Bin width of 0.01
        ), name='High Power'
    ))
    # Add the second trace with bins of size 0.01
    fig_hist.add_trace(go.Histogram(
        x=pwr_discharge['P_mid'], 
        histnorm='probability density',
        xbins=dict(
            size=0.01  # Bin width of 0.01
        ), name='Medium Power'
    ))

    # Add the third trace with bins of size 0.01
    fig_hist.add_trace(go.Histogram(
        x=pwr_discharge['P_low'], 
        histnorm='probability density',
        xbins=dict(
            size=0.01  # Bin width of 0.01
        ), name='Low Power'
    ))
    fig_hist.update_layout(
        title='Power Distribution during Discharge',
        xaxis_title='Power [W]',
        yaxis_title='Probability Density',
        height=600,
        legend_title='Power Category',

    )
    st.plotly_chart(fig_hist, use_container_width=True)



if __name__ == "__main__":
    app()
