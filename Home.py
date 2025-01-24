import streamlit as st
import os
from data_analysis import load_data
import plotly.graph_objects as go

# Define the directory for pages
PAGES_DIR = 'pages'

# st.cache_data.clear()
# file_path = os.path.join(PAGES_DIR, '1_All_Data.py')
# ../data/demo_data.parquet
# Function to load a module from a file
def load_module(module_name, filepath):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def create_figure(filtered_dict_V):
    fig = go.Figure()
    for key, df in filtered_dict_V.items():
        fig.add_trace(go.Scatter(x=df['Time_Hours'], y=df['Voltage'], mode='lines', name=f'Drive Cycle {key}'))

    fig.update_layout(
        title='Voltage vs Time for Filtered Drive Cycles',
        xaxis_title='Time (hours)',
        yaxis_title='Voltage',
        legend_title='Drive Cycles',
        height=600,
    )
    fig.update_xaxes(range=[-5, None])
    return fig

# Get a list of pages
page_files = [f for f in os.listdir(PAGES_DIR) if f.endswith('.py')]
pages = {f.replace('.py', '').replace('_', ' ').title(): os.path.join(PAGES_DIR, f) for f in page_files}

def app():
    st.title('Battery Data Analysis')
    if "data_path" not in st.session_state:
        st.session_state.data_path = ""  # Default value
    st.session_state.data_path = st.text_input(
        "Enter the path to the data file:", st.session_state.data_path
    )

    if st.session_state.data_path == "":
        st.write("Please enter the path to the data file.")
    else:
        st.session_state.dc_all_fil, st.session_state.data_all, st.session_state.dc_all = load_data(st.session_state.data_path)

        # cycle_status = {}
        for key, df in st.session_state.dc_all_fil.items():
            # cycle_status[key] = 'charge' if (df['Current'] >= 0).all() else 'discharge'
            if 'Time_Hours' not in df.columns:  # Avoid recomputing if already added
                df['Time_Hours'] = (df['DateTime'] - df['DateTime'].iloc[0]).dt.total_seconds() / 3600

        fig = create_figure(st.session_state.dc_all_fil)
        st.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    app()
# main_page = '0 Trip Visualisation'
# load_module(main_page, pages[main_page]).app()
