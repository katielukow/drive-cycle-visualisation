import streamlit as st
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from data_analysis import load_data, stats_calc, load_and_process_data
import numpy as np
import pandas as pd


@st.cache_resource
def scale_data(df, columns):
    scaler = StandardScaler()
    return scaler.fit_transform(df[columns])

@st.cache_resource
def compute_sse(scaled_df, max_clusters=16):
    sse = [KMeans(n_clusters=k, init="random", n_init=10, random_state=1).fit(scaled_df).inertia_
           for k in range(1, max_clusters)]
    return sse

@st.cache_resource
def cluster_data(scaled_df, clusters):
    return KMeans(n_clusters=clusters, init="random", n_init='auto', max_iter=300, random_state=1).fit_predict(scaled_df)

def create_new_dfs(df_temps):
    load_types = ['High_I', 'Medium_I', 'Low_I', 'High_P', 'Medium_P', 'Low_P']
    return [
        {
            'Drive Cycle ID': row['Drive Cycle ID'],
            'Load': row[load_type],
            'Load Type': load_type,
            'Cluster': row['Cluster']
        }
        for _, row in df_temps.iterrows()
        for load_type in load_types
    ]

stats_discharge = load_and_process_data()

def app():
    st.title('K-Means Clustering')

    cluster_columns = ['Duration [s]', 'Energy [Wh]', 'Mean Power [W]', 'High_P', 'Medium_P', 'Low_P', 'Max Power [W]']

    selected_columns = [col for col in cluster_columns if st.sidebar.checkbox(col, value=True)]
    
    scaled_df = scale_data(stats_discharge, selected_columns)

    max_clusters = 16
    sse = compute_sse(scaled_df, max_clusters)

    kn = KneeLocator(range(1, max_clusters), sse, curve='convex', direction='decreasing')
    k = kn.knee or 3

    st.write(f"Optimal number of clusters: {k}")

    # query_params = st.experimental_get_query_params()
    # default_clusters = int(query_params.get("clusters", [4])[0])

    clusters = st.slider("Number of clusters", 1, 15, k)

    # Update the query params when the slider changes
    # st.experimental_set_query_params(clusters=clusters)

    labels = cluster_data(scaled_df, clusters)
    df_clustered = stats_discharge.copy()
    df_clustered['Cluster'] = labels

    fig = go.Figure()
    for i in range(clusters):
        df_cluster = df_clustered[df_clustered["Cluster"] == i]
        mean_power = (df_cluster["Energy [Wh]"] * 1000) / df_cluster["Duration [s]"]

        fig.add_trace(
            go.Scatter3d(
                x=df_cluster["Duration [s]"],
                y=df_cluster["Energy [Wh]"],
                z=mean_power,
                mode='markers',
                name=f'Cluster {i}'
            )
        )

    fig.update_layout(
        autosize=True,
        scene=dict(
            xaxis_title="Duration [s]",
            yaxis_title="Energy [Wh]",
            zaxis_title="Mean Power [W]"
        ),
        legend_title="Clusters"
    )

    st.plotly_chart(fig, use_container_width=False)

    stats_discharge["Cluster"] = 'A'

    stats_concat = pd.concat([df_clustered, stats_discharge], ignore_index=False)

    df_temps = stats_concat[['Drive Cycle ID','High_I', 'Medium_I', 'Low_I', 'High_P', 'Medium_P', 'Low_P', 'Cluster', 'Energy [Wh]']]

    load_types = ['High_I', 'Medium_I', 'Low_I', 'High_P', 'Medium_P', 'Low_P']
    new_dfs = [
        {
            'Drive Cycle ID': row['Drive Cycle ID'],
            'Load': row[load_type],
            'Load Type': load_type,
            'Cluster': row['Cluster'],
            'Energy [Wh]': row['Energy [Wh]']
        }
        for _, row in df_temps.iterrows()
        for load_type in load_types
    ]

    query_params = st.experimental_get_query_params()
    default_plot_cluster = int(query_params.get("plot_cluster", [1])[0])

    # Update the slider value based on the query parameter
    plot_cluster = st.slider("Cluster to compare", 0, clusters-1, default_plot_cluster)

    # Update the query params when the slider changes
    st.experimental_set_query_params(plot_cluster=plot_cluster)

    violin_df = pd.DataFrame(new_dfs)
    cluster1vAll = violin_df[(violin_df['Cluster'] ==  plot_cluster) |(violin_df['Cluster'] == 'A')].copy()
    cluster1vAll.loc[:,'EnergyName'] = 'Energy [Wh]'
    fig_vio = go.Figure()

    # Add violin traces
    fig_vio.add_trace(go.Violin(
        x=cluster1vAll['EnergyName'][cluster1vAll['Cluster'] == plot_cluster], 
        y=cluster1vAll['Energy [Wh]'][cluster1vAll['Cluster'] == plot_cluster], 
        side='negative', 
        line_color='blue', width=0.5
    ))

    fig_vio.add_trace(go.Violin(
        x=cluster1vAll['EnergyName'][cluster1vAll['Cluster'] == 'A'], 
        y=cluster1vAll['Energy [Wh]'][cluster1vAll['Cluster'] == 'A'], 
        side='positive', 
        line_color='orange', 
        width=0.5
    ))

    # Make mean line visible and set violin plot mode
    fig_vio.update_traces(meanline_visible=True)

    # Update layout to adjust x-axis
    fig_vio.update_layout(
        # violingap=0, 
        violinmode='overlay',
        xaxis=dict(
            title='Energy Type',  # Title for x-axis
            categoryorder='category ascending'  # Order of categories
        ),
        yaxis=dict(title='Energy [Wh]')
    )

    # Optionally, set y-axis range to avoid zooming in too much
    y_min = cluster1vAll['Energy [Wh]'].min()
    y_max = cluster1vAll['Energy [Wh]'].max()
    padding = (y_max - y_min) * 0.3  # Add some padding to the y-axis range

    fig_vio.update_layout(
        violingap=0, 
        violinmode='overlay',
        yaxis=dict(range=[y_min - padding, y_max + padding])
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig_vio, use_container_width=True)



    

if __name__ == "__main__":
    app()