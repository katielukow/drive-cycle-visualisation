import streamlit as st
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from data_analysis import load_data, stats_calc
import numpy as np
import pandas as pd

# @st.cache_data
def load_and_process_data():
    dc_all_fil = load_data()
    stats_all = stats_calc(dc_all_fil)
    stats_discharge = stats_all[stats_all["Mean Power [W]"] < 0].dropna()
    return stats_discharge

@st.cache_resource
def scale_data(df, columns):
    scaler = StandardScaler()
    return scaler.fit_transform(df[columns])

@st.cache_resource
def compute_sse(scaled_df, max_clusters=16):
    sse = np.zeros(max_clusters)
    for k in range(1, max_clusters):
        kmeans = KMeans(n_clusters=k, init="random", n_init=10, random_state=1)
        sse[k] = kmeans.fit(scaled_df).inertia_
    return sse[1:]

@st.cache_resource
def cluster_data(scaled_df, clusters):
    kmeans = KMeans(n_clusters=clusters, init="random", n_init='auto', max_iter=300, random_state=1)
    return kmeans.fit_predict(scaled_df)

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

    clusters = st.slider("Number of clusters", 1, 15, 4)

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

    new_dfs = []
    for i, row in df_temps.iterrows():
        # if row['High'] > 0:
            new_dfs.append({
                'Drive Cycle ID': row['Drive Cycle ID'],
                'Load': row['High_I'],
                'Load Type': 'High_I',
                'Cluster': row['Cluster']
            })
        # if row['Medium'] > 0:
            new_dfs.append({
                'Drive Cycle ID': row['Drive Cycle ID'],
                'Load': row['Medium_I'],
                'Load Type': 'Medium_I',
                'Cluster': row['Cluster']
            })
        # if row['Low'] > 0:
            new_dfs.append({
                'Drive Cycle ID': row['Drive Cycle ID'],
                'Load': row['Low_I'],
                'Load Type': 'Low_I',
                'Cluster': row['Cluster']
            })

            new_dfs.append({
                'Drive Cycle ID': row['Drive Cycle ID'],
                'Load': row['High_P'],
                'Load Type': 'High_P',
                'Cluster': row['Cluster']
            })
        # if row['Medium'] > 0:
            new_dfs.append({
                'Drive Cycle ID': row['Drive Cycle ID'],
                'Load': row['Medium_P'],
                'Load Type': 'Medium_P',
                'Cluster': row['Cluster']
            })
        # if row['Low'] > 0:
            new_dfs.append({
                'Drive Cycle ID': row['Drive Cycle ID'],
                'Load': row['Low_P'],
                'Load Type': 'Low_P',
                'Cluster': row['Cluster']
            })
            

    violin_df = pd.DataFrame(new_dfs)
    cluster1vAll = violin_df[(violin_df['Cluster'] == 1) |(violin_df['Cluster'] == 'A')].copy()
    cluster1vAll.loc[:,'EnergyName'] = 'Energy [Wh]'

    fig_vio = go.Figure()
    fig_vio.add_trace(go.Violin(x=cluster1vAll['Load Type'][cluster1vAll['Cluster']==1], y = cluster1vAll['Load'][cluster1vAll['Cluster']==1], legendgroup='Yes', scalegroup='Yes', name='Yes', side='negative', line_color='blue'))

    fig_vio.add_trace(go.Violin(x=cluster1vAll['Load Type'][cluster1vAll['Cluster']=='A'], y = cluster1vAll['Load'][cluster1vAll['Cluster']=='A'], legendgroup='Yes', scalegroup='Yes', name='Yes', side='positive', line_color='orange'))
    fig_vio.update_traces(meanline_visible=True)
    fig_vio.update_layout(violingap=0, violinmode='overlay')

    st.plotly_chart(fig_vio, use_container_width=False)

    

if __name__ == "__main__":
    app()