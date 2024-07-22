import streamlit as st
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from data_analysis import load_data, stats_calc
import numpy as np

@st.cache_data
# def load_and_process_data():
#     dc_all_fil = load_data()
#     stats_all = stats_calc(dc_all_fil)
#     stats_discharge = stats_all[stats_all["Mean Power [W]"] < 0].dropna()
#     return stats_discharge

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

def app(data):
    st.title('K-Means Clustering')
    # dc_all_fil = data['dc_all_fil']
    # stats_all = data['stats_all']
    stats_discharge = data['stats_discharge']
    
    # stats_discharge = load_and_process_data()
    cluster_columns = ['Duration [s]', 'Energy [Wh]', 'Mean Power [W]']

    selected_columns = [col for col in cluster_columns if st.sidebar.checkbox(col, value=True)]
    
    scaled_df = scale_data(stats_discharge, selected_columns)

    max_clusters = 16
    sse = compute_sse(scaled_df, max_clusters)

    kn = KneeLocator(range(1, max_clusters), sse, curve='convex', direction='decreasing')
    k = kn.knee or 3

    st.write(f"Optimal number of clusters: {k}")

    clusters = st.slider("Number of clusters", 1, 15, k)

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
        scene=dict(
            xaxis_title="Duration [s]",
            yaxis_title="Energy [Wh]",
            zaxis_title="Mean Power [W]"
        ),
        legend_title="Clusters"
    )

    st.plotly_chart(fig, use_container_width=False)

# if __name__ == "__main__":
#     app(data)