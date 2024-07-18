import streamlit as st
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from data_analysis import *

# Cache the data loading
@st.cache_data
def load_and_process_data():
    dc_all_fil = load_data()
    stats_all = stats_calc(dc_all_fil)
    stats_discharge = stats_all[stats_all["Mean Power [W]"] < 0]
    return stats_discharge.dropna()



# Cache the scaling process
@st.cache_resource
def scale_data(df, columns):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df[columns])
    return scaled_df

# Cache the K-means SSE computation
@st.cache_resource
def compute_sse(scaled_df, kmeans_kwargs, max_clusters=16):
    sse = {}
    for k in range(1, max_clusters):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_df)
        sse[k] = kmeans.inertia_
    return sse

stats_discharge = load_and_process_data()
# cluster_columns = ['Duration [s]', 'Energy [Wh]', 'Mean Power [W]']
    
# scaled_df = scale_data(stats_discharge, cluster_columns)


def app():
    st.title('K-Means Clustering')


    cluster_columns = ['Duration [s]', 'Energy [Wh]', 'Mean Power [W]']

    # Streamlit sidebar for column selection
    selected_columns = []
    for column in cluster_columns:
        if st.sidebar.checkbox(column, value=True):
            selected_columns.append(column)
    
    scaled_df = scale_data(stats_discharge, selected_columns)

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "random_state": 1,
    }

    max_clusters = 16
    sse = compute_sse(scaled_df, kmeans_kwargs, max_clusters)

    kn = KneeLocator(x=list(sse.keys()), y=list(sse.values()), curve='convex', direction='decreasing')
    k = kn.knee or 3  # Default to 3 if knee is not found
    # Cache the final K-means clustering
    st.write(f"Optimal number of clusters: {k}")

    @st.cache_resource
    def cluster_data(scaled_df, clusters):
        kmeans = KMeans(
            init="random", 
            n_init='auto',
            n_clusters=clusters,
            max_iter=300,
            random_state=1
        )
        kmeans.fit(scaled_df)
        return kmeans.labels_
    
        
    clusters = st.slider("Number of clusters", 1, 15, k)

    labels = cluster_data(scaled_df, clusters)
    df_clustered = stats_discharge.copy()
    df_clustered['Cluster'] = labels

    fig = go.Figure()
    for i in df_clustered["Cluster"].unique():
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

