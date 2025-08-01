import streamlit as st
import pandas as pd
from data.preprocess import preprocess_lol_data
from models.clustering import (
    run_pca, run_kmeans, run_hdbscan,
    evaluate_kmeans_clusters,
    classify_playstyle
)
from utils.plotting import plot_clusters, plot_radar

st.title("LoL Team Playstyle Clustering")

# === SESSION STATE INITIALIZATION ===
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False
if "X_scaled" not in st.session_state:
    st.session_state.X_scaled = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "kmeans" not in st.session_state:
    st.session_state.kmeans = None
if "labels" not in st.session_state:
    st.session_state.labels = None

# === LOAD DATASET ===
if st.button("Load Dataset"):
    data_path = "src/League_of_Legends_Ranked_Match_Data.csv"
    X_scaled, original_df, scaler = preprocess_lol_data(data_path)
    st.session_state.dataset_loaded = True
    st.session_state.X_scaled = X_scaled
    st.session_state.original_df = original_df
    st.session_state.scaler = scaler
    st.success("Dataset loaded successfully.")

# === MAIN APP LOGIC ===
if st.session_state.dataset_loaded:
    X_scaled = st.session_state.X_scaled
    original_df = st.session_state.original_df
    scaler = st.session_state.scaler

    st.write("Processed Feature Sample", X_scaled.head())

    method = st.selectbox("Clustering Method", ["KMeans", "HDBSCAN"])

    if st.button("Run Clustering"):
        reduced_data, _ = run_pca(X_scaled)

        if method == "KMeans":
            labels, kmeans = run_kmeans(X_scaled)
        else:
            labels, kmeans = run_hdbscan(X_scaled)

        st.session_state.kmeans = kmeans
        st.session_state.labels = labels

        # PCA plot
        fig = plot_clusters(reduced_data, labels)
        st.pyplot(fig)

        # Cluster summaries
        st.subheader("Cluster Insights")
        labeled_data = X_scaled.copy()
        labeled_data["Cluster"] = labels
        cluster_summary = labeled_data.groupby("Cluster").mean().round(2)
        st.dataframe(cluster_summary)

        # Radar chart per cluster
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X_scaled.columns)
        for cluster_id, row in cluster_summary.iterrows():
            st.markdown(f"### Cluster {cluster_id}")
            top_features = row.sort_values(ascending=False).head(2).index.tolist()
            st.write(f"- Top characteristics: **{top_features[0]}**, **{top_features[1]}**")

            cluster_centroid_df = centroids.iloc[[cluster_id]]
            fig_cluster_radar = plot_radar(cluster_centroid_df, X_scaled.columns.tolist(), titles=[f"Cluster {cluster_id}"])
            st.pyplot(fig_cluster_radar)


        # Win rate by cluster
        if "win" in original_df.columns:
            original_df = original_df.reset_index(drop=True)
            original_df["Cluster"] = labels
            win_rates = original_df.groupby("Cluster")["win"].mean().round(3)
            st.subheader("Win Rate per Cluster")
            st.dataframe(win_rates)

# === PLAYER CLASSIFICATION ===
if st.checkbox("Classify a Sample Player"):
    if st.session_state.kmeans and st.session_state.original_df is not None:
        sample_idx = st.number_input("Select player row index", min_value=0, max_value=len(st.session_state.original_df) - 1, step=1)
        cluster_names = {
            0: "Teamfighter",
            1: "Macro Farmer",
            2: "Vision Support"
        }
        result = classify_playstyle(
            st.session_state.original_df.iloc[sample_idx],
            st.session_state.scaler,
            st.session_state.kmeans,
            cluster_names
        )
        st.markdown(f"**Player:** {st.session_state.original_df.iloc[sample_idx]['champion_name']}")
        st.markdown(f"**Playstyle:** {result['label']} (Cluster {result['cluster']})")
        st.markdown("**Feature Scores:**")
        st.json(result["features"])
    else:
        st.warning("Please load data and run clustering first.")
