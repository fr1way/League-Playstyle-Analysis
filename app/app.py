import streamlit as st
from data.preprocess import preprocess_lol_data
from models.clustering import run_pca, run_kmeans
from utils.plotting import plot_clusters

st.title("LoL Team Playstyle Clustering")

uploaded_file = st.file_uploader("Upload Ranked Match CSV", type=["csv"])
if uploaded_file:
    X_scaled = preprocess_lol_data(uploaded_file)
    st.write("Processed Feature Sample", X_scaled.head())

    if st.button("Run Clustering"):
        reduced_data, _ = run_pca(X_scaled)
        labels, _ = run_kmeans(reduced_data)
        fig = plot_clusters(reduced_data, labels)
        st.pyplot(fig)
