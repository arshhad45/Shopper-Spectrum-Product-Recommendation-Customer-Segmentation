import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Shopper Spectrum App", layout="wide")
st.title("🛍️ Shopper Spectrum: Product Recommendation & Customer Segmentation")

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_shopper_data.csv")
    pivot = pd.pivot_table(df, index='CustomerID', columns='Description', values='Quantity', aggfunc='sum').fillna(0)
    sim_matrix = cosine_similarity(pivot.T)
    sim_df = pd.DataFrame(sim_matrix, index=pivot.columns, columns=pivot.columns)
    return df, sim_df

df, sim_df = load_data()

scaler = joblib.load("scaler.pkl")
model = joblib.load("rfm_kmeans_model.pkl")

recommend_tab, segment_tab = st.tabs(["📦 Product Recommendation", "👥 Customer Segmentation"])

with recommend_tab:
    st.subheader("🎯 Product Recommendation Module")
    product_input = st.text_input("Enter a product name (case-sensitive, from 'Description' column):")
    if st.button("Get Recommendations"):
        if product_input in sim_df.index:
            top5 = sim_df[product_input].sort_values(ascending=False).iloc[1:6]
            st.success(f"Top 5 Recommendations for '{product_input}'")
            for i, (prod, score) in enumerate(top5.items(), 1):
                st.markdown(f"**{i}.** {prod} _(Similarity: {score:.2f})_")
        else:
            st.error("Product not found. Please check the spelling or casing.")

with segment_tab:
    st.subheader("🎯 Customer Segmentation Module")
    recency = st.number_input("Recency (days)", min_value=0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("Monetary (₹ total spend)", min_value=0.0)
    if st.button("Predict Cluster"):
        user_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        scaled_data = scaler.transform(user_data)
        cluster = model.predict(scaled_data)[0]
        def get_segment_label(r, f, m):
            if r < 30 and f > 10 and m > 1000:
                return "High-Value"
            elif f >= 5 and m >= 300:
                return "Regular"
            elif r > 150 and f < 3:
                return "At-Risk"
            else:
                return "Occasional"
        segment = get_segment_label(recency, frequency, monetary)
        st.success(f"🧠 Predicted Cluster: **{segment}** (Cluster #{cluster})")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
st.markdown("**Tags:** Pandas, Scikit-Learn, Cosine Similarity, KMeans, RFM, E-Commerce, Collaborative Filtering")
