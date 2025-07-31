# 🛍️ Shopper Spectrum: Product Recommendation & Customer Segmentation

A Streamlit-based interactive web application that provides:
- 📦 **Product Recommendations** using cosine similarity
- 👥 **Customer Segmentation** using RFM analysis and K-Means clustering

This project empowers e-commerce businesses to better understand customer behavior and improve sales through personalized insights.

---

## 🚀 Demo

Deploy it easily with:

- ✅ Run locally using `streamlit run app.py`
- ✅ Or deploy on **Streamlit Cloud** using GitHub

---

## 📂 Project Structure

shopper-spectrum-app/
│
├── app.py # Streamlit application

├── cleaned_shopper_data.csv # Cleaned transactional dataset

├── scaler.pkl # Pre-fitted scaler for RFM features

├── rfm_kmeans_model.pkl # Trained K-Means model

├── requirements.txt # Python dependencies

└── .gitattributes # Git LFS tracking for large files


---

## 📊 Features

### 🎯 Product Recommendation Module
- Input a **product name** from the dataset
- Get **Top 5 similar products** based on **cosine similarity**
- Built using `pandas`, `scikit-learn`, and `Streamlit`

### 🎯 Customer Segmentation Module
- Input customer’s **Recency**, **Frequency**, and **Monetary** values
- Predicts customer segment:
  - `High-Value`
  - `Regular`
  - `Occasional`
  - `At-Risk`

---

## 🧠 Models Used

- **Cosine Similarity**: For item-based collaborative filtering
- **K-Means Clustering**: For customer segmentation based on RFM analysis
- **Scaler (StandardScaler)**: To normalize RFM values

---
