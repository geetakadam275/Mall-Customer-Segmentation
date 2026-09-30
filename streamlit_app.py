import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Page configuration
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛍️ Mall Customer Segmentation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Segment customers using K-Means Clustering based on Annual Income & Spending Score</div>', unsafe_allow_html=True)

# 1. Load Data
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "Mall_Customers.csv")
    if not os.path.exists(file_path):
        file_path = "Mall_Customers.csv"
    data = pd.read_csv(file_path)
    return data

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}. Please ensure 'Mall_Customers.csv' is in the repository.")
    st.stop()

# 2. Train Model
@st.cache_resource
def train_kmeans_model(data):
    features = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    return scaler, kmeans, scaled_features, clusters

scaler, kmeans, X_scaled, clusters = train_kmeans_model(df)
df['Cluster'] = clusters

# Define Cluster Personas (derived from KMeans cluster centers)
cluster_info = {
    0: {
        "name": "Standard / Average Customers",
        "description": "Average income and average spending score. A steady and reliable customer base.",
        "strategy": "Engage with loyalty rewards, seasonal promotions, and steady product offerings.",
        "color": "#3B82F6"
    },
    1: {
        "name": "Target / Premium Shoppers",
        "description": "High income and high spending score. Highly valuable customer segment.",
        "strategy": "Target with VIP perks, luxury collections, premium memberships, and personalized offers.",
        "color": "#10B981"
    },
    2: {
        "name": "Careless / Impulsive Shoppers",
        "description": "Low income but high spending score. Trend-driven and frequent shoppers.",
        "strategy": "Attract with flash sales, trending items, budget-friendly installments, and gamified discounts.",
        "color": "#F59E0B"
    },
    3: {
        "name": "Careful / Savers",
        "description": "High income but low spending score. Wealthy but cautious with expenditures.",
        "strategy": "Highlight value-for-money, high durability, investment-grade items, and long-term benefits.",
        "color": "#8B5CF6"
    },
    4: {
        "name": "Sensible / Budget Customers",
        "description": "Low income and low spending score. Price-conscious and minimalist shoppers.",
        "strategy": "Offer essential goods, clear discounts, coupons, and clearance deals.",
        "color": "#EF4444"
    }
}

# Sidebar for User Input
st.sidebar.header("🎯 Customer Profile Prediction")
st.sidebar.markdown("Enter customer details to predict their cluster segment:")

input_income = st.sidebar.slider("Annual Income (k$):", min_value=15, max_value=140, value=65, step=1)
input_score = st.sidebar.slider("Spending Score (1-100):", min_value=1, max_value=100, value=50, step=1)

# Prediction Logic
user_data = np.array([[input_income, input_score]])
user_data_scaled = scaler.transform(user_data)
predicted_cluster = int(kmeans.predict(user_data_scaled)[0])
persona = cluster_info[predicted_cluster]

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediction & Map", "📋 Dataset Overview", "📈 Cluster Summary"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Prediction Result")
        st.markdown(f"""
        <div style="background-color: {persona['color']}15; padding: 1.2rem; border-radius: 10px; border-left: 5px solid {persona['color']};">
            <h3 style="color: {persona['color']}; margin-top:0;">Cluster {predicted_cluster}: {persona['name']}</h3>
            <p><strong>Profile:</strong> {persona['description']}</p>
            <hr style="border: 0.5px solid #E5E7EB;">
            <p><strong>💡 Recommended Strategy:</strong><br>{persona['strategy']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.metric(label="Selected Annual Income", value=f"${input_income}k")
        st.metric(label="Selected Spending Score", value=f"{input_score}/100")
        
    with col2:
        st.subheader("🗺️ Customer Segmentation Map")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        
        scatter = ax.scatter(
            df['Annual Income (k$)'], 
            df['Spending Score (1-100)'], 
            c=df['Cluster'], 
            cmap='viridis', 
            s=55, 
            alpha=0.75,
            edgecolors='white',
            linewidth=0.5
        )
        
        # Plot centroids
        centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
        ax.scatter(
            centroids_unscaled[:, 0], 
            centroids_unscaled[:, 1], 
            color='black', 
            marker='X', 
            s=160, 
            label='Cluster Centroids'
        )
        
        # Highlight user-selected point
        ax.scatter(
            [input_income], 
            [input_score], 
            color='red', 
            marker='*', 
            s=350, 
            edgecolors='black',
            linewidth=1.5,
            label='Your Customer'
        )
        
        ax.set_title("Customer Clusters with Current Customer Marked", fontsize=12, fontweight='bold')
        ax.set_xlabel("Annual Income (k$)", fontsize=10)
        ax.set_ylabel("Spending Score (1-100)", fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        st.pyplot(fig)

with tab2:
    st.subheader("📋 Mall Customers Dataset")
    st.dataframe(df, use_container_width=True)
    st.write("### Quick Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

with tab3:
    st.subheader("📈 Summary by Cluster")
    summary = df.groupby('Cluster').agg(
        Count=('CustomerID', 'count'),
        Avg_Income=('Annual Income (k$)', 'mean'),
        Avg_Spending=('Spending Score (1-100)', 'mean'),
        Avg_Age=('Age', 'mean')
    ).reset_index()
    
    summary['Segment Persona'] = summary['Cluster'].map(lambda c: cluster_info[c]['name'])
    st.dataframe(summary[['Cluster', 'Segment Persona', 'Count', 'Avg_Income', 'Avg_Spending', 'Avg_Age']], use_container_width=True)
