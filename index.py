import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -------------------------------
# Load and prepare data
# -------------------------------
file_path = "data.csv"
df = pd.read_csv(file_path, header=None)
df.columns = [f"Feature_{i}" for i in range(df.shape[1])]

# -------------------------------
# Sidebar: Data Transformations & Filtering
# -------------------------------
st.sidebar.header("Data Transformations")
normalize = st.sidebar.checkbox("Normalize Data")
standardize = st.sidebar.checkbox("Standardize Data")

# Apply transformation (only one at a time)
if normalize:
    df = (df - df.min()) / (df.max() - df.min())
elif standardize:
    df = (df - df.mean()) / df.std()

st.sidebar.header("Data Filtering")
selected_feature = st.sidebar.selectbox("Select Feature to Filter", df.columns)
min_val = float(df[selected_feature].min())
max_val = float(df[selected_feature].max())
filter_range = st.sidebar.slider(f"Select range for {selected_feature}", min_val, max_val, (min_val, max_val))

# Create filtered dataframe based on selected feature range
df_filtered = df[(df[selected_feature] >= filter_range[0]) & (df[selected_feature] <= filter_range[1])]

use_filtered = st.sidebar.checkbox("Use Filtered Data", value=False)
if use_filtered:
    data = df_filtered.copy()
else:
    data = df.copy()

# Download button for current data
st.sidebar.download_button(
    label="Download Current Data as CSV",
    data=data.to_csv(index=False).encode('utf-8'),
    file_name="current_data.csv",
    mime="text/csv"
)

# -------------------------------
# Visualization Functions (using global variable 'data')
# -------------------------------

def plot_heatmap():
    st.subheader("Heatmap of Dataset")
    plt.figure(figsize=(12, 5))
    sns.heatmap(data, cmap="coolwarm", annot=False, cbar=True)
    st.pyplot(plt.gcf())

def plot_pca():
    st.subheader("PCA Scatter Plot")
    # PCA on the transposed data (each feature as a sample)
    data_transposed = data.T
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_transposed)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(plt.gcf())

def plot_histogram():
    st.subheader("Histogram of Dataset Values")
    plt.figure(figsize=(10, 6))
    plt.hist(data.values.flatten(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    st.pyplot(plt.gcf())

def plot_population_vs_sample():
    st.subheader("Population vs Sample Distribution")
    population = data.values.flatten()
    sample_size = int(0.2 * len(population))
    sample = np.random.choice(population, sample_size, replace=False)
    plt.figure(figsize=(10, 6))
    sns.histplot(population, bins=50, color='blue', label='Population', kde=True, alpha=0.5)
    sns.histplot(sample, bins=50, color='red', label='Sample (20%)', kde=True, alpha=0.5)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(plt.gcf())

def plot_correlation_heatmap():
    st.subheader("Correlation Matrix Heatmap")
    corr = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    st.pyplot(plt.gcf())

def plot_pairplot():
    st.subheader("Pair Plot")
    sns.pairplot(data)
    st.pyplot(plt.gcf())

def plot_boxplot():
    st.subheader("Box Plot of Features")
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=data, orient="h")
    st.pyplot(plt.gcf())

def plot_violinplot():
    st.subheader("Violin Plot of Features")
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=data, orient="h", inner="quartile")
    st.pyplot(plt.gcf())

def plot_pca_variance():
    st.subheader("PCA Explained Variance")
    pca = PCA()
    pca.fit(data)
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    st.pyplot(plt.gcf())

def plot_kmeans_clusters():
    st.subheader("K-Means Clustering")
    num_clusters = st.slider("Select Number of Clusters", 2, 10, 3, key="kmeans_slider")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # Apply KMeans on transposed data so that each feature is a sample
    data_transposed = data.T
    kmeans.fit(data_transposed)
    labels = kmeans.labels_
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_transposed)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
    plt.title(f"K-Means Clustering with {num_clusters} Clusters")
    st.pyplot(plt.gcf())

# -------------------------------
# Streamlit Main UI
# -------------------------------
st.title("Y allah selamatkan aku dari sir tws")
st.write("### Dataset Overview")
st.dataframe(data.head())
st.write("### Descriptive Statistics")
st.dataframe(data.describe().T)

# Visualization Options
visualization = st.selectbox("Choose a visualization", [
    "Heatmap",
    "PCA Scatter Plot",
    "Histogram",
    "Population vs Sample Distribution",
    "Correlation Matrix Heatmap",
    "Pair Plot",
    "Box Plot",
    "Violin Plot",
    "PCA Explained Variance",
    "K-Means Clustering"
])

# Display the selected visualization
if visualization == "Heatmap":
    plot_heatmap()
elif visualization == "PCA Scatter Plot":
    plot_pca()
elif visualization == "Histogram":
    plot_histogram()
elif visualization == "Population vs Sample Distribution":
    plot_population_vs_sample()
elif visualization == "Correlation Matrix Heatmap":
    plot_correlation_heatmap()
elif visualization == "Pair Plot":
    plot_pairplot()
elif visualization == "Box Plot":
    plot_boxplot()
elif visualization == "Violin Plot":
    plot_violinplot()
elif visualization == "PCA Explained Variance":
    plot_pca_variance()
elif visualization == "K-Means Clustering":
    plot_kmeans_clusters()

st.write("bamor still learning")

