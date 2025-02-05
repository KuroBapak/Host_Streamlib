import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load dataset without header
file_path = "data.csv"
df = pd.read_csv(file_path, header=None)

# Add column names based on index
df.columns = [f"Feature_{i}" for i in range(df.shape[1])]

# Function to plot heatmap
def plot_heatmap():
    st.subheader("Heatmap of Dataset")
    plt.figure(figsize=(12, 5))
    sns.heatmap(df, cmap="coolwarm", annot=False, cbar=True)
    st.pyplot(plt)

# Function to plot PCA scatter plot
def plot_pca():
    st.subheader("PCA Scatter Plot")
    df_transposed = df.T
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_transposed)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(plt)

# Function to plot histogram
def plot_histogram():
    st.subheader("Histogram of Dataset Values")
    plt.figure(figsize=(10, 6))
    plt.hist(df.values.flatten(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    st.pyplot(plt)

# Function to plot population vs sample distribution
def plot_population_vs_sample():
    st.subheader("Population vs Sample Distribution")
    population = df.values.flatten()
    sample_size = int(0.2 * len(population))
    sample = np.random.choice(population, sample_size, replace=False)
    plt.figure(figsize=(10, 6))
    sns.histplot(population, bins=50, color='blue', label='Population', kde=True, alpha=0.5)
    sns.histplot(sample, bins=50, color='red', label='Sample (20%)', kde=True, alpha=0.5)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(plt)

# Function to plot correlation matrix heatmap
def plot_correlation_heatmap():
    st.subheader("Correlation Matrix Heatmap")
    corr = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    st.pyplot(plt)

# Function to plot pair plot
def plot_pairplot():
    st.subheader("Pair Plot")
    sns.pairplot(df)
    st.pyplot(plt)

# Function to plot box plot
def plot_boxplot():
    st.subheader("Box Plot of Features")
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, orient="h")
    st.pyplot(plt)

# Function to plot violin plot
def plot_violinplot():
    st.subheader("Violin Plot of Features")
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df, orient="h", inner="quartile")
    st.pyplot(plt)

# Function to plot PCA explained variance
def plot_pca_variance():
    st.subheader("PCA Explained Variance")
    pca = PCA()
    pca.fit(df)
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    st.pyplot(plt)

# Function to plot K-Means clustering
def plot_kmeans_clusters():
    st.subheader("K-Means Clustering")
    num_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=num_clusters)
    df_transposed = df.T
    kmeans.fit(df_transposed)
    labels = kmeans.labels_
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_transposed)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
    plt.title(f"K-Means Clustering with {num_clusters} Clusters")
    st.pyplot(plt)

# Streamlit UI
st.title("Enhanced Data Visualization Dashboard")
st.write("### Dataset Overview")
st.dataframe(df.head())
st.write("### Descriptive Statistics")
st.dataframe(df.describe().T)

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

# Display selected visualization
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
