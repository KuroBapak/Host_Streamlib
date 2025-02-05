import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load dataset without header
file_path = "data.csv"
df = pd.read_csv(file_path, header=None)

# Add column names based on index
df.columns = [f"Feature_{i}" for i in range(df.shape[1])]

def plot_heatmap():
    st.subheader("Heatmap of Dataset")
    plt.figure(figsize=(12, 5))
    sns.heatmap(df, cmap="coolwarm", annot=False, cbar=True)
    st.pyplot(plt)

def plot_pca():
    st.subheader("PCA Scatter Plot")
    df_transposed = df.T
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_transposed)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(plt)

def plot_histogram():
    st.subheader("Histogram of Dataset Values")
    plt.figure(figsize=(10, 6))
    plt.hist(df.values.flatten(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    st.pyplot(plt)

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

# Streamlit UI
st.title("Data Visualization Dashboard")
st.write("### Dataset Overview")
st.dataframe(df.head())
st.write("### Descriptive Statistics")
st.dataframe(df.describe().T)

# Add visualizations
plot_heatmap()
plot_pca()
plot_histogram()
plot_population_vs_sample()

st.write("Developed with ❤️ using Streamlit")
st.write("1. Rayhan Roshidi Nasrulloh")
st.write("2. Syah Reza Falevi")
st.write("3. Reza Fahlevi")
