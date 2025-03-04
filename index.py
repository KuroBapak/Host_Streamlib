import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df['datetime'] = pd.to_datetime(df['tgl'] + ' ' + df['ot'], errors='coerce')
    df['year'] = df['datetime'].dt.year
    return df

df = load_data()

# Page title
st.title("Indonesian Earthquake Data Analysis")
st.write("Dataset of earthquakes in Indonesia from 2008 to 2023. with 92,887 earthquake records with 13 columns")

# Year Filter
st.sidebar.header("Filter Data by Year")
min_year, max_year = int(df['year'].min()), int(df['year'].max())
start_year, end_year = st.sidebar.slider("Select Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

# Descriptive Statistics
st.header("Descriptive Statistics")
st.write(filtered_df[['depth', 'mag']].describe())

# Visualization 1: Magnitude Distribution
st.header("Magnitude Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_df['mag'], bins=30, kde=True, color='skyblue', ax=ax)
ax.set_xlabel("Magnitude")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Visualization 2: Depth vs. Magnitude
st.header("Depth vs. Magnitude")
fig2 = px.scatter(filtered_df, x='depth', y='mag', trendline='ols', title="Depth vs. Magnitude")
st.plotly_chart(fig2)

# Visualization 3: Time Series of Earthquake Magnitudes
st.header("Magnitude Over Time")
fig3 = px.line(filtered_df, x='datetime', y='mag', title="Earthquake Magnitudes Over Time")
st.plotly_chart(fig3)

# Visualization 4: Geographical Distribution
st.header("Geographical Distribution")
st.map(filtered_df[['lat', 'lon']])