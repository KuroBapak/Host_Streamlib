import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Load the dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df['datetime'] = pd.to_datetime(df['tgl'] + ' ' + df['ot'], errors='coerce')
    df['year'] = df['datetime'].dt.year
    return df

df = load_data()

# Page title and description
st.title("Indonesian Earthquake Data Analysis")
st.write("Analysis of earthquakes in Indonesia from 2008 to 2023 with 92,887 records and 13 columns.")

# Sidebar filters
st.sidebar.header("Filter The Data")

# Year Filter
min_year, max_year = int(df['year'].min()), int(df['year'].max())
start_year, end_year = st.sidebar.slider("Select Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

# Depth Range Filter
min_depth, max_depth = int(filtered_df['depth'].min()), int(filtered_df['depth'].max())
depth_range = st.sidebar.slider("Select Depth Range (km)", min_value=min_depth, max_value=max_depth, value=(min_depth, max_depth))
filtered_df = filtered_df[(filtered_df['depth'] >= depth_range[0]) & (filtered_df['depth'] <= depth_range[1])]

# Region Filter (using the 'remark' column)
all_regions = sorted(filtered_df['remark'].unique())
selected_regions = st.sidebar.multiselect("Select Region(s)", options=all_regions, default=all_regions)
filtered_df = filtered_df[filtered_df['remark'].isin(selected_regions)]

# Display Descriptive Statistics
st.header("Descriptive Statistics")
st.write(filtered_df[['depth', 'mag']].describe())
st.markdown("""
**Explanation:**  
This table summarizes key statistics such as count, mean, standard deviation, minimum, and maximum values for earthquake depth and magnitude. These metrics help you understand the central tendency and variability of the seismic data.
""")

# Visualization 1: Magnitude Distribution (Histogram with KDE)
st.header("Magnitude Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_df['mag'], bins=30, kde=True, color='skyblue', ax=ax)
ax.set_xlabel("Magnitude")
ax.set_ylabel("Frequency")
st.pyplot(fig)
st.markdown("""
**Explanation:**  
This histogram with a KDE overlay shows the distribution of earthquake magnitudes. The bars represent the frequency of earthquakes for different magnitude ranges, and the KDE curve approximates the probability density function. It helps in understanding how likely different magnitudes are to occur.
""")

# Advanced Visualization: Cumulative Distribution Function (CDF) of Magnitude
st.header("Cumulative Distribution Function (CDF) of Magnitude")
sorted_mag = np.sort(filtered_df['mag'])
cdf = np.arange(len(sorted_mag)) / float(len(sorted_mag))
fig_cdf, ax_cdf = plt.subplots()
ax_cdf.plot(sorted_mag, cdf, marker='.', linestyle='none')
ax_cdf.set_xlabel("Magnitude")
ax_cdf.set_ylabel("Cumulative Probability")
ax_cdf.set_title("CDF of Earthquake Magnitudes")
st.pyplot(fig_cdf)
st.markdown("""
**Explanation:**  
The CDF plot shows the cumulative probability of earthquake magnitudes up to a given value. For instance, you can estimate the probability that an earthquake will have a magnitude below a certain threshold, which is useful for risk assessment.
""")

# Visualization 2: Depth vs. Magnitude Scatter Plot with Regression
st.header("Depth vs. Magnitude")
fig2 = px.scatter(filtered_df, x='depth', y='mag', trendline='ols', title="Depth vs. Magnitude")
st.plotly_chart(fig2)
st.markdown("""
**Explanation:**  
This scatter plot examines the relationship between earthquake depth and magnitude. The regression trendline (using ordinary least squares) indicates whether there is a correlation between how deep an earthquake occurs and its magnitude. For example, if the trendline is downward sloping, it suggests that deeper earthquakes might tend to have lower magnitudes.
""")

# Visualization 3: Time Series of Earthquake Magnitudes
st.header("Magnitude Over Time")
fig3 = px.line(filtered_df, x='datetime', y='mag', title="Earthquake Magnitudes Over Time")
st.plotly_chart(fig3)
st.markdown("""
**Explanation:**  
This time series line chart shows how earthquake magnitudes vary over time. It helps in identifying trends, seasonal patterns, or potential anomalies in seismic activity over the years.
""")

# Advanced Time Series: Rolling Average of Magnitude (30-day window)
st.header("Rolling Average (30-day) of Earthquake Magnitudes")
# Resample only the 'mag' column after setting 'datetime' as index
time_series = filtered_df[['datetime', 'mag']].set_index('datetime').resample('D').mean()
time_series['rolling_mag'] = time_series['mag'].rolling(window=30).mean()
fig_roll, ax_roll = plt.subplots()
ax_roll.plot(time_series.index, time_series['mag'], label='Daily Average', alpha=0.5)
ax_roll.plot(time_series.index, time_series['rolling_mag'], label='30-Day Rolling Mean', color='red')
ax_roll.set_xlabel("Date")
ax_roll.set_ylabel("Magnitude")
ax_roll.set_title("Rolling Average of Earthquake Magnitudes")
ax_roll.legend()
st.pyplot(fig_roll)
st.markdown("""
**Explanation:**  
The rolling average plot smooths out daily fluctuations in earthquake magnitudes using a 30-day window. This helps reveal longer-term trends and patterns in seismic activity that might be obscured by daily variability.
""")

# Visualization 4: Geographical Distribution
st.header("Geographical Distribution")
st.map(filtered_df[['lat', 'lon']])
st.markdown("""
**Explanation:**  
This map displays the geographic locations of earthquakes across Indonesia. It visually highlights regions with high seismic activity, aiding in spatial analysis and risk assessment.
""")

# Advanced Visualization: Correlation Heatmap
st.header("Correlation Heatmap")
numeric_cols = filtered_df[['depth', 'mag']]
corr = numeric_cols.corr()
fig_corr, ax_corr = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
ax_corr.set_title("Correlation Heatmap")
st.pyplot(fig_corr)
st.markdown("""
**Explanation:**  
The correlation heatmap shows the strength and direction of the linear relationship between earthquake depth and magnitude. A correlation coefficient closer to 1 or -1 indicates a strong relationship, whereas a value near 0 indicates little or no linear correlation.
""")

# Conclusion Section
st.header("Conclusion and Implications")
st.markdown("""
**Conclusion:**  
This comprehensive analysis of Indonesian earthquake data reveals several key insights:  
- **Magnitude Distribution:** The histogram and CDF indicate how frequently earthquakes of various magnitudes occur, which is crucial for understanding seismic risk.  
- **Depth-Magnitude Relationship:** The scatter plot with a regression line provides insight into how earthquake depth might influence the magnitude, which can be significant for seismological research and hazard assessment.  
- **Temporal Trends:** The time series and rolling average analyses help identify trends and anomalies over time, indicating periods of increased seismic activity.  
- **Spatial Distribution:** The map visualization highlights regions with high earthquake density, which is vital for emergency preparedness and regional planning.  
- **Correlation Analysis:** The heatmap confirms the relationships between key variables, which is helpful for further statistical modeling and risk forecasting.

**Implications and Problems Addressed:**  
- **Risk Assessment and Mitigation:** By understanding the distribution and frequency of earthquakes, authorities can better prepare for seismic events and allocate resources for emergency response.  
- **Urban and Infrastructure Planning:** The insights into spatial distribution and depth-magnitude relationships assist in planning safer urban developments and infrastructure improvements in high-risk areas.  
- **Scientific Research:** The statistical relationships and temporal trends provide valuable data for seismologists and researchers working on earthquake prediction models.  
- **Public Awareness and Policy Making:** Clear visualizations and detailed analysis can support educational initiatives and inform policy decisions aimed at reducing earthquake-related risks.

Overall, this dashboard not only offers a detailed statistical and probabilistic overview of seismic activity in Indonesia but also serves as a powerful tool for addressing practical challenges related to earthquake risk management and preparedness.
""")

