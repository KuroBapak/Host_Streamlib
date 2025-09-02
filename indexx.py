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

# Additional Analysis: High-Risk Earthquakes for Building Resilience
st.header("High-Risk Earthquakes for Building Resilience")
st.markdown("""
**Explanation:**  
For designing structures to withstand earthquakes, it's crucial to identify high-risk events. Earthquakes that occur at shallow depths often generate stronger ground shaking, particularly when the magnitude is high. By adjusting the critical thresholds for depth and magnitude, we can pinpoint these high-risk events which should be considered when updating building codes.
""")

# Sliders for critical thresholds
critical_depth = st.slider("Critical Depth Threshold (km)", 
                           min_value=int(filtered_df['depth'].min()), 
                           max_value=int(filtered_df['depth'].max()), value=30)
critical_mag = st.slider("Critical Magnitude Threshold", 
                         min_value=float(filtered_df['mag'].min()), 
                         max_value=float(filtered_df['mag'].max()), value=5.5)

# Identify high-risk events
high_risk_df = filtered_df[(filtered_df['depth'] <= critical_depth) & (filtered_df['mag'] >= critical_mag)]
st.write(f"Number of high-risk events (Depth <= {critical_depth} km and Magnitude >= {critical_mag}): {len(high_risk_df)}")

# Plot the depth vs. magnitude scatter with high-risk events highlighted
fig_hr, ax_hr = plt.subplots()
sns.scatterplot(data=filtered_df, x='depth', y='mag', color='blue', label='Other Events', ax=ax_hr)
sns.scatterplot(data=high_risk_df, x='depth', y='mag', color='red', label='High-Risk Events', ax=ax_hr)
ax_hr.axvline(critical_depth, color='red', linestyle='--', label='Critical Depth Threshold')
ax_hr.axhline(critical_mag, color='red', linestyle='--', label='Critical Magnitude Threshold')
ax_hr.set_xlabel("Depth (km)")
ax_hr.set_ylabel("Magnitude")
ax_hr.set_title("High-Risk Earthquake Events (for Building Requirements)")
ax_hr.legend()
st.pyplot(fig_hr)
st.markdown("""
**Discussion:**  
Earthquakes occurring at shallow depths with high magnitudes pose the greatest risk to structures. These events produce intense ground shaking, which can lead to catastrophic damage if buildings are not designed to withstand such forces. This analysis helps identify conditions under which enhanced building standards and earthquake-resistant construction techniques are most critical.
""")

# New Section: Recommended Pillar Depth for Earthquake-Resistant Structures
st.header("Recommended Pillar Depth for Earthquake-Resistant Structures")
st.markdown("""
**Explanation:**  
For earthquake-resistant design, it's essential to have foundations that are deep enough to provide stability during seismic shaking. Based on simplified engineering guidelines (and supported by various studies and building codes), a rule of thumb is:
    
    **Recommended Pillar Depth (m) = (Magnitude - 3) + 2  (for Magnitude ≥ 3)**
    
This means that as the earthquake magnitude increases, the recommended minimum embedment depth of the foundation should also increase. For example:
- A magnitude 4 event suggests a minimum depth of ~3 m.
- A magnitude 6 event suggests a minimum depth of ~5 m.
- A magnitude 7 event suggests a minimum depth of ~6 m.
    
These values are illustrative and must be adjusted according to local soil conditions, building requirements, and safety factors.
""")

# Create recommended pillar depth data for a range of magnitudes
magnitude_range = np.linspace(3, 8, 50)  # Considering magnitudes from 3 to 8
recommended_depth = (magnitude_range - 3) + 2

# Plot the recommended pillar depth vs earthquake magnitude
fig_rec, ax_rec = plt.subplots()
ax_rec.plot(magnitude_range, recommended_depth, color='green', label='Recommended Pillar Depth')
ax_rec.set_xlabel("Earthquake Magnitude")
ax_rec.set_ylabel("Recommended Pillar Depth (m)")
ax_rec.set_title("Recommended Minimum Pillar Depth vs Earthquake Magnitude")
ax_rec.legend()
st.pyplot(fig_rec)

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
This rolling average plot smooths out daily fluctuations in earthquake magnitudes using a 30-day window, revealing longer-term trends in seismic activity.
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
The correlation heatmap shows the strength and direction of the linear relationship between earthquake depth and magnitude. A coefficient near 1 or -1 indicates a strong relationship, while a value near 0 indicates little or no linear correlation.
""")

# Conclusion Section
st.header("Conclusion and Implications")
st.markdown("""
**Conclusion:**  
This comprehensive analysis of Indonesian earthquake data reveals several key insights:
- **Magnitude Distribution:** The histogram and CDF illustrate the frequency and cumulative probability of different earthquake magnitudes, which is vital for risk assessment.
- **Depth-Magnitude Relationship:** The scatter plot with regression provides insight into how earthquake depth might affect magnitude. Notably, shallow, high-magnitude events (highlighted in our high-risk analysis) are critical for building resilience.
- **Building Resilience:** The additional analysis on recommended pillar depth suggests that as earthquake magnitude increases, deeper foundations (pillars) may be necessary to improve structural stability. Although simplified, this guideline can support preliminary design considerations for earthquake-resistant structures.
- **Temporal Trends:** Time series and rolling average analyses reveal trends and anomalies in seismic activity over time.
- **Spatial Distribution:** The geographic map helps identify regions with high earthquake density, crucial for regional planning and emergency response.
- **Correlation Analysis:** The heatmap confirms the relationship between key variables, supporting further statistical modeling.

**Implications and Applications:**  
- **Risk Mitigation:** Understanding the distribution of earthquake magnitudes and the depth-magnitude relationship can inform risk assessment and emergency preparedness strategies.
- **Urban Planning and Building Codes:** Insights from the recommended pillar depth analysis can guide building code revisions and foundation design, particularly in regions prone to shallow, high-magnitude earthquakes.
- **Engineering Research:** The visualizations provide a basis for more detailed studies on the seismic performance of structures, paving the way for advanced engineering models.
- **Public Policy and Awareness:** Clear visualizations and data-driven insights support educational initiatives and policy decisions aimed at reducing earthquake-related risks.

Overall, this dashboard not only delivers a detailed statistical and probabilistic overview of seismic activity in Indonesia but also offers actionable insights for enhancing building resilience in earthquake-prone areas.
""")
