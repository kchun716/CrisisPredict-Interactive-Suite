# CrisisPredict-Interactive-Suite
A comprehensive tool developed to aid humanitarian organizations, such as Doctors Without Borders (DWB), in monitoring, analyzing, and predicting conflict events using ACLED data.

## Project Description
CrisisPredict-Interactive-Suite combines real-time interactive visualization with predictive modeling to provide actionable insights for strategic planning, resource allocation, and worker safety in conflict zones.

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Usage Instructions](#usage-instructions)
- [Technical Approach](#technical-approach)
- [Future Enhancements](#future-enhancements)
- [Contributions](#contributions)
- [License](#license)

## Introduction
Humanitarian missions face significant challenges when operating in conflict-prone areas due to the unpredictability of violence and unrest. CrisisPredict-Interactive-Suite addresses these challenges by providing real-time data and predictive insights, enhancing situational awareness and operational planning.

## Problem Statement
Humanitarian organizations like DWB encounter several key challenges:
- **Limited Situational Awareness**: Without comprehensive, real-time mapping, field teams are unable to assess current conflict zones quickly, leading to suboptimal resource allocation.
- **Lack of Predictive Insights**: The absence of predictive tools hinders the ability to forecast future conflict hotspots, preventing proactive planning.
- **Complex Data Interpretation**: The raw data provided by sources such as ACLED is extensive and complex, making it difficult to extract actionable insights without substantial processing and analysis.

CrisisPredict-Interactive-Suite was developed to tackle these issues by offering a solution that integrates real-time visual data mapping, predictive analytics, and risk-level classifications.

## Features
- **Interactive Real-Time Map**: Displays conflict events with color-coded markers and popups providing event details (e.g., event type, date, fatalities).
- **Time Series Forecasting**: Uses the `Prophet` library to predict future conflict trends for strategic planning.
- **Risk Level Classification**: Categorizes regions into low, medium, and high risk levels based on predictive analytics.
- **Clustered Visualization**: Groups data points for better navigation of high-density conflict regions.
- **Heatmap**: Visualizes the intensity of conflicts across various regions to highlight high-risk zones.

## Usage Instructions
1. **Run the Notebook**:
   - Open `DoctorDatathon.ipynb` in Jupyter Notebook or a similar environment.
   - Execute the code cells in order.

2. **Access Interactive Maps**:
   - Generate and save interactive maps as HTML files.
   - Download and open `.html` files locally to explore the maps.

## Technical Approach
### Data Extraction and Preprocessing
- **API Integration**: Leveraged `ACLEDExtractor` from `msftoolbox` to connect with the ACLED API, ensuring efficient data retrieval. The API was authenticated using an API key and email, with data extraction parameters set for specific time frames and event attributes.
   ```python
   acled_extractor = ACLEDExtractor(
       api_key="your_api_key",
       email="your_email@example.com",
       limit=100000,
       format="csv"
   )
   events_list = acled_extractor.list_events(
       event_date="2023-01-01|2025-01-01",
       event_date_where="BETWEEN",
       fatalities="0",
       fatalities_where=">"
   )
   events = pd.DataFrame(events_list)
   ```
## Visualization
### Interactive Map Creation
Used `folium` to build an interactive real-time map that displays conflict data with detailed popups. The map markers were color-coded based on the severity of events (e.g., red for events with fatalities).
```python
map = folium.Map(location=[events['latitude'].mean(), events['longitude'].mean()], zoom_start=5)
for _, row in events.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='red' if row['fatalities'] > 0 else 'blue',
        fill=True
    ).add_to(map)
map.save('conflict_map.html')
```
## Clustered Markers
Implemented `folium.plugins.MarkerCluster` to cluster data points for high-density areas, improving visualization and navigation.

## Time Series Forecasting
### Modeling with Prophet
Used the `Prophet` library to build time series models for predicting future conflict trends for different administrative regions. Each model was trained on the aggregated fatalities over time, and a 60-day forecast was generated.
```python
model = Prophet(yearly_seasonality=True)
model.fit(loc_data)
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)
```
## Risk Classification
Classified forecasted values into risk categories (low, medium, high) based on quantile thresholds.

```python
def classify_risk(row, lower_quantile, upper_quantile):
    if row['yhat'] < lower_quantile:
        return 'Low Risk'
    elif row['yhat'] < upper_quantile:
        return 'Medium Risk'
    else:
        return 'High Risk'
```

## Clustering and Heatmap
KMeans Clustering
Grouped regions based on forecasted trends using KMeans, which assigned a risk level to each region (low, medium, high).
```python
kmeans = KMeans(n_clusters=3, random_state=2024)
trend_summary['risk_cluster'] = kmeans.fit_predict(clustering_features)
```
## Heatmap Visualization
Used folium.plugins.HeatMap to create a heatmap showing conflict intensity and risk levels across different regions.
```python
HeatMap(heatmap_points, radius=10, blur=15, max_zoom=1).add_to(m)
```

## Future Enhancements
Real-Time Alerts: Integrate real-time notifications to alert users of new conflict developments as they happen.
Enhanced Filtering: Allow users to filter data by specific event types, regions, or timeframes for more tailored insights.
Additional Data Layers: Incorporate related datasets such as infrastructure, weather, and socio-political information for a more comprehensive view of potential challenges.
