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
Organizations like DWB need tools to effectively monitor and anticipate conflict zones for safe worker deployment and resource allocation. This project provides:
- Real-time mapping of conflict events.
- Predictive modeling for future conflict trends.
- Interactive tools for strategic decision-making.

## Features
- **Interactive Real-Time Map**: Displays conflict events with color-coded markers and interactive details.
- **Time Series Forecasting**: Uses the `Prophet` library to predict future conflict trends by region.
- **Risk Level Classification**: Categorizes regions into low, medium, and high risk levels.
- **Clustered Visualization**: Groups conflict data points for improved navigation.
- **Heatmap**: Visualizes the intensity of conflicts across regions.

## Usage Instructions
1. **Run the Notebook**:
   - Open `DoctorDatathon.ipynb` in Jupyter Notebook or a similar environment.
   - Execute the code cells in order.

2. **Access Interactive Maps**:
   - Generate and save interactive maps as HTML files.
   - Download and open `.html` files locally to explore the maps.

## Technical Approach
### Data Extraction and Preprocessing
- **API Integration**: Used `ACLEDExtractor` for efficient data retrieval from the ACLED API with API key authentication.
- **Preprocessing**: Cleaned and formatted data to handle missing values and numerical conversions.

### Visualization
- **Real-Time Map**: Created using `folium`, with interactive markers showing event details.
- **Clustered View**: Implemented `folium.plugins.MarkerCluster` for improved visualization of dense data points.

### Time Series Forecasting
- **Prophet Model**: Forecasted conflict trends for administrative regions, predicting future risk.
- **Risk Classification**: Categorized forecasted risk using quantiles to label regions as low, medium, or high risk.

### Clustering and Heatmap
- **KMeans Clustering**: Grouped regions based on forecasted trends.
- **Heatmap**: Visualized conflict intensity using `folium.plugins.HeatMap`.

## Future Enhancements
- **Real-Time Alerts**: Integrate notifications for emerging conflicts.
- **Enhanced Filters**: Add user-defined filters for event type or region.
- **Additional Data Layers**: Include infrastructure and weather data for comprehensive situational awareness.

