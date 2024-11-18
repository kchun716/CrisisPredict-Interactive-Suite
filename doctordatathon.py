

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install git+https://github.com/MSF-Collaborate/msf-toolbox.git@features/acled_addition
# !pip install prophet

from msftoolbox.acled.extract import ACLEDExtractor
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
import folium

"""**Initialization**"""

acled_extractor = ACLEDExtractor(
    api_key="ESkuYk0zNSuhPHhP5gH4",
    email="tpham.27@berkeley.edu",
    limit = 100000,
    format="csv"
)

events_list = acled_extractor.list_events(
    event_date="2023-01-01|2025-01-01",
    event_date_where="BETWEEN",
    fatalities="0",
    fatalities_where=">"
)

events = pd.DataFrame(events_list)

"""**EDA**"""

len(events)

# Preprocessing

events['event_date'] = pd.to_datetime(events['event_date'])
fix_event_date = events['event_date']

for col in events.columns:
    try:
        events[col] = pd.to_numeric(events[col], errors='raise')
    except ValueError:
        pass

events['event_date'] = fix_event_date

with pd.option_context('display.max_columns', None):
    print(events.head())

# Shows the columns that have empty rows, and the # of empty rows for that column
for col in events.columns:
  empty_sum = sum(events[col]=='')
  if empty_sum > 0:
    print(col + ": " + str(empty_sum))

events = events[events['admin1'] != '']

map = folium.Map(location=[events['latitude'].mean(), events['longitude'].mean()], zoom_start=5)
for _, row in events.iloc[:999, :].iterrows():
    folium.CircleMarker(location=[row['latitude'], row['longitude']],
                        radius=5,
                        color='red',
                        fill=True).add_to(map)
map.save('conflict_map.html')

map

"""**Time Series Analysis**"""

events['week_start'] = events['event_date'] - pd.to_timedelta(events['event_date'].dt.weekday, unit='D')
events['week_start'] = events['event_date']

# Aggregate events by admin1 and location, count the number of events (or fatalities)
events_agg = events.groupby(['admin1', 'week_start'])['fatalities'].sum().reset_index(name='fatality_sum')
events_agg.head()

admin1_agg = events_agg.groupby('admin1').filter(lambda sf: len(sf)>0).groupby('admin1').size()
admin1_agg

bins = np.arange(0, max(admin1_agg) + 50, 50)

sns.histplot(admin1_agg, bins=bins)

plt.title("Frequency of Administrative Regions Having Fatalities")
plt.xlabel("# of Events")
plt.ylabel("Frequency (# of Administrative Regions)")

plt.show()

events_agg = events_agg.groupby('admin1').filter(lambda sf: len(sf)>50)

from prophet import Prophet

region_values = events_agg['admin1'].unique()

forecast_results = []

for admin1 in region_values:
    loc_data = events_agg[events_agg['admin1'] == admin1][['week_start', 'fatality_sum']]
    loc_data = loc_data.rename(columns={'week_start': 'ds', 'fatality_sum': 'y'})  # Prophet format

    loc_data = loc_data.dropna(subset=['ds', 'y'])


    model = Prophet(yearly_seasonality=True)
    model.fit(loc_data)

    future = model.make_future_dataframe(periods=60)

    forecast = model.predict(future)

    forecast['admin1'] = admin1

    forecast_results.append(forecast)

all_forecasts = pd.concat(forecast_results)

all_forecasts.head()

i = 3
selected_admin1 = all_forecasts['admin1'].unique()[i]

filtered_data = all_forecasts[all_forecasts['admin1']=='selected_admin1']
model.plot_components(filtered_data)

def classify_risk(row, lower_quantile, upper_quantile):
    if row['yhat'] < lower_quantile:
        return 'Low Risk'
    elif row['yhat'] < upper_quantile:
        return 'Medium Risk'
    elif row['yhat'] >= upper_quantile:
        return 'High Risk'
    else:
        return 'Extreme Risk'

lower_quantile = all_forecasts['yhat_lower'].quantile(0.25)
upper_quantile = all_forecasts['yhat_upper'].quantile(0.75)

all_forecasts['predicted_risk_level'] = all_forecasts.apply(classify_risk, axis=1,
                                                           lower_quantile=lower_quantile,
                                                           upper_quantile=upper_quantile)

print(all_forecasts[['ds', 'admin1', 'yhat', 'predicted_risk_level']].head())

all_forecasts[all_forecasts['predicted_risk_level']=='Low Risk']['admin1'].value_counts()

all_forecasts=all_forecasts[all_forecasts['admin1'] != "Taraba"]
sns.histplot(all_forecasts.groupby('admin1')['yhat'].mean())

def compute_trend(group):
    X = (group['ds'] - group['ds'].min()).dt.days.values.reshape(-1, 1)  # Convert dates to days since min date
    y = group['yhat'].values

    model = LinearRegression()
    model.fit(X, y)

    return model.coef_[0]

trend_summary = (
    all_forecasts.groupby('admin1')
    .apply(lambda group: pd.Series({
        'yhat_mean': group['yhat'].mean(),
        'yhat_trend': compute_trend(group),
    }))
    .reset_index()
)

clustering_features = trend_summary[['yhat_mean', 'yhat_trend']]

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=2024)  # 3 risk levels: low, medium, high
trend_summary['risk_cluster'] = kmeans.fit_predict(clustering_features)

cluster_map = {
    0: 'Low Risk',
    1: 'Medium Risk',
    2: 'High Risk',
}
trend_summary['admin1_risk_level'] = trend_summary['risk_cluster'].map(cluster_map)

events_with_risk = events.merge(
    trend_summary[['admin1', 'admin1_risk_level']], on='admin1', how='left'
)

events_with_risk.head()

test=all_forecasts[all_forecasts['admin1']=='Taraba'].groupby('admin1').agg(
        yhat_mean=('yhat', 'mean'),
        yhat_trend=('yhat', lambda x: (x.iloc[-1] - x.iloc[0]) / len(x)),  # Rate of increase
        yhat_spread=('yhat_upper', lambda y: y.mean() - all_forecasts.loc[y.index, 'yhat_lower'].mean())  # Spread
    ).reset_index()
test

events_with_risk['admin1_risk_level'] = events_with_risk['admin1_risk_level'].fillna('No Risk')
events_with_risk.groupby('admin1').first()['admin1_risk_level'].value_counts()

import folium
from folium.plugins import MarkerCluster

latitude_mean = events_with_risk['latitude'].mean()
longitude_mean = events_with_risk['longitude'].mean()

m = folium.Map(location=[latitude_mean, longitude_mean], zoom_start=6)

marker_cluster = MarkerCluster().add_to(m)

cluster_colors = {
    'No Risk': 'green',
    'Low Risk': 'blue',
    'Medium Risk': 'orange',
    'High Risk': 'red',
}

for _, row in events_with_risk.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Admin1: {row['admin1']}<br>Risk Level: {row['admin1_risk_level']}",
        icon=folium.Icon(color=cluster_colors[row['admin1_risk_level']])
    ).add_to(marker_cluster)

m.save("risk_clusters_map.html")
m

from folium.plugins import HeatMap

heatmap_data = events_with_risk[['latitude', 'longitude', 'admin1_risk_level']]

risk_level_mapping = {
    'No Risk': 1,
    'Low Risk': 2,
    'Medium Risk': 3,
    'High Risk': 4
}
heatmap_data['risk_numeric'] = heatmap_data['admin1_risk_level'].map(risk_level_mapping)

heatmap_points = heatmap_data[['latitude', 'longitude', 'risk_numeric']].values.tolist()

m = folium.Map(location=[heatmap_data['latitude'].mean(), heatmap_data['longitude'].mean()], zoom_start=6)

HeatMap(heatmap_points, radius=10, blur=15, max_zoom=1).add_to(m)

m.save("heatmap.html")

from IPython.display import IFrame
m

import IPython

IPython.display.HTML(filename='/content/real_time_conflict_map (2).html')

acled_extractor = ACLEDExtractor(
    api_key="88KWzjVrWU-1f-7Nguud",
    email="kc716@berkeley.edu",
    limit = 100000,
    format="csv"
)

import folium
from folium.plugins import HeatMap

events_list = acled_extractor.list_events(
    event_date="2022-01-01|2024-01-01",
    event_date_where="BETWEEN",
    fatalities="0",
    fatalities_where=">"
)
events = pd.DataFrame(events_list)
events.to_csv('events_data_full.csv', index=False)
events_df = pd.read_csv('events_data_full.csv')
events_df['event_date'] = pd.to_datetime(events_df['event_date'])
events_df = events_df[['event_date', 'latitude', 'longitude', 'event_type', 'fatalities', 'country', 'region']]

map_real_time = folium.Map(location=[events_df['latitude'].mean(), events_df['longitude'].mean()], zoom_start=3)

for _, row in events_df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='blue' if row['fatalities'] == 0 else 'red',
        popup=f"Event Type: {row['event_type']}, Date: {row['event_date'].date()}, Fatalities: {row['fatalities']}",
        fill=True,
        fill_opacity=0.6
    ).add_to(map_real_time)

map_real_time.save('real_time_conflict_map.html')



"""STEPs to run the interactive map:

1. Go to the notebook Files and tract "real_time_conflict_map.html"

2. Download the html file and open on local device.

3. Run the the html file!
"""