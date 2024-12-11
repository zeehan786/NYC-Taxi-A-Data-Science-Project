# File: streamlit_nyc_taxi_app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium import Choropleth, GeoJson
from folium.features import GeoJsonTooltip
from xgboost import XGBRegressor
from branca.colormap import linear
from datetime import datetime

# Load model and shapefile
@st.cache_resource
def load_model_and_data():
    model = XGBRegressor()
    model.load_model("models/xgboost_model_yellow_taxi.json")  # Replace with your model path
    shapefile = gpd.read_file("taxi_zones/taxi_zones.shp")  # Replace with your shapefile path
    return model, shapefile

model, shapefile = load_model_and_data()

# Define tourist zones and airport stations
tourist_zones = {230, 103, 43, 164, 161, 163, 261, 158, 162, 186, 239, 236, 90, 234, 113}
airport_stations = {132, 138, 1, 186, 162, 100}

# Map day numbers to names with Sunday as 1
day_name_map = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}

# Title
st.title("NYC Yellow Taxi Demand Forecasting")
st.sidebar.header("Input Features")

# User inputs
selected_date = st.sidebar.date_input("Select Date", value=datetime.today())
pickup_hour = st.sidebar.slider("Pickup Hour", 0, 23, 12)
is_weather_bad = st.sidebar.checkbox("Is Weather Bad?", False)

# Infer features from the date
dayofmonth = selected_date.day
month = selected_date.month
day_of_week = (selected_date.weekday() + 2)  # Adjust to make Sunday = 1, Monday = 2, ..., Saturday = 7
if day_of_week > 7:
    day_of_week = 1  # Wrap Sunday
is_weekend = day_of_week in [1, 7]  # Sunday or Saturday

# Prepare input features for all PULocationID
all_pulocations = shapefile['OBJECTID'].unique()
input_features = pd.DataFrame({
    'pickup_hour': [pickup_hour] * len(all_pulocations),
    'month': [month] * len(all_pulocations),
    'dayofmonth': [dayofmonth] * len(all_pulocations),
    'day_of_week': [day_of_week] * len(all_pulocations),
    'is_weekend': [int(is_weekend)] * len(all_pulocations),
    'PULocationID': all_pulocations,
    'is_weather_bad': [int(is_weather_bad)] * len(all_pulocations),
})

# Automatically set is_tourist_zone and is_airport_station based on PULocationID
input_features['is_tourist_zone'] = input_features['PULocationID'].apply(lambda x: int(x in tourist_zones))
input_features['is_airport_station'] = input_features['PULocationID'].apply(lambda x: int(x in airport_stations))

# Predict taxi demand including PULocationID
input_features['predicted_count'] = model.predict(input_features)

# Merge predictions with shapefile
shapefile = shapefile.merge(input_features[['PULocationID', 'predicted_count']], left_on='OBJECTID', right_on='PULocationID', how='left')

# Create a folium map
nyc_map = folium.Map(location=[40.7000, -73.8000], zoom_start=10.5)
colormap = linear.YlOrRd_09.scale(shapefile['predicted_count'].min(), shapefile['predicted_count'].max())
colormap.caption = "Predicted Taxi Demand"

# Add interactive tooltips to the map
geojson = GeoJson(
    shapefile,
    style_function=lambda x: {
        'fillColor': colormap(x['properties']['predicted_count']),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=GeoJsonTooltip(
    fields=['zone', 'predicted_count'],  # Use 'zone' for the name and 'predicted_count' for the demand
    aliases=['<b>Zone:</b>', '<b>Taxi Count:</b>'],  # Add bold labels
    localize=True,
    style=("font-size: 14px; font-weight: bold;"),  # Bold text with increased font size
    sticky=True
),
)
geojson.add_to(nyc_map)
colormap.add_to(nyc_map)

# Display the map
st.write("### Heatmap of Predicted Yellow Taxi Demand")
st.components.v1.html(nyc_map._repr_html_(), height=600)
