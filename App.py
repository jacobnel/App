import pandas as pd
import numpy as np
import streamlit as st
import joblib
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium

# Load your saved model and cluster centroids
model = joblib.load("pricing_model.pkl")
cluster_centroids = pd.read_csv("cluster_centroids.csv")

# Cluster assignment function
def assign_cluster(lat, lon):
    min_dist = float('inf')
    assigned_cluster = '-1'
    for idx, row in cluster_centroids.iterrows():
        dist = geodesic((lat, lon), (row['latitude'], row['longitude'])).miles
        if dist < min_dist:
            min_dist = dist
            assigned_cluster = row['cluster']
    return assigned_cluster

# Streamlit App
st.title("Real Estate Interactive Pricing Model")

st.write("\n**Step 1: Select property location (click on map)**")

# Create map with click support
m = folium.Map(location=[28.55, -81.60], zoom_start=12)
m.add_child(folium.LatLngPopup())
map_data = st_folium(m, width=700, height=500)

# Handle map click
if map_data and map_data['last_clicked']:
    lat = map_data['last_clicked']['lat']
    lon = map_data['last_clicked']['lng']
    st.success(f"Selected Coordinates: {lat:.5f}, {lon:.5f}")
    
    st.write("\n**Step 2: Enter property details**")
    
    city = st.text_input("City", "Winter Garden")
    product_style = st.text_input("Product Style", "Single-Family")
    unit_size = st.number_input("Unit Size (sqft)", 500, 10000, 2500)
    subdivision = st.text_input("Subdivision", "")
    seller = st.text_input("Seller", "The Pulte Group")
    zip_code = st.text_input("Zip Code", "34787")

    if st.button("Predict Sale Price"):
        location_cluster = assign_cluster(lat, lon)
        input_df = pd.DataFrame([{
            'city': city,
            'product_style': product_style,
            'unit_size': unit_size,
            'subdivision': subdivision,
            'seller': seller,
            'zip_code': zip_code,
            'location_cluster': str(location_cluster)
        }])
        
        predicted_log_price = model.predict(input_df)[0]
        predicted_price = np.expm1(predicted_log_price)
        
        st.subheader(f"Estimated Sale Price: ${predicted_price:,.2f}")

else:
    st.warning("Click on the map to select property location.")

