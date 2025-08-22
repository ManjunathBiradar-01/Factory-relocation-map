import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import AntPath

st.set_page_config(page_title="Factory Volume Shift Dashboard", layout="wide")

# Load data from Excel
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df_from = pd.read_excel(path, sheet_name="From", engine="openpyxl")
    df_to = pd.read_excel(path, sheet_name="To", engine="openpyxl")
    df_sub = pd.read_excel(path, sheet_name="Sub-Factory", engine="openpyxl")

    for df in [df_from, df_to, df_sub]:
        df.columns = df.columns.str.strip()

    df_from = df_from.rename(columns={"Latitude": "Lat_from", "Longitude": "Lon_from"})
    df_to = df_to.rename(columns={"Latitude": "Lat_to", "Longitude": "Lon_to"})
    df_sub = df_sub.rename(columns={"Latitude": "Lat_sub", "Longitude": "Lon_sub"})

    merged = df_from.merge(df_to[["FM", "Plan Lead Factory", "Lat_to", "Lon_to"]], on="FM", how="left")
    merged = merged.merge(df_sub[["FM", "Plan Sub Factory", "Lat_sub", "Lon_sub"]], on="FM", how="left")

    return merged

# Upload Excel file
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.stop()

# Center map
coords = []
for col in ["Lat_from", "Lon_from", "Lat_to", "Lon_to", "Lat_sub", "Lon_sub"]:
    if col in df.columns:
        coords.extend(df[col].dropna().tolist())

center_lat = np.mean([coords[i] for i in range(0, len(coords), 2)]) if coords else 20.0
center_lon = np.mean([coords[i] for i in range(1, len(coords), 2)]) if coords else 0.0

# Create map
m = folium.Map(location=[center_lat, center_lon], zoom_start=2)

# Plot arrows
for _, row in df.iterrows():
    if pd.notnull(row["Lat_from"]) and pd.notnull(row["Lon_from"]):
        folium.Marker([row["Lat_from"], row["Lon_from"]], tooltip="Factory Today").add_to(m)
    if pd.notnull(row["Lat_to"]) and pd.notnull(row["Lon_to"]):
        folium.Marker([row["Lat_to"], row["Lon_to"]], tooltip="Lead Factory").add_to(m)
        if pd.notnull(row["Lat_from"]) and pd.notnull(row["Lon_from"]):
            AntPath([[row["Lat_from"], row["Lon_from"]], [row["Lat_to"], row["Lon_to"]]], color="blue").add_to(m)
    if pd.notnull(row["Lat_sub"]) and pd.notnull(row["Lon_sub"]):
        folium.Marker([row["Lat_sub"], row["Lon_sub"]], tooltip="Sub Factory", icon=folium.Icon(color="green")).add_to(m)
        if pd.notnull(row["Lat_to"]) and pd.notnull(row["Lon_to"]):
            AntPath([[row["Lat_to"], row["Lon_to"]], [row["Lat_sub"], row["Lon_sub"]]], color="green").add_to(m)

# Display map
st.title("Factory Volume Shift Map")
st.components.v1.html(m._repr_html_(), height=600)







































































