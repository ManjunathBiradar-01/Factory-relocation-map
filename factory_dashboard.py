import streamlit as st
import pandas as pd
import numpy as np
import folium
import io

st.set_page_config(page_title="Bomag SDMs Factory Production Relocation Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df_from = pd.read_excel(path, sheet_name="From", engine="openpyxl")
    df_to = pd.read_excel(path, sheet_name="To", engine="openpyxl")
    df_val = pd.read_excel(path, sheet_name="Values", engine="openpyxl")

    for d in (df_from, df_to, df_val):
        d.columns = d.columns.str.strip()

    df_from = df_from.rename(columns={"Latitude": "Lat_today", "Longitude": "Lon_today"})
    df_to = df_to.rename(columns={"Latitude": "Lat_lead", "Longitude": "Lon_lead"})

    df_to_keep = df_to[["FM", "Plan Lead Factory", "Lat_lead", "Lon_lead"]].copy()
    df_val_keep = df_val[["FM", "Volume Lead Plant (%)"]].copy()

    merged = df_from.merge(df_to_keep, on="FM", how="left").merge(df_val_keep, on="FM", how="left")

    for c in ["Lat_today", "Lon_today", "Lat_lead", "Lon_lead"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    return merged

def find_sales_region_col(columns):
    normalized = {c.lower().strip(): c for c in columns}
    candidates = [
        "sales region", "main sales region", "mainsales region",
        "mainsalesregion", "salesregion", "main_sales_region"
    ]
    for key in candidates:
        if key in normalized:
            return normalized[key]
    for c in columns:
        cl = c.lower()
        if "sales" in cl and "region" in cl:
            return c
    return None

def format_coords(lat, lon, decimals: int = 5) -> str:
    if pd.notnull(lat) and pd.notnull(lon):
        return f"{lat:.{decimals}f}, {lon:.{decimals}f}"
    return "n/a"

# Sidebar
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"], key="file_uploader")
edit_mode = st.sidebar.button("Edit Dataset", key="edit_button")

# Load data
try:
    if uploaded_file:
        df = load_data(uploaded_file)
    else:
        df = load_data("https://raw.githubusercontent.com/ManjunathBiradar-01/Factory-relocation-map/main/Footprint_SDR.xlsx")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Edit dataset
if edit_mode:
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.subheader("Edit Full Dataset")
        edited_df = st.data_editor(df, num_rows="dynamic", key="data_editor")
        if st.button("Download Updated Excel File", key="download_button"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                edited_df.to_excel(writer, index=False, sheet_name='UpdatedData')
            st.download_button(
                label="Click to Download",
                data=output.getvalue(),
                file_name="updated_factory_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_link"
            )
    else:
        st.warning("No data available to edit.")

# Main dashboard
st.title("Factory Production Relocation Dashboard")
sales_region_col = find_sales_region_col(df.columns)

# Filters
c1, c2, c3, c4 = st.columns(4)
with c1:
    machine_code_filter = st.multiselect("Machine Code (FM)", sorted(df["FM"].dropna().astype(str).unique()))
with c2:
    machine_name_filter = st.multiselect("Machine Name", sorted(df["Name"].dropna().astype(str).unique()))
with c3:
    engine_filter = st.multiselect("Select Engine Type", sorted(df["Engine"].dropna().astype(str).unique()))
with c4:
    emission_filter = st.multiselect("Select Emission Level", sorted(df["Emission"].dropna().astype(str).unique()))

if sales_region_col:
    sales_region_filter = st.multiselect("Sales Region", sorted(df[sales_region_col].dropna().astype(str).unique()))
else:
    sales_region_filter = []
    st.info("Sales Region column not found.")

# Apply filters
filtered_df = df.copy()
if machine_code_filter:
    filtered_df = filtered_df[filtered_df["FM"].astype(str).isin(machine_code_filter)]
if machine_name_filter:
    filtered_df = filtered_df[filtered_df["Name"].astype(str).isin(machine_name_filter)]
if engine_filter:
    filtered_df = filtered_df[filtered_df["Engine"].astype(str).isin(engine_filter)]
if emission_filter:
    filtered_df = filtered_df[filtered_df["Emission"].astype(str).isin(emission_filter)]
if sales_region_col and sales_region_filter:
    filtered_df = filtered_df[filtered_df[sales_region_col].astype(str).isin(sales_region_filter)]

# Map centering
coords = []
if not filtered_df.empty:
    if "Lat_today" in filtered_df.columns and "Lon_today" in filtered_df.columns:
        coords.extend(filtered_df[["Lat_today", "Lon_today"]].dropna().values.tolist())
    if "Lat_lead" in filtered_df.columns and "Lon_lead" in filtered_df.columns:
        coords.extend(filtered_df[["Lat_lead", "Lon_lead"]].dropna().values.tolist())

if coords:
    center_lat = float(np.mean([c[0] for c in coords]))
    center_lon = float(np.mean([c[1] for c in coords]))
else:
    center_lat, center_lon = 20.0, 0.0

# Map
m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles="OpenStreetMap")
st.subheader("Production Relocation Map")
st.components.v1.html(m._repr_html_(), height=600)






































































