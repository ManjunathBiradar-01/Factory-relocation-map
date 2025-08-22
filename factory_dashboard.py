import pandas as pd
import numpy as np
import folium
import io
from folium.plugins import AntPath

st.set_page_config(page_title="Bomag SDMs Factory Production Relocation Dashboard", layout="wide")
st.set_page_config(page_title="Factory Volume Shift Dashboard", layout="wide")

# Load data from Excel
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df_from = pd.read_excel(path, sheet_name="From", engine="openpyxl")
    df_to = pd.read_excel(path, sheet_name="To", engine="openpyxl")
    df_val = pd.read_excel(path, sheet_name="Values", engine="openpyxl")
    df_sub = pd.read_excel(path, sheet_name="Sub-Factory", engine="openpyxl")

    for d in (df_from, df_to, df_val):
        d.columns = d.columns.str.strip()
    for df in [df_from, df_to, df_sub]:
        df.columns = df.columns.str.strip()

    df_from = df_from.rename(columns={"Latitude": "Lat_today", "Longitude": "Lon_today"})
    df_to = df_to.rename(columns={"Latitude": "Lat_lead", "Longitude": "Lon_lead"})
    df_from = df_from.rename(columns={"Latitude": "Lat_from", "Longitude": "Lon_from"})
    df_to = df_to.rename(columns={"Latitude": "Lat_to", "Longitude": "Lon_to"})
    df_sub = df_sub.rename(columns={"Latitude": "Lat_sub", "Longitude": "Lon_sub"})

    df_to_keep = df_to[["FM", "Plan Lead Factory", "Lat_lead", "Lon_lead"]].copy()
    df_val_keep = df_val[["FM", "Volume Lead Plant (%)"]].copy()

    merged = df_from.merge(df_to_keep, on="FM", how="left").merge(df_val_keep, on="FM", how="left")

    for c in ["Lat_today", "Lon_today", "Lat_lead", "Lon_lead"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged = df_from.merge(df_to[["FM", "Plan Lead Factory", "Lat_to", "Lon_to"]], on="FM", how="left")
    merged = merged.merge(df_sub[["FM", "Plan Sub Factory", "Lat_sub", "Lon_sub"]], on="FM", how="left")

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
# Upload Excel file
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.stop()


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
# Center map
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


































































