import pandas as pd
import numpy as np
import folium
import streamlit as st

# ---------- Page Settings ----------
st.set_page_config(page_title="Factory Production Relocation Dashboard", layout="wide")

# ---------- Data Loader ----------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    df_from = pd.read_excel(uploaded_file, sheet_name="From", engine="openpyxl")
    df_to   = pd.read_excel(uploaded_file, sheet_name="To", engine="openpyxl")
    df_val  = pd.read_excel(uploaded_file, sheet_name="Values", engine="openpyxl")

    for d in (df_from, df_to, df_val):
        d.columns = d.columns.str.strip()

    required_from = {"FM", "Name", "Emission", "Engine", "Factory today", "Latitude", "Longitude"}
    required_to   = {"FM", "Plan Lead Factory", "Latitude", "Longitude"}
    required_val  = {"FM", "Volume Lead Plant (%)"}

    missing = [
        ("From", required_from - set(df_from.columns)),
        ("To", required_to - set(df_to.columns)),
        ("Values", required_val - set(df_val.columns)),
    ]
    missing = [(s, cols) for s, cols in missing if cols]
    if missing:
        msg = "; ".join([f"{s} missing: {sorted(cols)}" for s, cols in missing])
        raise ValueError(f"Expected columns not found -> {msg}")

    df_from = df_from.rename(columns={"Latitude": "Lat_today", "Longitude": "Lon_today"})
    df_to   = df_to.rename(columns={"Latitude": "Lat_lead",  "Longitude": "Lon_lead"})

    df_to_keep  = df_to[["FM", "Plan Lead Factory", "Lat_lead", "Lon_lead"]].copy()
    df_val_keep = df_val[["FM", "Volume Lead Plant (%)"]].copy()

    merged = (
        df_from
        .merge(df_to_keep,  on="FM", how="left")
        .merge(df_val_keep, on="FM", how="left")
    )

    for c in ["Lat_today", "Lon_today", "Lat_lead", "Lon_lead"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    return merged

# ---------- File Upload ----------
st.title("Factory Production Relocation Dashboard")
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if not uploaded_file:
    st.warning("Please upload the Excel file to proceed.")
    st.stop()

df = load_data(uploaded_file)

# ---------- Filters ----------
col1, col2, col3 = st.columns(3)
with col1:
    factory_filter = st.multiselect("Select Factory Today", sorted(df["Factory today"].dropna().unique()))
with col2:
    engine_filter = st.multiselect("Select Engine Type", sorted(df["Engine"].dropna().unique()))
with col3:
    emission_filter = st.multiselect("Select Emission Level", sorted(df["Emission"].dropna().unique()))

filtered_df = df.copy()
if factory_filter:
    filtered_df = filtered_df[filtered_df["Factory today"].isin(factory_filter)]
if engine_filter:
    filtered_df = filtered_df[filtered_df["Engine"].isin(engine_filter)]
if emission_filter:
    filtered_df = filtered_df[filtered_df["Emission"].isin(emission_filter)]

# ---------- Map Centering ----------
coords = []
if not filtered_df.empty:
    coords.extend(filtered_df[["Lat_today", "Lon_today"]].dropna().values.tolist())
    coords.extend(filtered_df[["Lat_lead", "Lon_lead"]].dropna().values.tolist())

center_lat, center_lon = (np.mean([c[0] for c in coords]), np.mean([c[1] for c in coords])) if coords else (20.0, 0.0)
m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles="OpenStreetMap")

# ---------- Plot Markers & Flows ----------
for _, row in filtered_df.iterrows():
    lat_today, lon_today = row["Lat_today"], row["Lon_today"]
    lat_lead,  lon_lead  = row["Lat_lead"],  row["Lon_lead"]

    if pd.notnull(lat_today) and pd.notnull(lon_today):
        folium.Marker(
            [lat_today, lon_today],
            popup=folium.Popup(
                f"<b>Factory Today:</b> {row.get('Factory today','')}<br>"
                f"<b>Name:</b> {row.get('Name','')}<br>"
                f"<b>Engine:</b> {row.get('Engine','')}<br>"
                f"<b>Emission:</b> {row.get('Emission','')}",
                max_width=300
            ),
            icon=folium.Icon(color="red", icon="industry", prefix="fa"),
            tooltip="Factory Today"
        ).add_to(m)

    if pd.notnull(lat_lead) and pd.notnull(lon_lead):
        folium.Marker(
            [lat_lead, lon_lead],
            popup=folium.Popup(
                f"<b>Lead Factory:</b> {row.get('Plan Lead Factory','')}<br>"
                f"<b>Name:</b> {row.get('Name','')}<br>"
                f"<b>Engine:</b> {row.get('Engine','')}<br>"
                f"<b>Emission:</b> {row.get('Emission','')}",
                max_width=300
            ),
            icon=folium.Icon(color="green", icon="flag", prefix="fa"),
            tooltip="Plan Lead Factory"
        ).add_to(m)

    if pd.notnull(lat_today) and pd.notnull(lon_today) and pd.notnull(lat_lead) and pd.notnull(lon_lead):
        vol = row.get("Volume Lead Plant (%)")
        vol_txt = f"{vol:.0f}%" if pd.notnull(vol) else "n/a"
        folium.PolyLine(
            [[lat_today, lon_today], [lat_lead, lon_lead]],
            color="blue", weight=3, opacity=0.7,
            tooltip=f"Volume Lead Plant: {vol_txt}"
        ).add_to(m)

# ---------- Render Map ----------
st.subheader("Production Relocation Map")
st.components.v1.html(m._repr_html_(), height=600)

# ---------- Show Data ----------
with st.expander("Show filtered data"):
    st.dataframe(
        filtered_df[
            ["FM","Name","Emission","Engine","Factory today",
             "Plan Lead Factory","Volume Lead Plant (%)",
             "Lat_today","Lon_today","Lat_lead","Lon_lead"]
        ].reset_index(drop=True)
    )
