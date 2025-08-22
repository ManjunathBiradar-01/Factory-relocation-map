import streamlit as st
import pandas as pd
import folium

# ---------- Data loader ----------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """
    Loads and merges the required sheets:
      - From  (original location data)
      - To    (lead factory coordinates)
      - Values (volume %)
    Performs basic validations and returns a single merged DataFrame.
    """
    df_from = pd.read_excel(path, sheet_name="From", engine="openpyxl")
    df_to = pd.read_excel(path, sheet_name="To", engine="openpyxl")
    df_val = pd.read_excel(path, sheet_name="Values", engine="openpyxl")

    for d in (df_from, df_to, df_val):
        d.columns = d.columns.str.strip()

    merged = df_val.merge(df_from, on="FM", how="left").merge(df_to, on="FM", how="left")
    return merged

# ---------- Helper: Find a 'Sales Region' column ----------
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

# ---------- Load data ----------
excel_path = "Footprint_SDR.xlsx"
try:
    df = load_data(excel_path)
except Exception as e:
    st.error(f"Failed to load data from '{excel_path}'.\n\n{e}")
    st.stop()

# ---------- UI ----------
st.title("Factory Production Relocation Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    factory_filter = st.multiselect(
        "Select Factory Today",
        options=sorted(df["Factory today"].dropna().unique().tolist())
    )

sales_region_col = find_sales_region_col(df.columns)

c1, c2, c3, c4 = st.columns(4)
with c1:
    machine_code_filter = st.multiselect(
        "Machine Code (FM)",
        options=sorted(df["FM"].dropna().astype(str).unique().tolist())
    )
with c2:
    machine_name_filter = st.multiselect(
        "Machine Name",
        options=sorted(df["Name"].dropna().astype(str).unique().tolist())
    )
with c3:
    engine_filter = st.multiselect(
        "Select Engine Type",
        options=sorted(df["Engine"].dropna().astype(str).unique().tolist())
    )
with c4:
    emission_filter = st.multiselect(
        "Select Emission Level",
        options=sorted(df["Emission"].dropna().astype(str).unique().tolist())
    )

if sales_region_col:
    (c5,) = st.columns(1)
    with c5:
        sales_region_filter = st.multiselect(
            "Sales Region",
            options=sorted(df[sales_region_col].dropna().astype(str).unique().tolist())
        )
else:
    sales_region_filter = []
    st.info(
        "Sales Region column not found. Looking for variations like "
        "'Sales Region', 'Main Sales Region', 'MainSales Region', or 'SalesRegion'."
    )

# ---------- Apply filters ----------
filtered_df = df.copy()
if factory_filter:
    filtered_df = filtered_df[filtered_df["Factory today"].isin(factory_filter)]
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

# ---------- Map ----------
m = folium.Map(location=[20, 78], zoom_start=4)
for _, row in filtered_df.iterrows():
    lat_today, lon_today = row["Lat_today"], row["Lon_today"]
    lat_lead, lon_lead = row["Lat_lead"], row["Lon_lead"]

    sales_region_line = ""
    if sales_region_col and pd.notnull(row.get(sales_region_col, None)):
        sales_region_line = f"<br><b>Sales Region:</b> {row.get(sales_region_col, '')}"

    if pd.notnull(lat_today) and pd.notnull(lon_today):
        folium.Marker(
            [lat_today, lon_today],
            popup=folium.Popup(
                f"<b>Factory Today:</b> {row.get('Factory today','')}<br>"
                f"<b>Name:</b> {row.get('Name','')}<br>"
                f"<b>Machine Code (FM):</b> {row.get('FM','')}<br>"
                f"<b>Engine:</b> {row.get('Engine','')}<br>"
                f"<b>Emission:</b> {row.get('Emission','')}"
                f"{sales_region_line}",
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
                f"<b>Machine Code (FM):</b> {row.get('FM','')}<br>"
                f"<b>Engine:</b> {row.get('Engine','')}<br>"
                f"<b>Emission:</b> {row.get('Emission','')}"
                f"{sales_region_line}",
                max_width=300
            ),
            icon=folium.Icon(color="blue", icon="flag", prefix="fa"),
            tooltip="Plan Lead Factory"
        ).add_to(m)

    if pd.notnull(lat_today) and pd.notnull(lon_today) and pd.notnull(lat_lead) and pd.notnull(lon_lead):
        vol = row.get("Volume Lead Plant (%)")
        tooltip = f"Volume: {vol}%" if pd.notnull(vol) else "Volume: N/A"
        folium.PolyLine(
            locations=[[lat_today, lon_today], [lat_lead, lon_lead]],
            color="gray", weight=2, tooltip=tooltip
        ).add_to(m)

st.components.v1.html(m._repr_html_(), height=600)

# ---------- Show filtered data ----------
with st.expander("Show filtered data"):
    cols_to_show = [
        "FM", "Name", "Emission", "Engine", "Factory today",
        "Plan Lead Factory", "Volume Lead Plant (%)",
        "Lat_today", "Lon_today", "Lat_lead", "Lon_lead"
    ]
    if sales_region_col and sales_region_col not in cols_to_show:
        cols_to_show.insert(2, sales_region_col)

    cols_to_show = [c for c in cols_to_show if c in filtered_df.columns]
    st.dataframe(filtered_df[cols_to_show].reset_index(drop=True))






























































