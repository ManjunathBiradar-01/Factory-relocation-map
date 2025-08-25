import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import pydeck as pdk

# -------------------- Data loader (single source of truth) --------------------
@st.cache_data(show_spinner=False)
def load_data(xlsx_file) -> pd.DataFrame:
    def read_sheet_any(file, candidates: list[str]) -> pd.DataFrame:
        last_err = None
        for name in candidates:
            try:
                return pd.read_excel(file, sheet_name=name, engine="openpyxl")
            except Exception as e:
                last_err = e
        raise ValueError(
            f"Could not find any of these sheets: {candidates}. Last error: {last_err}"
        )

    def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        norm = {c.lower().strip(): c for c in df.columns}
        for x in candidates:
            if x.lower().strip() in norm:
                return norm[x.lower().strip()]
        return None

    # ---- Read minimum sheets ----
    df_from = read_sheet_any(xlsx_file, ["From"])
    df_to   = read_sheet_any(xlsx_file, ["To"])
    df_sub  = read_sheet_any(xlsx_file, ["Sub-Factory"])

    for d in (df_from, df_to, df_sub):
        d.columns = d.columns.str.strip()

    # ---- Required columns validation ----
    required_from = {"FM", "Name", "Emission", "Engine", "Factory today", "Latitude", "Longitude", "Volume", "Main sales region"}
    missing_from = required_from - set(df_from.columns)
    if missing_from:
        raise ValueError(f"'From' missing columns: {sorted(missing_from)}")

    required_to = {"FM", "Plan Lead Factory", "Latitude", "Longitude", "Volume", "Main sales region"}
    missing_to = required_to - set(df_to.columns)
    if missing_to:
        raise ValueError(f"'To' missing columns: {sorted(missing_to)}")

    required_sub = {"FM", "Latitude", "Longitude", "Volume", "Plan Sub Factory", "Volume", "Main sales region"}
    missing_sub = required_sub - set(df_sub.columns)
    if missing_sub:
        raise ValueError(f"'Sub' missing columns: {sorted(missing_sub)}")

    # ---- Rename coordinates to avoid collisions ----
    df_from = df_from.rename(columns={"Latitude": "Lat_today", "Longitude": "Lon_today"})
    df_to   = df_to.rename(columns={"Latitude": "Lat_lead",  "Longitude": "Lon_lead"})
    df_sub  = df_sub.rename(columns={"Latitude": "Lat_sub",   "Longitude": "Lon_sub"})

    # ---- Detect optional volume columns ----
    lead_pct_col = find_col(df_to, [
        "Volume Lead Plant (%)", "Lead Volume (%)", "Lead Allocation (%)",
        "Lead %", "Lead Percent", "Volume (%)", "Volume"
    ])
    sub_pct_col = find_col(df_sub, [
        "Volume Sub (%)", "Sub Volume (%)", "Sub Allocation (%)", "Sub %",
        "Volume (%)", "Volume"
    ])

    # ---- Keep only necessary columns prior to merge ----
    df_to_keep  = ["FM", "Plan Lead Factory", "Lat_lead", "Lon_lead"]
    if lead_pct_col:
        df_to_keep.append(lead_pct_col)
    df_to_keep = df_to[df_to_keep].copy()

    df_sub_keep = ["FM", "Plan Sub Factory", "Lat_sub", "Lon_sub"]
    if sub_pct_col:
        df_sub_keep.append(sub_pct_col)
    df_sub_keep = df_sub[df_sub_keep].copy()

    # ---- Merge From + To (one-to-one on FM) ----
    merged = df_from.merge(df_to_keep, on="FM", how="left")
    # ---- Attach Sub (one-to-many on FM) ----
    merged = merged.merge(df_sub_keep, on="FM", how="left", suffixes=("", "_sub"))

    # ---- Convert to numeric coords ----
    for c in ["Lat_today", "Lon_today", "Lat_lead", "Lon_lead", "Lat_sub", "Lon_sub"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # ---- Compute % flows ----
    if lead_pct_col and lead_pct_col in merged.columns:
        merged["Lead_Pct"] = pd.to_numeric(merged[lead_pct_col], errors="coerce")
    else:
        merged["Lead_Pct"] = 100.0

    if sub_pct_col and sub_pct_col in merged.columns:
        merged["Sub_Pct"] = pd.to_numeric(merged[sub_pct_col], errors="coerce")
    else:
        # If no sub percentage: split equally among sub factories per FM
        counts = (
            merged.assign(has_sub=merged["Plan Sub Factory"].notna())
                  .groupby("FM", dropna=False)["has_sub"].sum()
                  .rename("n_sub")
        )
        merged = merged.merge(counts, on="FM", how="left")
        def _infer_sub_pct(row):
            if pd.isna(row.get("Plan Sub Factory")):
                return 100.0
            n = row.get("n_sub", 0)
            return 100.0 / n if n and n > 0 else 100.0
        merged["Sub_Pct"] = merged.apply(_infer_sub_pct, axis=1)
        merged.drop(columns=["n_sub"], inplace=True, errors="ignore")

    merged["From_to_Sub_Pct"] = (merged["Lead_Pct"].fillna(0) * merged["Sub_Pct"].fillna(0)) / 100.0

    return merged

# -------------------- Helper: find 'Sales Region' column --------------------
def find_sales_region_col(columns) -> str | None:
    normalized = {c.lower().strip(): c for c in columns}
    candidates = [
        "sales region", "main sales region", "mainsales region",
        "mainsalesregion", "salesregion", "main_sales_region",
    ]
    for key in candidates:
        if key in normalized:
            return normalized[key]
    for c in columns:
        cl = c.lower()
        if "sales" in cl and "region" in cl:
            return c
    return None

# -------------------- Small utility: format coordinates --------------------
def format_coords(lat, lon, decimals: int = 5) -> str:
    if pd.notnull(lat) and pd.notnull(lon):
        return f"{lat:.{decimals}f}, {lon:.{decimals}f}"
    return "n/a"

# -------------------- File Upload --------------------
st.sidebar.subheader("Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()
else:
    default_url = "https://raw.githubusercontent.com/ManjunathBiradar-01/Factory-relocation-map/main/Footprint_SDR.xlsx"
    try:
        df = load_data(default_url)
        st.sidebar.info("Using default dataset from GitHub (upload 'Footprint_SDR 4.xlsx' to override).")
    except Exception as e:
        st.error(f"Failed to load default file from GitHub: {e}")
        st.stop()

# -------------------- Tabs --------------------
tab1, tab2 = st.tabs(["Dashboard", "Edit Dataset"])

with tab1:
    st.title("Factory Production Relocation Dashboard")

    sales_region_col = find_sales_region_col(df.columns)

    # Filters
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        machine_code_filter = st.multiselect(
            "Machine Code (FM)",
            sorted(df["FM"].dropna().astype(str).unique())
        )
    with c2:
        machine_name_filter = st.multiselect(
            "Machine Name",
            sorted(df["Name"].dropna().astype(str).unique())
        )
    with c3:
        engine_filter = st.multiselect(
            "Select Engine Type",
            sorted(df["Engine"].dropna().astype(str).unique())
        )
    with c4:
        emission_filter = st.multiselect(
            "Select Emission Level",
            sorted(df["Emission"].dropna().astype(str).unique())
        )

    c5, c6 = st.columns(2)
    with c5:
        lead_filter = st.multiselect(
            "Lead Factory (To)",
            sorted(df["Plan Lead Factory"].dropna().astype(str).unique())
        )
    with c6:
        sub_factory_filter = st.multiselect(
            "Sub Factory",
            sorted(df["Plan Sub Factory"].dropna().astype(str).unique())
        )

    if sales_region_col:
        sales_region_filter = st.multiselect(
            "Sales Region",
            sorted(df[sales_region_col].dropna().astype(str).unique())
        )
    else:
        sales_region_filter = []
        st.info("Sales Region column not found (optional).")

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
    if lead_filter:
        filtered_df = filtered_df[filtered_df["Plan Lead Factory"].astype(str).isin(lead_filter)]
    if sub_factory_filter:
        filtered_df = filtered_df[filtered_df["Plan Sub Factory"].astype(str).isin(sub_factory_filter)]
    if sales_region_col and sales_region_filter:
        filtered_df = filtered_df[filtered_df[sales_region_col].astype(str).isin(sales_region_filter)]

    # Friendly coordinate strings (optional for table)
    filtered_df["Coords_today"] = filtered_df.apply(lambda r: format_coords(r["Lat_today"], r["Lon_today"]), axis=1)
    filtered_df["Coords_lead"]  = filtered_df.apply(lambda r: format_coords(r["Lat_lead"],  r["Lon_lead"]),  axis=1)
    filtered_df["Coords_sub"]   = filtered_df.apply(lambda r: format_coords(r["Lat_sub"],   r["Lon_sub"]),   axis=1)

    # KPIs
    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        st.metric("Unique FMs", filtered_df["FM"].nunique())
    with kc2:
        st.metric("Main Factories", filtered_df["Factory today"].nunique())
    with kc3:
        st.metric("Lead Factories", filtered_df["Plan Lead Factory"].nunique())
    with kc4:
        st.metric("Sub Factories", filtered_df["Plan Sub Factory"].nunique())

    st.subheader("Volume Flow (From → Lead → Sub)")

    # ---- Map Visualization (Static) ----
    # Markers for each location
import pydeck as pdk
import pandas as pd

# Aggregate volumes for markers
def aggregate_marker_data(df):
    from_vol = df.groupby(["Lat_today", "Lon_today", "Factory today"]).agg({"Volume": "sum"}).reset_index()
    from_vol["type"] = "From"
    from_vol.rename(columns={"Lat_today": "lat", "Lon_today": "lon", "Factory today": "name", "Volume": "from_volume"}, inplace=True)

    lead_vol = df.groupby(["Lat_lead", "Lon_lead", "Plan Lead Factory"]).agg({"Lead_Pct": "sum"}).reset_index()
    lead_vol["type"] = "Lead"
    lead_vol.rename(columns={"Lat_lead": "lat", "Lon_lead": "lon", "Plan Lead Factory": "name", "Lead_Pct": "lead_volume"}, inplace=True)

    

    markers = pd.concat([from_vol, lead_vol], ignore_index=True)
    markers["icon_data"] = [{
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Map_pin_icon.svg",
        "width": 128,
        "height": 128,
        "anchorY": 128
    }] * len(markers)
    return markers

# Arrows for connections
def create_arrow_data(df):
    arrows_main_to_lead = df.dropna(subset=["Lat_today", "Lon_today", "Lat_lead", "Lon_lead"]).copy()
    arrows_main_to_lead["start"] = arrows_main_to_lead[["Lon_today", "Lat_today"]].values.tolist()
    arrows_main_to_lead["end"] = arrows_main_to_lead[["Lon_lead", "Lat_lead"]].values.tolist()
    arrows_main_to_lead["color"] = [[255, 140, 0]] * len(arrows_main_to_lead)
    arrows_main_to_lead["name"] = arrows_main_to_lead["Factory today"] + " → " + arrows_main_to_lead["Plan Lead Factory"]
    arrows_main_to_lead["volume"] = arrows_main_to_lead["Lead_Pct"]
    arrows_main_to_lead["type"] = "Lead Volume Shifted"

    
    return arrows_main_to_lead

# Generate layers
def generate_layers(markers, arrows_main_to_lead):
    marker_layer = pdk.Layer(
        "IconLayer",
        data=markers,
        get_icon="icon_data",
        get_size=4,
        size_scale=15,
        get_position='[lon, lat]',
        pickable=True
    )

    arrow_layer_main_to_lead = pdk.Layer(
        "ArcLayer",
        data=arrows_main_to_lead,
        get_source_position="start",
        get_target_position="end",
        get_source_color="color",
        get_target_color="color",
        get_width=5,
        pickable=True
    )

  
    return [marker_layer, arrow_layer_main_to_lead]

# Tooltip
tooltip = {
    "html": "<b>{name}</b><br/>{type}<br/>Volume: {volume}",
    "style": {
        "backgroundColor": "white",
        "color": "black"
    }
}

# Final rendering
markers = aggregate_marker_data(filtered_df)
arrows_main_to_lead = create_arrow_data(filtered_df)
view_state = pdk.ViewState(latitude=markers["lat"].mean(), longitude=markers["lon"].mean(), zoom=3, pitch=35)

st.pydeck_chart(pdk.Deck(
    layers=generate_layers(markers, arrows_main_to_lead, arrows_lead_to_sub),
    initial_view_state=view_state,
    tooltip=tooltip
))


   # ---- Map Visualization (Static) ----
    # Markers for each location
import pydeck as pdk
import pandas as pd

# Aggregate volumes for markers
def aggregate_marker_data(df):

    lead_vol = df.groupby(["Lat_lead", "Lon_lead", "Plan Lead Factory"]).agg({"Lead_Pct": "sum"}).reset_index()
    lead_vol["type"] = "Lead"
    lead_vol.rename(columns={"Lat_lead": "lat", "Lon_lead": "lon", "Plan Lead Factory": "name", "Lead_Pct": "lead_volume"}, inplace=True)

    sub_vol = df.groupby(["Lat_sub", "Lon_sub", "Plan Sub Factory"]).agg({"Sub_Pct": "sum"}).reset_index()
    sub_vol["type"] = "Sub"
    sub_vol.rename(columns={"Lat_sub": "lat", "Lon_sub": "lon", "Plan Sub Factory": "name", "Sub_Pct": "sub_volume"}, inplace=True)

    markers = pd.concat([from_vol, lead_vol, sub_vol], ignore_index=True)
    markers["icon_data"] = [{
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Map_pin_icon.svg",
        "width": 128,
        "height": 128,
        "anchorY": 128
    }] * len(markers)
    return markers

# Arrows for connections
def create_arrow_data(df):
    arrows_lead_to_sub = df.dropna(subset=["Lat_lead", "Lon_lead", "Lat_sub", "Lon_sub"]).copy()
    arrows_lead_to_sub["start"] = arrows_lead_to_sub[["Lon_lead", "Lat_lead"]].values.tolist()
    arrows_lead_to_sub["end"] = arrows_lead_to_sub[["Lon_sub", "Lat_sub"]].values.tolist()
    arrows_lead_to_sub["color"] = [[0, 0, 255]] * len(arrows_lead_to_sub)
    arrows_lead_to_sub["name"] = arrows_lead_to_sub["Plan Lead Factory"] + " → " + arrows_lead_to_sub["Plan Sub Factory"]
    arrows_lead_to_sub["volume"] = arrows_lead_to_sub["Sub_Pct"]
    arrows_lead_to_sub["type"] = "Sub Volume Shifted"

    return arrows_lead_to_sub

# Generate layers
def generate_layers(markers, arrows_lead_to_sub):
    marker_layer = pdk.Layer(
        "IconLayer",
        data=markers,
        get_icon="icon_data",
        get_size=4,
        size_scale=15,
        get_position='[lon, lat]',
        pickable=True
    )


    arrow_layer_lead_to_sub = pdk.Layer(
        "ArcLayer",
        data=arrows_lead_to_sub,
        get_source_position="start",
        get_target_position="end",
        get_source_color="color",
        get_target_color="color",
        get_width=5,
        pickable=True
    )

    return [marker_layer, arrow_layer_main_to_lead, arrow_layer_lead_to_sub]

# Tooltip
tooltip = {
    "html": "<b>{name}</b><br/>{type}<br/>Volume: {volume}",
    "style": {
        "backgroundColor": "white",
        "color": "black"
    }
}

# Final rendering
markers = aggregate_marker_data(filtered_df)
arrows_lead_to_sub = create_arrow_data(filtered_df)
view_state = pdk.ViewState(latitude=markers["lat"].mean(), longitude=markers["lon"].mean(), zoom=3, pitch=35)

st.pydeck_chart(pdk.Deck(
    layers=generate_layers(markers, arrows_lead_to_sub),
    initial_view_state=view_state,
    tooltip=tooltip
))

    # ---- Detail table: per FM → Sub row with % ----
st.subheader("Detailed Flow Table")
flow_cols = [
    "FM", "Name", "Engine", "Emission",
    "Factory today", "Plan Lead Factory", "Plan Sub Factory",
    "Lead_Pct", "Sub_Pct", "From_to_Sub_Pct",
    "Coords_today", "Coords_lead", "Coords_sub"
]
present_cols = [c for c in flow_cols if c in filtered_df.columns]
st.dataframe(
    filtered_df[present_cols]
    .sort_values(["Factory today", "Plan Lead Factory", "Plan Sub Factory", "FM"], na_position="last"),
    use_container_width=True
)

with tab2:
    st.subheader("Edit Dataset")
    st.write("Use the sidebar to upload a new Excel. Ensure the file contains:")
    st.markdown("""
    - **From** sheet with: `FM`, `Name`, `Emission`, `Engine`, `Factory today`, `Latitude`, `Longitude`
    - **To** sheet with: `FM`, `Plan Lead Factory`, `Latitude`, `Longitude`, *(optional)* `Lead %`
    - **Sub** sheet with: `FM`, `Plan Sub Factory`, `Latitude`, `Longitude`, *(optional)* `Sub %`
    """)







