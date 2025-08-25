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
    df_from = df_from.rename(columns={"Latitude": "Lat_today", "Longitude": "Lon_today", "Volume" : "Main Volume"})
    df_to   = df_to.rename(columns={"Latitude": "Lat_lead",  "Longitude": "Lon_lead",  "Volume" : "Lead Volume"})
    df_sub  = df_sub.rename(columns={"Latitude": "Lat_sub",   "Longitude": "Lon_sub",  "Volume" : "Sub Volume"})

    # ---- Detect optional volume columns ----
    lead_pct_col = find_col(df_to, [
        "Volume"])
    sub_pct_col = find_col(df_sub, [
       "Volume"])

    # ---- Keep only necessary columns prior to merge ----
    df_to_keep  = ["FM", "Plan Lead Factory", "Lat_lead", "Lon_lead", "Lead Volume"]
    if lead_pct_col:
        df_to_keep.append(lead_pct_col)
    df_to_keep = df_to[df_to_keep].copy()

    df_sub_keep = ["FM", "Plan Sub Factory", "Lat_sub", "Lon_sub",  "Sub Volume"]
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
# -------------------- Data Upload & Persistence --------------------
st.sidebar.subheader("Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

# --- Keep uploaded file in session_state ---
if uploaded_file is not None:
    st.session_state["uploaded_file"] = uploaded_file   # Save file to session state

# --- Allow clearing uploaded file ---
if "uploaded_file" in st.session_state:
    if st.sidebar.button("Clear uploaded file"):
        del st.session_state["uploaded_file"]
        st.rerun()

# --- Load data (uploaded → fallback to GitHub) ---
if "uploaded_file" in st.session_state:
    try:
        df = load_data(st.session_state["uploaded_file"])
        st.sidebar.success(f"Using uploaded file: {


# ---------------- Dashboard starts here (outside if/else) ----------------
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

c5, c6 = st.columns(2)
with c5:
    lead_filter = st.multiselect("Lead Factory (To)", sorted(df["Plan Lead Factory"].dropna().astype(str).unique()))
with c6:
    sub_factory_filter = st.multiselect("Sub Factory", sorted(df["Plan Sub Factory"].dropna().astype(str).unique()))

if sales_region_col:
    sales_region_filter = st.multiselect("Sales Region", sorted(df[sales_region_col].dropna().astype(str).unique()))
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
with kc1: st.metric("Unique FMs", filtered_df["FM"].nunique())
with kc2: st.metric("Main Factories", filtered_df["Factory today"].nunique())
with kc3: st.metric("Lead Factories", filtered_df["Plan Lead Factory"].nunique())
with kc4: st.metric("Sub Factories", filtered_df["Plan Sub Factory"].nunique())

st.subheader("Volume Flow (From → Lead → Sub)")


import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(layout="wide")
st.title("Animated Arrows: Factory Relocation Maps")

# ---- Tooltip ----
tooltip = {
    "html": "<b>{name}</b><br/>{type}<br/>Volume: {volume}",
    "style": {
        "backgroundColor": "white",
        "color": "black"
    }
}

# ---- Map 1: Main Factory → Lead Factory ----
def aggregate_main_to_lead_markers(df):
    from_vol = df.groupby(["Lat_today", "Lon_today", "Factory today"]).agg({"Main Volume": "sum"}).reset_index()
    from_vol["type"] = "From"
    from_vol.rename(columns={"Lat_today": "lat", "Lon_today": "lon", "Factory today": "name"}, inplace=True)   
    from_vol["volume"] = from_vol["Main Volume"]

    

    lead_vol = df.groupby(["Lat_lead", "Lon_lead", "Plan Lead Factory"]).agg({"Lead Volume": "sum"}).reset_index()
    lead_vol["type"] = "Lead"
    lead_vol.rename(columns={"Lat_lead": "lat", "Lon_lead": "lon", "Plan Lead Factory": "name"}, inplace=True)
    lead_vol["volume"] = lead_vol["Lead Volume"]


    markers = pd.concat([from_vol, lead_vol], ignore_index=True)
    markers["icon_data"] = [{
        "url": "https://www.freeiconspng.com/uploads/map-location-icon-29.png",
        "width": 128,
        "height": 128,
        "anchorY": 128
    }] * len(markers)
    return markers


def create_main_to_lead_trips(df):
    df = df.dropna(subset=["Lat_today", "Lon_today", "Lat_lead", "Lon_lead", "Lead Volume"]).copy()
    df = df[df["Lead Volume"] > 0]  # Only include rows with volume > 0

    # Group by route to get total volume
    grouped = df.groupby([
        "Factory today", "Plan Lead Factory", "Lat_today", "Lon_today", "Lat_lead", "Lon_lead"
    ], as_index=False).agg({"Lead Volume": "sum"})

    grouped["path"] = grouped.apply(lambda row: [
        [row["Lon_today"], row["Lat_today"]],
        [row["Lon_lead"], row["Lat_lead"]]
    ], axis=1)
    grouped["timestamps"] = [[0, 100]] * len(grouped)
    grouped["color"] = [[255, 140, 0]] * len(grouped)
    grouped["name"] = grouped["Factory today"] + " → " + grouped["Plan Lead Factory"]
    grouped["volume"] = grouped["Lead Volume"]
    grouped["type"] = "Lead Volume Shifted"
    return grouped
def aggregate_lead_to_sub_markers(df):
    lead_vol = df.groupby(["Lat_lead", "Lon_lead", "Plan Lead Factory"]).agg({"Lead Volume": "sum"}).reset_index()
    lead_vol["type"] = "Lead"
    lead_vol.rename(columns={"Lat_lead": "lat", "Lon_lead": "lon", "Plan Lead Factory": "name"}, inplace=True)
    lead_vol["volume"] = lead_vol["Lead Volume"]


    sub_vol = df.groupby(["Lat_sub", "Lon_sub", "Plan Sub Factory"]).agg({"Sub Volume": "sum"}).reset_index()
    sub_vol["type"] = "Sub"
    sub_vol.rename(columns={"Lat_sub": "lat", "Lon_sub": "lon", "Plan Sub Factory": "name"}, inplace=True)
    sub_vol["volume"] = sub_vol["Sub Volume"]


    markers = pd.concat([lead_vol, sub_vol], ignore_index=True)
    markers["icon_data"] = [{
        "url": "https://www.freeiconspng.com/uploads/map-location-icon-29.png",
        "width": 128,
        "height": 128,
        "anchorY": 128
    }] * len(markers)
    return markers

def create_lead_to_sub_trips(df):
    df = df.dropna(subset=["Lat_lead", "Lon_lead", "Lat_sub", "Lon_sub", "Sub Volume"]).copy()
    df = df[df["Sub Volume"] > 0]  # Only include rows where Sub Volume > 0
    df["path"] = df.apply(lambda row: [
        [row["Lon_lead"], row["Lat_lead"]],
        [row["Lon_sub"], row["Lat_sub"]]
    ], axis=1)
    df["timestamps"] = [[0, 100]] * len(df)
    df["color"] = [[0, 0, 255]] * len(df)
    df["name"] = df["Plan Lead Factory"] + " → " + df["Plan Sub Factory"]
    df["volume"] = df["Sub Volume"]
    df["type"] = "Sub Volume Shifted"
    return df



filtered_df = filtered_df.dropna(subset=["Lat_today", "Lon_today", "Lat_lead", "Lon_lead", "Lat_sub", "Lon_sub"])


# ---- Render Map 1 ----
# Example DataFrame (replace with your actual filtered_df)
# filtered_df = load_data(...) or use your existing filtered_df

markers1 = aggregate_main_to_lead_markers(filtered_df)
trips1 = create_main_to_lead_trips(filtered_df)

view_state1 = pdk.ViewState(
    latitude=markers1["lat"].mean(),
    longitude=markers1["lon"].mean(),
    zoom=4,
    pitch=35
)

layer1_markers = pdk.Layer(
    "IconLayer",
    data=markers1,
    get_icon="icon_data",
    get_size=30,
    size_scale=30,
    get_position='[lon, lat]',
    pickable=True
)

layer1_trips = pdk.Layer(
    "TripsLayer",
    data=trips1,
    get_path="path",
    get_timestamps="timestamps",
    get_color="color",
    opacity=0.8,
    width_min_pixels=5,
    rounded=True,
    trail_length=180,
    current_time=100,
    pickable=True
)

st.pydeck_chart(pdk.Deck(
    layers=[layer1_markers, layer1_trips],
    initial_view_state=view_state1,
    tooltip=tooltip
))


# ---- Render Map 2 ----
st.subheader("Lead Factory → Sub Factory")
markers2 = aggregate_lead_to_sub_markers(filtered_df)
trips2 = create_lead_to_sub_trips(filtered_df)
view_state2 = pdk.ViewState(latitude=markers2["lat"].mean(), longitude=markers2["lon"].mean(), zoom=3, pitch=35)

layer2_markers = pdk.Layer("IconLayer", data=markers2, get_icon="icon_data", get_size=30, size_scale=30, get_position='[lon, lat]', pickable=True)
layer2_trips = pdk.Layer(
    "TripsLayer",
    data=trips2,
    get_path="path",
    get_timestamps="timestamps",
    get_color="color",
    opacity=0.8,
    width_min_pixels=5,
    rounded=True,
    trail_length=180,
    current_time=100,
    pickable=True
)

st.pydeck_chart(pdk.Deck(
    layers=[layer2_markers, layer2_trips],
    initial_view_state=view_state2,
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


















































