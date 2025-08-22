import pandas as pd
import numpy as np
import folium
import streamlit as st
import io
from datetime import datetime
import plotly.graph_objects as go


# -------------------- Data loader (single source of truth) --------------------
@st.cache_data(show_spinner=False)
def load_data(xlsx_file) -> pd.DataFrame:
    """
    Loads and merges the required sheets:
      - From (original location data)
      - To (lead factory coordinates & optional %)
      - Sub (sub factory coordinates & optional %)
    Performs validations and returns a merged long DataFrame with per-sub rows.
    Accepts both file-like objects (Streamlit UploadedFile) and file paths/URLs.
    """

    # ---- Helper: read a sheet by any of several possible names ----
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

    # ---- Helper: column finder (first match from a set of candidates) ----
    def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        norm = {c.lower().strip(): c for c in df.columns}
        for x in candidates:
            if x.lower().strip() in norm:
                return norm[x.lower().strip()]
        return None

    # ---- Read minimum sheets (NO 'Values' sheet used) ----
    df_from = read_sheet_any(xlsx_file, ["From"])
    df_to   = read_sheet_any(xlsx_file, ["To"])
    df_sub  = read_sheet_any(xlsx_file, ["Sub-Factory"])

    # Normalize column names
    for d in (df_from, df_to, df_sub):
        d.columns = d.columns.str.strip()

    # ---- Required columns validation ----
    # From
    required_from = {"FM", "Name", "Emission", "Engine", "Factory today", "Latitude", "Longitude", "SFC RTM", "Main sales region"}
    missing_from = required_from - set(df_from.columns)
    if missing_from:
        raise ValueError(f"'From' missing columns: {sorted(missing_from)}")

    # To
    required_to = {"FM", "Plan Lead Factory", "Latitude", "Longitude", "SFC RTM", "Main sales region"}
    missing_to = required_to - set(df_to.columns)
    if missing_to:
        raise ValueError(f"'To' missing columns: {sorted(missing_to)}")

    # sub
    required_sub = {"FM", "Latitude", "Longitude", "SFC RTM", "Plan Sub Factory", "Volume", "Main sales region"}
    missing_sub = required_to - set(df_to.columns)
    if missing_to:
        raise ValueError(f"'To' missing columns: {sorted(missing_sub)}")
        raise ValueError(
            "Could not find a Sub Factory name column in 'Sub' sheet. "
            "Expected one of: 'Sub Factory', 'Sub-Factory', 'SubFactory', 'Sub Plant', 'Allocated Sub Factory'."
        )

    # ---- Rename coordinates to avoid collisions ----
    df_from = df_from.rename(columns={"Latitude": "Lat_today", "Longitude": "Lon_today"})
    df_to   = df_to.rename(columns={"Latitude": "Lat_lead",  "Longitude": "Lon_lead"})
    df_sub  = df_sub.rename(columns={"Latitude": "Lat_sub",  "Longitude": "Lon_sub"})

    # ---- Detect optional volume columns ----
    # Lead % candidates (in 'To')
    lead_pct_col = find_col(df_to, [
        "Volume Lead Plant (%)", "Lead Volume (%)", "Lead Allocation (%)",
        "Lead %", "Lead Percent", "Volume (%)", "Volume"
    ])
    # Sub % candidates (in 'Sub')
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
    # If Sub sheet is given per FM only, allow duplicates; else it's fine.
    df_sub_keep = df_sub[df_sub_keep].copy()

    # ---- Merge From + To (one-to-one on FM) ----
    merged = df_from.merge(df_to_keep, on="FM", how="left")

    # ---- Attach Sub (one-to-many on FM) ----
    merged = merged.merge(df_sub_keep, on="FM", how="left", suffixes=("", "_sub"))

    # ---- Convert to numeric coords ----
    for c in ["Lat_today", "Lon_today", "Lat_lead", "Lon_lead", "Lat_sub", "Lon_sub"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # ---- Compute % flows (no Values sheet) ----
    # Lead_Pct
    if lead_pct_col and lead_pct_col in merged.columns:
        merged["Lead_Pct"] = pd.to_numeric(merged[lead_pct_col], errors="coerce")
    else:
        # Default: 100% from -> lead if not specified
        merged["Lead_Pct"] = 100.0

    # Sub_Pct
    if sub_pct_col and sub_pct_col in merged.columns:
        merged["Sub_Pct"] = pd.to_numeric(merged[sub_pct_col], errors="coerce")
    else:
        # If no sub percentage is provided:
        # - If an FM has N sub factories, split equally (100/N) for that FM.
        # - If an FM has exactly 1 sub or none, set 100% to that single sub (or NaN sub means no split).
        # Compute counts per FM where Sub Factory is present
        counts = (
            merged.assign(has_sub=merged["Sub Factory"].notna())
                  .groupby("FM", dropna=False)["has_sub"].sum()
                  .rename("n_sub")
        )
        merged = merged.merge(counts, on="FM", how="left")
        def _infer_sub_pct(row):
            if pd.isna(row.get("Sub Factory")):
                # no sub designated: treat as 100% at lead with no further split
                return 100.0
            n = row.get("n_sub", 0)
            return 100.0 / n if n and n > 0 else 100.0
        merged["Sub_Pct"] = merged.apply(_infer_sub_pct, axis=1)
        merged.drop(columns=["n_sub"], inplace=True, errors="ignore")

    # Overall From->Sub %
    merged["From_to_Sub_Pct"] = (merged["Lead_Pct"].fillna(0) * merged["Sub_Pct"].fillna(0)) / 100.0

    return merged


# -------------------- Helper: find 'Sales Region' column (unchanged) --------------------
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


# -------------------- Small utility: format coordinates (unchanged) --------------------
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
    # Fallback (keep your GitHub default or upload your 'Footprint_SDR 4.xlsx')
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

    # NEW: Lead & Sub factory filters
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
        st.metric("From Factories", filtered_df["Factory today"].nunique())
    with kc3:
        st.metric("Lead Factories", filtered_df["Plan Lead Factory"].nunique())
    with kc4:
        st.metric("Sub Factories", filtered_df["Plan Sub Factory"].nunique())

    st.subheader("Volume Flow (From → Lead → Sub)")


import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(layout="wide")
st.title("Factory Relocation Flow Map")

# ---- Assume filtered_df is already defined ----
# Replace this with your actual data loading logic
# For example:
# filtered_df = load_data(...)

# ---- Tabs ----
tab1, tab2 = st.tabs(["📍 Map View", "📝 Edit Dataset"])

with tab1:
    st.subheader("Factory Flow Map")

# Combine flow lines and pinpoint markers
from_to_lead = filtered_df.dropna(subset=["Lat_today", "Lon_today", "Lat_lead", "Lon_lead"])
lead_to_sub = filtered_df.dropna(subset=["Lat_lead", "Lon_lead", "Lat_sub", "Lon_sub"])

# Line data


    "IconLayer",
    data=arrow_df,
    get_icon="icon_data",
    get_size=4,
    size_scale=15,
    get_position="[lon, lat]",
    get_color=[255, 0, 0],
    pickable=True,
)


lead_to_sub_lines = [
    {
        "from_lat": row["Lat_lead"],
        "from_lon": row["Lon_lead"],
        "to_lat": row["Lat_sub"],
        "to_lon": row["Lon_sub"],
        "label": f"{row['Plan Lead Factory']} → {row['Plan Sub Factory']}"
    }
    for _, row in lead_to_sub.iterrows()
]

lines_df = pd.DataFrame(from_to_lead_lines + lead_to_sub_lines)

# Marker data

pdk.Layer(
    "IconLayer",
    data=markers,
    get_icon="icon_data",
    get_size=4,
    size_scale=15,
    get_position="[lon, lat]",
    get_color=[0, 128, 255],
    pickable=True,
)


# Layers
line_layer = pdk.Layer(
    "LineLayer",
    data=lines_df,
    get_source_position="[from_lon, from_lat]",
    get_target_position="[to_lon, to_lat]",
    get_width=3,
    get_color=[255, 0, 0],
    pickable=True,
    auto_highlight=True
)

marker_layer = pdk.Layer(
    "ScatterplotLayer",
    data=markers,
    get_position="[lon, lat]",
    get_color=[0, 128, 255],
    get_radius=50000,
    pickable=True
)

# View state
view_state = pdk.ViewState(
    latitude=markers["lat"].mean(),
    longitude=markers["lon"].mean(),
    zoom=2,
    pitch=0
)

# Render map
st.pydeck_chart(pdk.Deck(
    layers=[line_layer, marker_layer],
    initial_view_state=view_state,
    tooltip={"text": "{label}"}
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


















































































