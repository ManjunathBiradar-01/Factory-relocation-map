import pandas as pd
import numpy as np
import folium
import streamlit as st

# ---------- Settings ----------
st.set_page_config(
    page_title="Bomag SDMs Factory Production Relocation Dashboard",
    layout="wide"
)

# ---------- Data loader (define BEFORE calling it) ----------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """
    Loads and merges the required sheets:
      - From  (original location data)
      - To    (lead factory coordinates)
      - Values (volume %)
    Performs basic validations and returns a single merged DataFrame.
    """
    # Read only the required sheets
    df_from = pd.read_excel(path, sheet_name="From", engine="openpyxl")
    df_to   = pd.read_excel(path, sheet_name="To", engine="openpyxl")
    df_val  = pd.read_excel(path, sheet_name="Values", engine="openpyxl")

    # Normalize column names (trim spaces)
    for d in (df_from, df_to, df_val):
        d.columns = d.columns.str.strip()

    # Required columns check (helps debug early)
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

    # Rename coordinates to avoid collisions
    df_from = df_from.rename(columns={"Latitude": "Lat_today", "Longitude": "Lon_today"})
    df_to   = df_to.rename(columns={"Latitude": "Lat_lead",  "Longitude": "Lon_lead"})

    # Keep only necessary columns prior to merge
    df_to_keep  = df_to[["FM", "Plan Lead Factory", "Lat_lead", "Lon_lead"]].copy()
    df_val_keep = df_val[["FM", "Volume Lead Plant (%)"]].copy()

    # Merge on FM
    merged = (
        df_from
        .merge(df_to_keep,  on="FM", how="left")
        .merge(df_val_keep, on="FM", how="left")
    )

    # Convert to numeric coords
    for c in ["Lat_today", "Lon_today", "Lat_lead", "Lon_lead"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    return merged


# ---------- Helper: Find a 'Sales Region' column ----------
def find_sales_region_col(columns):
    """
    Tries to find a sales region column in the merged dataframe.
    Accepts common variants like:
      'Sales Region', 'Main Sales Region', 'MainSales Region',
      'MainSalesRegion', 'SalesRegion', 'main_sales_region'
    Returns the exact column name if found, otherwise None.
    """
    normalized = {c.lower().strip(): c for c in columns}
    candidates = [
        "sales region", "main sales region", "mainsales region",
        "mainsalesregion", "salesregion", "main_sales_region"
    ]
    for key in candidates:
        if key in normalized:
            return normalized[key]

    # Fallback: any column containing both 'sales' and 'region'
    for c in columns:
        cl = c.lower()
        if "sales" in cl and "region" in cl:
            return c

    return None


# ---------- Small utility: format coordinates ----------
def format_coords(lat, lon, decimals: int = 5) -> str:
    """Return a friendly 'lat, lon' string or 'n/a' if missing."""
    if pd.notnull(lat) and pd.notnull(lon):
        return f"{lat:.{decimals}f}, {lon:.{decimals}f}"
    return "n/a"


# ---------- Path & load ----------
excel_path = "Footprint_SDR.xlsx"
try:
    df = load_data(excel_path)
except Exception as e:
    st.error(f"Failed to load data from '{excel_path}'.\n\n{e}")
    st.stop()

# ---------- UI (updated) ----------
st.title("Factory Production Relocation Dashboard")

# Detect sales region column dynamically
sales_region_col = find_sales_region_col(df.columns)

# Row 1: Machine Code, Machine Name, Engine, Emission
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

# Row 2: Sales Region (if column found)
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

# ---------- Apply filters (updated) ----------
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

# ---------- Map centering ----------
coords = []
if not filtered_df.empty:
    coords.extend(filtered_df[["Lat_today", "Lon_today"]].dropna().values.tolist())
    coords.extend(filtered_df[["Lat_lead", "Lon_lead"]].dropna().values.tolist())

if coords:
    center_lat = float(np.mean([c[0] for c in coords]))
    center_lon = float(np.mean([c[1] for c in coords]))
else:
    center_lat, center_lon = 20.0, 0.0  # global fallback

m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles="OpenStreetMap")

# ---------- Plot markers & flows ----------
for _, row in filtered_df.iterrows():
    lat_today, lon_today = row["Lat_today"], row["Lon_today"]
    lat_lead,  lon_lead  = row["Lat_lead"],  row["Lon_lead"]

    # Optional Sales Region line for popup
    sales_region_line = ""
    if sales_region_col and pd.notnull(row.get(sales_region_col, None)):
        sales_region_line = f"<br><b>Sales Region:</b> {row.get(sales_region_col, '')}"

    # Factory Today marker with LOCATION + "Open in Maps" link
    if pd.notnull(lat_today) and pd.notnull(lon_today):
        loc_str_today = format_coords(lat_today, lon_today)
        maps_link_today = f"https://www.google.com/maps?q={lat_today},{lon_today}"
        folium.Marker(
            [lat_today, lon_today],
            popup=folium.Popup(
                f"<b>Factory Today:</b> {row.get('Factory today','')}<br>"
                f"{sales_region_line}",
                max_width=320
            ),
            icon=folium.Icon(color="red", icon="industry", prefix="fa"),
            tooltip="Factory Today"
        ).add_to(m)

    # Lead Factory marker with location (bonus)
    if pd.notnull(lat_lead) and pd.notnull(lon_lead):
        loc_str_lead = format_coords(lat_lead, lon_lead)
        maps_link_lead = f"https://www.google.com/maps?q={lat_lead},{lon_lead}"
        folium.Marker(
            [lat_lead, lon_lead],
            popup=folium.Popup(
                f"<b>Factory Today:</b> {row.get('Factory Today','')}<br>"
                f"{sales_region_line}",
                max_width=320
            ),
            icon=folium.Icon(color="blue", icon="flag", prefix="fa"),
            tooltip="Factory Today"
        ).add_to(m)

# ---------- Flow line with volume + from/to in tooltip ----------
if (pd.notnull(lat_today) and pd.notnull(lon_today) and
    pd.notnull(lat_lead) and pd.notnull(lon_lead)):

      # Flow line with volume tooltip
    if (pd.notnull(lat_today) and pd.notnull(lon_today) and
        pd.notnull(lat_lead) and pd.notnull(lon_lead)):
        vol = row.get("Volume Lead Plant (%)")
        vol_txt = f"{vol:.0f}%" if pd.notnull(vol) else "n/a"
        folium.PolyLine(
            [[lat_today, lon_today], [lat_lead, lon_lead]],
            color="green", weight=3, opacity=0.7,
            tooltip=f"Volume Lead Plant: {vol_txt}"
        ).add_to(m)

# ---------- Render ----------
st.subheader("Production Relocation Map")
st.components.v1.html(m._repr_html_(), height=600)

# Add location columns to the table view
filtered_df = filtered_df.copy()
filtered_df["Factory Today Location"] = filtered_df.apply(
    lambda r: format_coords(r["Lat_today"], r["Lon_today"]), axis=1
)
filtered_df["Lead Factory Location"] = filtered_df.apply(
    lambda r: format_coords(r["Lat_lead"], r["Lon_lead"]), axis=1
)

with st.expander("Show filtered data"):
    cols_to_show = [
        "FM","Name",
        # Insert Sales Region after Name if present
    ]
    if sales_region_col:
        cols_to_show.append(sales_region_col)
    cols_to_show += [
        "Emission","Engine","Factory today",
        "Factory Today Location",  # <--- added
        "Plan Lead Factory",
        "Lead Factory Location",   # <--- added
        "Volume Lead Plant (%)",
        "Lat_today","Lon_today","Lat_lead","Lon_lead"
    ]

    # Only keep columns that actually exist (robust)
    cols_to_show = [c for c in cols_to_show if c in filtered_df.columns]

    st.dataframe(filtered_df[cols_to_show].reset_index(drop=True)) 

















