import pandas as pd
import numpy as np
import folium
import streamlit as st
import requests
from io import BytesIO


# ---------- Settings ----------
st.set_page_config(
    page_title="Bomag SDMs Factory Production Relocation Dashboard",
    layout="wide"
)

# ---------- Data loader (define BEFORE calling it) ----------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    # Load sheets
    df_from = pd.read_excel(path, sheet_name="From", engine="openpyxl")
    df_to = pd.read_excel(path, sheet_name="To", engine="openpyxl")
    df_sub = pd.read_excel(path, sheet_name="Sub-Factory", engine="openpyxl")

    # Normalize column names
    for df in [df_from, df_to, df_sub]:
        df.columns = df.columns.str.strip()

    # Extract volumes
    df_to_vol = df_to[["FM", "Volume"]].rename(columns={"Volume": "Volume_To"})
    df_sub_vol = df_sub[["FM", "Volume"]].rename(columns={"Volume": "Volume_SubFactory"})

    # Merge volumes into 'From'
    merged = df_from.merge(df_to_vol, on="FM", how="left")
    merged = merged.merge(df_sub_vol, on="FM", how="left")

    # Calculate volume shifts
    merged["Volume_From_To"] = merged["Volume_To"]
    merged["Volume_To_SubFactory"] = merged["Volume_SubFactory"]

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




# ---------- Sidebar File Upload ----------
DEFAULT_FILE_URL = "https://raw.githubusercontent.com/ManjunathBiradar-01/Factory-relocation-map/main/Footprint_SDR.xlsx"

with st.sidebar:
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    st.session_state["excel_file"] = uploaded_file
elif "excel_file" in st.session_state:
    uploaded_file = st.session_state["excel_file"]
else:
    try:
        response = requests.get(DEFAULT_FILE_URL)
        response.raise_for_status()
        uploaded_file = BytesIO(response.content)
        st.session_state["excel_file"] = uploaded_file
        st.sidebar.info("Using default file from GitHub.")
    except Exception as e:
        st.error(f"Failed to load default file from GitHub.\n\n{e}")
        st.stop()

# ---------- Data loader ----------
@st.cache_data(show_spinner=False)
def load_data(path: BytesIO) -> pd.DataFrame:
    df_from = pd.read_excel(path, sheet_name="From", engine="openpyxl")
    df_to = pd.read_excel(path, sheet_name="To", engine="openpyxl")
    df_sub = pd.read_excel(path, sheet_name="Sub-Factory", engine="openpyxl")
    for d in (df_from, df_to, df_sub):
        d.columns = d.columns.str.strip()
    df_from = df_from.rename(columns={"Latitude": "Lat_today", "Longitude": "Lon_today", "Volume" : "main_vol"})
    df_to = df_to.rename(columns={"Latitude": "Lat_lead", "Longitude": "Lon_lead", "Volume" : "lead_vol"})
    df_sub = df_sub.rename(columns={"Latitude": "Lat_sub", "Longitude": "Lon_sub", "Volume" : "sub_vol"})
    df_to_keep = df_to[["FM", "Name", "Plan Lead Factory", "Lat_lead", "Lon_lead", "lead_vol" ]].copy()
    df_sub_keep = df_sub[["FM","Name", "Plan Sub Factory", "Lat_sub", "Lon_sub",  "sub_vol" ]].copy()
    merged = df_from.merge(df_to_keep, on="FM", how="left").merge(df_sub_keep, on="FM", how="left")
    for c in ["Lat_today", "Lon_today", "Lat_lead", "Lon_lead", "Lat_sub", "Lon_sub"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    return merged

# ---------- Load Data ----------
try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Failed to load data.\n\n{e}")
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


import folium
import pandas as pd
import numpy as np
from folium.plugins import AntPath
from folium import JavascriptLink, Element

import folium
from folium.plugins import AntPath
from folium import JavascriptLink, Element

# Center map based on available coordinates
coords = []
if not filtered_df.empty:
    coords.extend(filtered_df[["Lat_today", "Lon_today"]].dropna().values.tolist())
    coords.extend(filtered_df[["Lat_lead", "Lon_lead"]].dropna().values.tolist())

center_lat = float(np.mean([c[0] for c in coords])) if coords else 20.0
center_lon = float(np.mean([c[1] for c in coords])) if coords else 0.0

m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles="OpenStreetMap")

# Load Leaflet arrowheads plugin
m.get_root().header.add_child(JavascriptLink(
    "https://unpkg.com/leaflet-arrowheads@1.2.2/src/leaflet-arrowheads.js"
))

# Custom CSS for better styling
FONT_SIZE_PX = 16
css = f"""
<style>
.leaflet-tooltip {{
    font-size: {FONT_SIZE_PX}px;
    font-weight: 600;
    color: #111;
}}
.leaflet-popup-content {{
    font-size: {FONT_SIZE_PX}px;
    line-height: 1.35;
    color: #111;
}}
.leaflet-popup-content-wrapper {{
    padding: 8px 12px;
}}
@media (max-width: 768px) {{
    .leaflet-tooltip,
    .leaflet-popup-content {{
        font-size: {FONT_SIZE_PX + 2}px;
    }}
}}
</style>
"""
m.get_root().header.add_child(Element(css))

# Normalize names and coerce volumes
filtered_df = filtered_df.copy()

for col in ["Factory today", "Plan Lead Factory"]:
    filtered_df[col] = (
        filtered_df[col]
        .astype(str)
        .str.strip()
    )

for vol_col in ["lead_vol", "main_vol"]:
    if vol_col in filtered_df.columns:
        filtered_df[vol_col] = pd.to_numeric(filtered_df[vol_col], errors="coerce")

# Group volume by route on normalized names
grouped = (
    filtered_df
    .dropna(subset=["Factory today", "Plan Lead Factory"])
    .groupby(["Factory today", "Plan Lead Factory"], as_index=False)["lead_vol"]
    .sum()
)

volume_lookup = {
    (r["Factory today"], r["Plan Lead Factory"]): r["lead_vol"]
    for _, r in grouped.iterrows()
}

# Plot markers and flows
bounds = []
for _, row in filtered_df.iterrows():
      factory_name = row.get("Factory today", "n/a")  # already stripped above
      lead_factory_name = row.get("Plan Lead Factory", "n/a")
      main_vol = row.get("main_vol", "n/a")
      lead_vol = row.get("lead_vol", "n/a")
      lat_today = row.get("Lat_today", None)
      lon_today = row.get("Lon_today", None)
      lat_lead = row.get("Lat_lead", None)
      lon_lead = row.get("Lon_lead", None)
      sales_region = row.get(sales_region_col, "n/a") if sales_region_col else "n/a"

    # Add markers
      if pd.notnull(lat_today) and pd.notnull(lon_today):
          tooltip = f"{factory_name} | Main Vol: {main_vol}"
          popup = f"<b>Factory:</b> {factory_name}<br><b>Main Volume:</b> {main_vol}<br><b>Sales Region:</b> {sales_region}"
          folium.Marker(
              [lat_today, lon_today],
              tooltip=tooltip,
              popup=folium.Popup(popup, max_width=320),
              icon=folium.Icon(color="red", icon="industry", prefix="fa")
          ).add_to(m)

      if pd.notnull(lat_lead) and pd.notnull(lon_lead):
          tooltip = f"{lead_factory_name} | Lead Vol: {lead_vol}"
          popup = f"<b>Lead Factory:</b> {lead_factory_name}<br><b>Lead Volume:</b> {lead_vol}<br><b>Sales Region:</b> {sales_region}"
          folium.Marker(
              [lat_lead, lon_lead],
              tooltip=tooltip,
              popup=folium.Popup(popup, max_width=320),
              icon=folium.Icon(color="blue", icon="flag", prefix="fa")
          ).add_to(m)

    # Draw flow path with summed volume
    if pd.notnull(lat_today) and pd.notnull(lon_today) and pd.notnull(lat_lead) and pd.notnull(lon_lead):
        route_key = (factory_name, lead_factory_name)
        total_volume = volume_lookup.get(route_key, None)
        vol_txt = f"{total_volume:,.0f}" if total_volume is not None else "n/a"
       
        tooltip_html = f"{factory_name} → {lead_factory_name}<br>Volume: {vol_txt}"
        popup_html = f"<b>From:</b> {factory_name} → <b>To:</b> {lead_factory_name}<br><b>Volume:</b> {vol_txt}"

        path = AntPath(
            locations=[[lat_today, lon_today], [lat_lead, lon_lead]],
            color="#e63946",
            weight=5,
            opacity=0.9,
            dash_array=[10, 20],
            delay=800,
            pulse_color="#ffd166",
            paused=False,
            reverse=False,
            hardware_accelerated=True
        )
        folium.Tooltip(tooltip_html, sticky=True).add_to(path)
        folium.Popup(popup_html, max_width=320).add_to(path)
        path.add_to(m)

        # Add arrowheads
        arrow_js = f"""
        <script>
        try {{
            var lyr = {path.get_name()};
            if (lyr && typeof lyr.arrowheads === 'function') {{
                lyr.arrowheads({{
                    size: '16px',
                    frequency: 'endonly',
                    yawn: 45,
                    fill: true,
                    color: '#e63946'
                }});
            }}
        }} catch (e) {{
            console.warn('Arrowheads plugin failed:', e);
        }}
        </script>
        """
        m.get_root().html.add_child(Element(arrow_js))
        bounds.extend([[lat_today, lon_today], [lat_lead, lon_lead]])

# Fit map to bounds
if bounds:
    m.fit_bounds(bounds)

# Render in Streamlit
st.subheader("Production Relocation Map")
st.components.v1.html(m._repr_html_(), height=600)




# ---------- Map centering ----------
coords = []
if not filtered_df.empty:
    coords.extend(filtered_df[["Lat_lead", "Lon_lead"]].dropna().values.tolist())
    coords.extend(filtered_df[["Lat_sub", "Lon_sub"]].dropna().values.tolist())

if coords:
    center_lat = float(np.mean([c[0] for c in coords]))
    center_lon = float(np.mean([c[1] for c in coords]))
else:
    center_lat, center_lon = 20.0, 0.0  # global fallback

m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles="OpenStreetMap")

# Load Leaflet arrowheads plugin
m.get_root().header.add_child(JavascriptLink(
    "https://unpkg.com/leaflet-arrowheads@1.2.2/src/leaflet-arrowheads.js"
))

# ---------- Custom CSS ----------
FONT_SIZE_PX = 16
css = f"""
<style>
  .leaflet-tooltip {{
    font-size: {FONT_SIZE_PX}px;
    font-weight: 600;
    color: #111;
  }}
  .leaflet-popup-content {{
    font-size: {FONT_SIZE_PX}px;
    line-height: 1.35;
    color: #111;
  }}
  .leaflet-popup-content-wrapper {{
    padding: 8px 12px;
  }}
  @media (max-width: 768px) {{
    .leaflet-tooltip,
    .leaflet-popup-content {{
      font-size: {FONT_SIZE_PX + 2}px;
    }}
  }}
</style>
"""
m.get_root().header.add_child(Element(css))

# ---------- Group volume by route ----------
grouped = filtered_df.groupby(["Plan Sub Factory", "Plan Lead Factory"])["sub_vol"].sum().reset_index()
grouped = grouped[grouped["sub_vol"] > 0]  # ✅ Only keep routes with volume > 0
volume_lookup = {
    (row["Plan Sub Factory"], row["Plan Lead Factory"]): row["sub_vol"]
    for _, row in grouped.iterrows()
}

# ---------- Plot markers & flows ----------
bounds = []

for _, row in filtered_df.iterrows():
    lat_sub, lon_sub = row["Lat_sub"], row["Lon_sub"]
    lat_lead,  lon_lead  = row["Lat_lead"],  row["Lon_lead"]

    sales_region_line = ""
    if sales_region_col and pd.notnull(row.get(sales_region_col, None)):
        sales_region_line = f"<br><b>Sales Region:</b> {row.get(sales_region_col, '')}"

    if pd.notnull(lat_today) and pd.notnull(lon_today):
        folium.Marker(
            [lat_lead, lon_lead],
            popup=folium.Popup(
                f"<b></b> {row.get('plan lead factory','')}",
                max_width=320
            ),
            icon=folium.Icon(color="red", icon="industry", prefix="fa"),
            tooltip="Factory Today"
        ).add_to(m)

    if pd.notnull(lat_sub) and pd.notnull(lon_sub):
        folium.Marker(
            [lat_sub, lon_sub],
            popup=folium.Popup(
                f"<b></b> {row.get('Plan sub Factory','')}",
                max_width=320
            ),
            icon=folium.Icon(color="blue", icon="flag", prefix="fa"),
            tooltip="Plan sub Factory"
        ).add_to(m)

    if (pd.notnull(lat_today) and pd.notnull(lon_today) and
        pd.notnull(lat_lead)  and pd.notnull(lon_lead)):

        
        sub_name = row.get("Plan Sub Factory", "n/a")
        if isinstance(sub_name, str):
            sub_name = sub_name.strip() or "n/a"

        to_name_raw = row.get("Plan lead Factory", "n/a")
    to_name = to_name_raw.strip() if isinstance(to_name_raw, str) else str(to_name_raw) or "n/a"
    route_key = (to_name, sub_name)
    total_volume = volume_lookup.get(route_key, None)
    vol_txt = f"{total_volume:.0f}" if total_volume is not None else "n/a"

    tooltip_html = f"{from_name} → {to_name}<br>Volume: {vol_txt}"
    popup_html   = (
        f"<b>From:</b> {from_name} → <b>To:</b> {to_name}<br>"
        f"<b>Volume:</b> {vol_txt}"
    )

    path = AntPath(
        locations=[[lat_today, lon_today], [lat_lead, lon_lead]],
        color="#e63946",
        weight=5,
        opacity=0.9,
        dash_array=[10, 20],
        delay=800,
        pulse_color="#ffd166",
        paused=False,
        reverse=False,
        hardware_accelerated=True
    )
    folium.Tooltip(tooltip_html, sticky=True).add_to(path)
    folium.Popup(popup_html, max_width=320).add_to(path)
    path.add_to(m)

    arrow_js = f"""
    <script>
    try {{
        var lyr = {path.get_name()};
        if (lyr && typeof lyr.arrowheads === 'function') {{
        lyr.arrowheads({{
            size: '16px',
            frequency: 'endonly',
            yawn: 45,
            fill: true,
            color: '#e63946'
        }});
        }}
    }} catch (e) {{
        console.warn('Arrowheads plugin failed:', e);
    }}
    </script>
    """
    m.get_root().html.add_child(Element(arrow_js))

    bounds.extend([[lat_today, lon_today], [lat_lead, lon_lead]])
# Optional: Fit map to bounds
if bounds:
    m.fit_bounds(bounds)


# Keep for auto-zoom
bounds.extend([[lat_today, lon_today], [lat_lead, lon_lead]])

# Auto-zoom to all drawn flows
if bounds:
    m.fit_bounds(bounds)



# ---------- Render ----------
st.subheader("Lead Factory to Sub Factory")
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






































































































