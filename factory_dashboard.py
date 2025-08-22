import pandas as pd
import numpy as np
import folium
import streamlit as st
import io
from datetime import datetime


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

import streamlit as st
import pandas as pd
import io

# --- Helper Functions ---
@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_excel(path, engine="openpyxl")

def find_sales_region_col(columns):
    possible_names = ['Sales Region', 'Main Sales Region', 'MainSales Region', 'SalesRegion']
    for name in possible_names:
        if name in columns:
            return name
    return None

# --- File Upload ---
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load data.{e}")
        st.stop()
else:
    try:
        df = load_data("https://raw.githubusercontent.com/ManjunathBiradar-01/Factory-relocation-map/main/Footprint_SDR.xlsx")
    except Exception as e:
        st.error(f"Failed to load default file from GitHub. {e}")
        st.stop()

# --- Tabs ---
tab1, tab2 = st.tabs(["Dashboard", "Edit Dataset"])

with tab1:
    st.title("Factory Production Relocation Dashboard")

    sales_region_col = find_sales_region_col(df.columns)

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

    st.dataframe(filtered_df)

with tab2:
    st.subheader("Edit Full Dataset")
    edited_df = st.data_editor(df, num_rows="dynamic")

    if st.button("Download Updated Excel File"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='UpdatedData')
        st.download_button(
            label="Click to Download",
            data=output.getvalue(),
            file_name="updated_factory_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )



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

# Load Leaflet arrowheads plugin (JS) once per map
from folium import JavascriptLink, Element

m.get_root().header.add_child(JavascriptLink(
    "https://unpkg.com/leaflet-arrowheads@1.2.2/src/leaflet-arrowheads.js"
))



from folium.plugins import AntPath


# After: m = folium.Map(...)

from folium import Element

# You can even parameterize this via a Streamlit slider (see Option C)
FONT_SIZE_PX = 16  # try 16–18 for desktop, maybe 18–20 for presentations

css = f"""
<style>
  /* All Leaflet tooltips */
  .leaflet-tooltip {{
    font-size: {FONT_SIZE_PX}px;
    font-weight: 600;        /* optional */
    color: #111;             /* tweak for dark/light themes */
  }}

  /* All popup content */
  .leaflet-popup-content {{
    font-size: {FONT_SIZE_PX}px;
    line-height: 1.35;
    color: #111;             /* tweak for dark/light themes */
  }}

  /* Optional: make popup wrapper spacing a bit roomier */
  .leaflet-popup-content-wrapper {{
    padding: 8px 12px;
  }}

  /* Optional: bump sizes on small screens */
  @media (max-width: 768px) {{
    .leaflet-tooltip,
    .leaflet-popup-content {{
      font-size: {FONT_SIZE_PX + 2}px;
    }}
  }}
</style>
"""
m.get_root().header.add_child(Element(css))




# --- Plot markers & flows (animated arrow) ---
bounds = []  # collect endpoints for fit_bounds

for _, row in filtered_df.iterrows():
    lat_today, lon_today = row["Lat_today"], row["Lon_today"]
    lat_lead,  lon_lead  = row["Lat_lead"],  row["Lon_lead"]

    # Optional Sales Region line for popup
    sales_region_line = ""
    if sales_region_col and pd.notnull(row.get(sales_region_col, None)):
        sales_region_line = f"<br><b>Sales Region:</b> {row.get(sales_region_col, '')}"

    # Markers (unchanged, with a small label fix on the lead factory popup)
    if pd.notnull(lat_today) and pd.notnull(lon_today):
        folium.Marker(
            [lat_today, lon_today],
            popup=folium.Popup(
                f"<b>Factory Today:</b> {row.get('Factory today','')}{sales_region_line}",
                max_width=320
            ),
            icon=folium.Icon(color="red", icon="industry", prefix="fa"),
            tooltip="Factory Today"
        ).add_to(m)

    if pd.notnull(lat_lead) and pd.notnull(lon_lead):
        folium.Marker(
            [lat_lead, lon_lead],
            popup=folium.Popup(
                f"<b>Plan Lead Factory:</b> {row.get('Plan Lead Factory','')}{sales_region_line}",
                max_width=320
            ),
            icon=folium.Icon(color="blue", icon="flag", prefix="fa"),
            tooltip="Plan Lead Factory"
        ).add_to(m)

    # Animated arrow path
    if (pd.notnull(lat_today) and pd.notnull(lon_today) and
        pd.notnull(lat_lead)  and pd.notnull(lon_lead)):

    # Volume formatting
        vol_raw = row.get("Volume Lead Plant (%)")
        try:
            vol_num = float(vol_raw) if pd.notnull(vol_raw) else None
        except Exception:
            vol_num = None

vol_txt = f"{vol_num:.0f}%" if vol_num is not None else ("n/a" if pd.isna(vol_raw) else str(vol_raw))


from_name = (row.get("Factory today", "") or "").strip() or "n/a"
to_name = (row.get("Plan Lead Factory", "") or "").strip() or "n/a"

tooltip_html = f"From: {from_name} → To: {to_name}<br>Volume Lead Plant: {vol_txt}"
popup_html   = (f"<b>From:</b> {from_name} → <b>To:</b> {to_name}<br>"
            f"<b>Volume Lead Plant:</b> {vol_txt}")

        # 1) Animated path (AntPath) – the moving dashes show direction
path = AntPath(
    locations=[[lat_today, lon_today], [lat_lead, lon_lead]],  # [lat, lon]
    color="#e63946",          # red
    weight=5,
    opacity=0.9,
    dash_array=[10, 20],      # pattern of dash/space
    delay=800,                # smaller is faster
    pulse_color="#ffd166",    # glow color
    paused=False,
    reverse=False,
    hardware_accelerated=True
)
        # Attach tooltip & popup
folium.Tooltip(tooltip_html, sticky=True).add_to(path)
folium.Popup(popup_html, max_width=320).add_to(path)
path.add_to(m)

        # 2) Add an arrowhead at the END of the path (via plugin)
arrow_js = f"""
<script>
try {{
    var lyr = {path.get_name()};
    if (lyr && typeof lyr.arrowheads === 'function') {{
    lyr.arrowheads({{
        size: '16px',
        frequency: 'endonly',   // only at the destination
        yawn: 45,               // arrow opening angle
        fill: true,
        color: '#e63946'        // match the path color
    }});
    }}
}} catch (e) {{
    console.warn('Arrowheads plugin failed:', e);
}}
</script>
"""
m.get_root().html.add_child(Element(arrow_js))

        # Keep for auto-zoom
bounds.extend([[lat_today, lon_today], [lat_lead, lon_lead]])

# Auto-zoom to all drawn flows
if bounds:
    m.fit_bounds(bounds)



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































































