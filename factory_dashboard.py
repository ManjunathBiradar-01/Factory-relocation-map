import pandas as pd
import numpy as np
import folium
import streamlit as st
import requests
from io import BytesIO


# ---------- Custom Sidebar Background Color ----------
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: rgba(255, 220, 67, 0.6); /* 0.6 = 60% opacity */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Custom Header Styling ----------
st.markdown(
    """
    <style>
        .custom-header {
            background-color: #0052a0;
            color: white;
            padding: 10px 20px;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
    <div class="custom-header">
        Bomag SDMs Factory Production Relocation Dashboard
    </div>
    """,
    unsafe_allow_html=True
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
        import requests
        from io import BytesIO
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



with st.sidebar:
    machine_code_filter = st.multiselect(
        "Machine Code (FM)",
        options=sorted(df["FM"].dropna().astype(str).unique().tolist())
    )
    machine_name_filter = st.multiselect(
        "Machine Name",
        options=sorted(df["Name"].dropna().astype(str).unique().tolist())
    )
    engine_filter = st.multiselect(
        "Select Engine Type",
        options=sorted(df["Engine"].dropna().astype(str).unique().tolist())
    )
    emission_filter = st.multiselect(
        "Select Emission Level",
        options=sorted(df["Emission"].dropna().astype(str).unique().tolist())
    )
    sales_region_col = find_sales_region_col(df.columns)
    if sales_region_col:
        sales_region_filter = st.multiselect(
            "Sales Region",
            options=sorted(df[sales_region_col].dropna().astype(str).unique().tolist())
        )
    else:
        sales_region_filter = []


def normalize_factory_name(name):
    return str(name).strip().lower()



# === 0) Map Style Selector ===
with st.sidebar:
    st.header("Map Settings")
    tile_options = {
        "OpenStreetMap": "OpenStreetMap",
        "CartoDB Positron": "CartoDB positron",
        "CartoDB Dark Matter": "CartoDB dark_matter"
    }
    selected_tile = st.selectbox("Choose Map Style", list(tile_options.keys()))

# ---------- UI (updated) ----------

# Detect sales region column dynamically
sales_region_col = find_sales_region_col(df.columns)

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



# Friendly coordinate strings (optional for table)
filtered_df["Coords_today"] = filtered_df.apply(lambda r: format_coords(r["Lat_today"], r["Lon_today"]), axis=1)
filtered_df["Coords_lead"]  = filtered_df.apply(lambda r: format_coords(r["Lat_lead"],  r["Lon_lead"]),  axis=1)
filtered_df["Coords_sub"]   = filtered_df.apply(lambda r: format_coords(r["Lat_sub"],   r["Lon_sub"]),   axis=1)



# === Additional KPIs ===


# You can make this dynamic using a dropdown if needed
USA = "South Carolina, USA"  
Germany = "Boppard, Germany"
China = "Changzhou, China"
India = "Pune, India"



# Custom CSS for KPI cards
st.markdown("""
    <style>
    .kpi-card {
        background-color: #ffdc43;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        font-family: 'Arial', sans-serif;
        margin-bottom: 20px;
    }
    .kpi-title {
        font-size: 15px;
        font-weight: 300;
        color: black;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: bolder;
        color: black;
        font-family: 'Roboto', monospace;
    }
    </style>
""", unsafe_allow_html=True)

# Layout for KPI cards
kpi_main, kpi_lead, kpi_sub = st.columns(3)

with kpi_main:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Main Factories</div>
            <div class="kpi-value">{filtered_df["Factory today"].nunique()}</div>
            <div class="kpi-title">Main Volume</div>
            <div class="kpi-value">{filtered_df['main_vol'].sum()}</div>
        </div>
    """, unsafe_allow_html=True)

with kpi_lead:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Lead Factories</div>
            <div class="kpi-value">{filtered_df["Plan Lead Factory"].nunique()}</div>
            <div class="kpi-title">Lead Volume</div>
            <div class="kpi-value">{filtered_df['lead_vol'].sum()}</div>
        </div>
    """, unsafe_allow_html=True)

with kpi_sub:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Sub Factories</div>
            <div class="kpi-value">{filtered_df["Plan Sub Factory"].nunique()}</div>
            <div class="kpi-title">Sub Volume</div>
            <div class="kpi-value">{filtered_df['sub_vol'].sum():,.0f}</div>
        </div>
    """, unsafe_allow_html=True)



import folium
import pandas as pd
import numpy as np
from folium.plugins import AntPath
from folium import JavascriptLink, Element


# === 0) Normalize keys & coerce numeric BEFORE any grouping / plotting ===
filtered_df = filtered_df.copy()

# Ensure names match between grouping and lookup (strip whitespace)
for col in ["Factory today", "Plan Lead Factory"]:
    if col in filtered_df.columns:
        filtered_df[col] = filtered_df[col].astype(str).str.strip()

# Coerce volume columns to numeric
for vcol in ["lead_vol", "main_vol"]:
    if vcol in filtered_df.columns:
        filtered_df[vcol] = pd.to_numeric(filtered_df[vcol], errors="coerce")

# Center map based on available coordinates
coords = []
if not filtered_df.empty:
    coords.extend(filtered_df[["Lat_today", "Lon_today"]].dropna().values.tolist())
    coords.extend(filtered_df[["Lat_lead", "Lon_lead"]].dropna().values.tolist())

center_lat = float(np.mean([c[0] for c in coords])) if coords else 20.0
center_lon = float(np.mean([c[1] for c in coords])) if coords else 0.0

m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles=tile_options[selected_tile])

# Load Leaflet arrowheads plugin
m.get_root().header.add_child(JavascriptLink(
    "https://unpkg.com/leaflet-arrowheads@1.2.2/src/leaflet-arrowheads.js"
))

# Custom CSS for better styling  (✅ use real <style> tags, not escaped)
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

# === 1) Build unique coordinates per factory ===
coords_today = (
    filtered_df.dropna(subset=["Factory today", "Lat_today", "Lon_today"])
    .drop_duplicates("Factory today")
    .set_index("Factory today")[["Lat_today", "Lon_today"]]
    .to_dict("index")
)

coords_lead = (
    filtered_df.dropna(subset=["Plan Lead Factory", "Lat_lead", "Lon_lead"])
    .drop_duplicates("Plan Lead Factory")
    .set_index("Plan Lead Factory")[["Lat_lead", "Lon_lead"]]
    .to_dict("index")
)

# === 2) Aggregate volumes ===
routes = (
    filtered_df.dropna(subset=["Factory today", "Plan Lead Factory"])
    .groupby(["Factory today", "Plan Lead Factory"], as_index=False)["lead_vol"]
    .sum()
)

main_by_factory = (
    filtered_df.dropna(subset=["Factory today"])
    .groupby("Factory today", as_index=False)["main_vol"]
    .sum()
)

lead_by_factory = (
    filtered_df.dropna(subset=["Plan Lead Factory"])
    .groupby("Plan Lead Factory", as_index=False)["lead_vol"]
    .sum()
)

# (Optional) representative Sales Region (mode)
if sales_region_col:
    region_today = (
        filtered_df.dropna(subset=["Factory today"])
        .groupby("Factory today")[sales_region_col]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "n/a")
    )
    region_lead = (
        filtered_df.dropna(subset=["Plan Lead Factory"])
        .groupby("Plan Lead Factory")[sales_region_col]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "n/a")
    )
else:
    # safe defaults so we can reference below
    region_today = pd.Series(dtype="object")
    region_lead = pd.Series(dtype="object")

# === 3) 'Today' factory markers once each (aggregated main_vol) ===
lead_factories_clean = lead_by_factory["Plan Lead Factory"].astype(str).str.strip().str.lower()

for _, r in main_by_factory.iterrows():
    f = str(r["Factory today"]).strip()
    if f in coords_today:
        lat_today = coords_today[f]["Lat_today"]
        lon_today = coords_today[f]["Lon_today"]
        vol_txt = f"{r['main_vol']:,.0f}" if pd.notnull(r["main_vol"]) else "n/a"
        sr = region_today[f] if sales_region_col and f in region_today.index else "n/a"

        # Check for matching lead factory
        lead_vol = lead_by_factory.loc[lead_by_factory["Plan Lead Factory"].astype(str).str.strip().str.lower() == f.lower(), "lead_vol"].sum()
        lead_vol_txt = f"{lead_vol:,.0f}" if lead_vol > 0 else "n/a"

        tooltip = f"{f} | Main Vol: {vol_txt} | Lead Vol: {lead_vol_txt}"
        popup = (
            f"<b>Factory:</b> {f}"
            f"<br><b>Main Volume:</b> {vol_txt}"
            f"<br><b>Lead Volume:</b> {lead_vol_txt}"
            + (f"<br><b>Sales Region:</b> {sr}" if sales_region_col else "")
        )

        folium.Marker(
            [lat_today, lon_today],
            tooltip=tooltip,
            popup=folium.Popup(popup, max_width=320),
            icon=folium.Icon(color="red", icon="industry", prefix="fa")
        ).add_to(m)

# === 4) 'Lead' factory markers once each (aggregated lead_vol) ===
main_factories_clean = main_by_factory["Factory today"].astype(str).str.strip().str.lower()

for _, r in lead_by_factory.iterrows():
    f = str(r["Plan Lead Factory"]).strip()
    if f in coords_lead:
        lat_lead = coords_lead[f]["Lat_lead"]
        lon_lead = coords_lead[f]["Lon_lead"]
        vol_txt = f"{r['lead_vol']:,.0f}" if pd.notnull(r["lead_vol"]) else "n/a"
        sr = region_lead[f] if sales_region_col and f in region_lead.index else "n/a"

        # Check for matching main factory
        main_vol = main_by_factory.loc[main_by_factory["Factory today"].astype(str).str.strip().str.lower() == f.lower(), "main_vol"].sum()
        main_vol_txt = f"{main_vol:,.0f}" if main_vol > 0 else "n/a"

        tooltip = f"{f} | Lead Vol: {vol_txt} | Main Vol: {main_vol_txt}"
        popup = (
            f"<b>Lead Factory:</b> {f}"
            f"<br><b>Lead Volume:</b> {vol_txt}"
            f"<br><b>Main Volume:</b> {main_vol_txt}"
            + (f"<br><b>Sales Region:</b> {sr}" if sales_region_col else "")
        )

        folium.Marker(
            [lat_lead, lon_lead],
            tooltip=tooltip,
            popup=folium.Popup(popup, max_width=320),
            icon=folium.Icon(color="blue", icon="flag", prefix="fa")
        ).add_to(m)
# === 5) Draw each route once with summed lead_vol ===
bounds = []
for _, r in routes.iterrows():
    fr = r["Factory today"]
    to = r["Plan Lead Factory"]
    vol = r["lead_vol"]

    if fr in coords_today and to in coords_lead:
        lat_today = coords_today[fr]["Lat_today"]
        lon_today = coords_today[fr]["Lon_today"]
        lat_lead = coords_lead[to]["Lat_lead"]
        lon_lead = coords_lead[to]["Lon_lead"]

        lead_vol_txt = f"{vol:,.0f}" if pd.notnull(vol) else "n/a"

        tooltip_html = f"{fr} → {to}<br>Lead Volume: {lead_vol_txt}"
        popup_html = (
            f"<b>From:</b> {fr} → <b>To:</b> {to}<br>"
            f"<b>Lead Volume:</b> {lead_vol_txt}"
        )

        path = AntPath(
            locations=[[lat_today, lon_today], [lat_lead, lon_lead]],
            color="#0052a0",
            weight=5,
            opacity=0.9,
            dash_array=[10, 20],
            delay=800,
            pulse_color="#ffdc43",
            paused=False,
            reverse=False,
            hardware_accelerated=True
        )
        folium.Tooltip(tooltip_html, sticky=True).add_to(path)
        folium.Popup(popup_html, max_width=320).add_to(path)
        path.add_to(m)

        bounds.extend([[lat_today, lon_today], [lat_lead, lon_lead]])

# === 5b) Apply arrowheads ONCE after all paths are on the map ===
arrowheads_once_js = f"""
<script>
(function() {{
  var map = {m.get_name()};
  function applyArrowheads() {{
    try {{
      Object.values(map._layers || {{}}).forEach(function(layer) {{
        if (layer && typeof layer.arrowheads === 'function') {{
          layer.arrowheads({{
            size: '16px',
            frequency: 'endonly',
            yawn: 45,
            fill: true,
            color: '#e63946'
          }});
        }}
      }});
    }} catch (e) {{
      console.warn('Arrowheads plugin failed:', e);
    }}
  }}
  map.whenReady(function() {{
    applyArrowheads();
    map.on('layeradd', applyArrowheads);
  }});
}})();
</script>
"""
m.get_root().html.add_child(Element(arrowheads_once_js))

# === 6) Fit map to all aggregated bounds ===
if bounds:
    m.fit_bounds(bounds)
# Render in Streamlit
st.subheader("Main Factory To Lead Factory")
st.components.v1.html(m._repr_html_(), height=400)


# Summarize volumes by factory type
lead_summary = filtered_df.groupby("Plan Lead Factory")["lead_vol"].sum().reset_index()
sub_summary = filtered_df.groupby("Plan Sub Factory")["sub_vol"].sum().reset_index()
main_summary = filtered_df.groupby("Factory today")["main_vol"].sum().reset_index()

# Rename columns to a common key for merging
lead_summary = lead_summary.rename(columns={"Plan Lead Factory": "Factory"})
sub_summary = sub_summary.rename(columns={"Plan Sub Factory": "Factory"})
main_summary = main_summary.rename(columns={"Factory today": "Factory"})

# Merge all summaries on 'Factory'
merged_df = pd.merge(main_summary, lead_summary, on="Factory", how="outer")
merged_df = pd.merge(merged_df, sub_summary, on="Factory", how="outer")

# Fill missing values with 0 and convert to integers
merged_df = merged_df.fillna(0)
merged_df[["main_vol", "lead_vol", "sub_vol"]] = merged_df[["main_vol", "lead_vol", "sub_vol"]].astype(int)

# === Apply styling and remove index ===
styled_html = merged_df.style.set_table_styles([
    {
        'selector': 'table',
        'props': [('width', '100%')]
    },
    {
        'selector': 'th',
        'props': [
            ('background-color', '#f9f9f9'),
            ('color', 'black'),
            ('font-size', '16px'),
            ('font-family', 'Arial'),
            ('font-weight', 'bold'),
            ('text-align', 'center')
        ]
    },
    {
        'selector': 'td',
        'props': [
            ('font-size', '14px'),
            ('font-family', 'Arial'),
            ('background-color', 'white')
        ]
    }
]).hide(axis="index").to_html()

# === Display the styled table ===
st.markdown("### Combined Factory Summary", unsafe_allow_html=True)
st.markdown(styled_html, unsafe_allow_html=True)

#2nd map 
# === 0) Normalize keys & coerce numeric BEFORE any grouping / plotting ===
filtered_df = filtered_df.copy()

# Keep NA as NA while trimming whitespace
for col in ["Plan Lead Factory", "Plan Sub Factory"]:
    if col in filtered_df.columns:
        filtered_df[col] = filtered_df[col].astype("string").str.strip()

# Coerce volume columns to numeric
for vcol in ["lead_vol", "sub_vol"]:
    if vcol in filtered_df.columns:
        filtered_df[vcol] = pd.to_numeric(filtered_df[vcol], errors="coerce")

# === 0b) Keep only rows that can form a connection to Sub and have positive volume ===
df_pos = filtered_df[
    filtered_df["Plan Sub Factory"].notna()
    & filtered_df["Plan Lead Factory"].notna()
    & pd.notnull(filtered_df["Lat_lead"]) & pd.notnull(filtered_df["Lon_lead"])
    & pd.notnull(filtered_df["Lat_sub"])  & pd.notnull(filtered_df["Lon_sub"])
    & (filtered_df["sub_vol"] > 0)
].copy()

# If nothing to draw, exit gracefully
if df_pos.empty:
    st.subheader("Lead Factory To Sub Factory")
    st.info("No Sub routes with sub_vol > 0 for the current filters.")
    st.stop()

# Center map based on available coordinates (use the positive subset)
coords = []
coords.extend(df_pos[["Lat_lead", "Lon_lead"]].dropna().values.tolist())
coords.extend(df_pos[["Lat_sub", "Lon_sub"]].dropna().values.tolist())

center_lat = float(np.mean([c[0] for c in coords])) if coords else 20.0
center_lon = float(np.mean([c[1] for c in coords])) if coords else 0.0

m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles=tile_options[selected_tile])

# Load Leaflet arrowheads plugin
m.get_root().header.add_child(JavascriptLink(
    "https://unpkg.com/leaflet-arrowheads@1.2.2/src/leaflet-arrowheads.js"
))

# Custom CSS (✅ real HTML)
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

# === 1) Build unique coordinates per factory (from positive routes only) ===
coords_lead = (
    df_pos.dropna(subset=["Plan Lead Factory", "Lat_lead", "Lon_lead"])
    .drop_duplicates("Plan Lead Factory")
    .set_index("Plan Lead Factory")[["Lat_lead", "Lon_lead"]]
    .to_dict("index")
)

coords_sub = (
    df_pos.dropna(subset=["Plan Sub Factory", "Lat_sub", "Lon_sub"])
    .drop_duplicates("Plan Sub Factory")
    .set_index("Plan Sub Factory")[["Lat_sub", "Lon_sub"]]
    .to_dict("index")
)

# === 2) Aggregate volumes (from positive routes only) ===
routes = (
    df_pos
    .groupby(["Plan Lead Factory", "Plan Sub Factory"], as_index=False)["sub_vol"]
    .sum()
)


# Lead factory: total main_vol across rows that participate in a positive Sub connection
lead_by_factory = (
    df_pos.groupby("Plan Lead Factory", as_index=False)["lead_vol"].sum()
)

# Sub factory: total sub_vol across positive connections
sub_by_factory = (
    df_pos.groupby("Plan Sub Factory", as_index=False)["sub_vol"].sum()
)

# (Optional) representative Sales Region (mode) for each factory
if sales_region_col:
    region_lead = (
        df_pos.groupby("Plan Lead Factory")[sales_region_col]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "n/a")
    )
    region_sub = (
        df_pos.groupby("Plan Sub Factory")[sales_region_col]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "n/a")
    )
else:
    region_lead = pd.Series(dtype="object")
    region_sub = pd.Series(dtype="object")

# Clean factory names for reliable matching
lead_factories_clean = lead_by_factory["Plan Lead Factory"].astype(str).str.strip().str.lower()
sub_factories_clean = sub_by_factory["Plan Sub Factory"].astype(str).str.strip().str.lower()


# === 4) 'Lead' factory markers once each (aggregated lead_vol) ===
for _, r in lead_by_factory.iterrows():
    f = str(r["Plan Lead Factory"]).strip()
    if f in coords_lead:
        lat_lead = coords_lead[f]["Lat_lead"]
        lon_lead = coords_lead[f]["Lon_lead"]
        lead_vol_txt = f"{r['lead_vol']:,.0f}" if pd.notnull(r["lead_vol"]) else "n/a"
        sr = region_lead[f] if sales_region_col and f in region_lead.index else "n/a"

        # Get sub volume for this lead factory
        sub_vol = sub_by_factory.loc[
            sub_by_factory["Plan Sub Factory"].astype(str).str.strip().str.lower() == f.lower(),
            "sub_vol"
        ].sum()
        sub_vol_txt = f"{sub_vol:,.0f}" if sub_vol > 0 else "n/a"

        tooltip = f"{f} | Lead Vol: {lead_vol_txt} | Sub Vol: {sub_vol_txt}"
        popup = (
            f"<b>Lead Factory:</b> {f}"
            f"<br><b>Lead Volume:</b> {lead_vol_txt}"
            f"<br><b>Sub Volume:</b> {sub_vol_txt}"
            + (f"<br><b>Sales Region:</b> {sr}" if sales_region_col else "")
        )

        folium.Marker(
            [lat_lead, lon_lead],
            tooltip=tooltip,
            popup=folium.Popup(popup, max_width=320),
            icon=folium.Icon(color="blue", icon="flag", prefix="fa")
        ).add_to(m)

# === 3) 'Sub' factory markers once each (aggregated sub_vol) ===
lead_factories_clean = lead_by_factory["Plan Lead Factory"].astype(str).str.strip().str.lower()

for _, r in sub_by_factory.iterrows():
    f = str(r["Plan Sub Factory"]).strip()
    if f in coords_sub:
        lat_sub = coords_sub[f]["Lat_sub"]
        lon_sub = coords_sub[f]["Lon_sub"]
        sub_vol_txt = f"{r['sub_vol']:,.0f}" if pd.notnull(r["sub_vol"]) else "n/a"
        sr = region_sub[f] if sales_region_col and f in region_sub.index else "n/a"


        # print("lead_by_factory columns:", lead_by_factory.columns.tolist())

        # Check for matching lead factory
        lead_vol = lead_by_factory.loc[lead_by_factory["Plan Lead Factory"].astype(str).str.strip().str.lower() == f.lower(), "lead_vol"].sum()
        lead_vol_txt = f"{lead_vol:,.0f}" if lead_vol > 0 else "n/a"
        tooltip = f"{f} | Sub Vol: {sub_vol_txt} | Lead Vol: {lead_vol_txt}"
        popup = (
            f"<b>Factory:</b> {f}"
            f"<br><b>Sub Volume:</b> {sub_vol_txt}"
            f"<br><b>Lead Volume:</b> {lead_vol_txt}"
            + (f"<br><b>Sales Region:</b> {sr}" if sales_region_col else "")
        )

        folium.Marker(
            [lat_sub, lon_sub],
            tooltip=tooltip,
            popup=folium.Popup(popup, max_width=320),
            icon=folium.Icon(color="red", icon="industry", prefix="fa")
        ).add_to(m)

# === 5) Draw each route once with summed sub_vol ===
bounds = []
for _, r in routes.iterrows():
    fr = r["Plan Lead Factory"]
    to = r["Plan Sub Factory"]
    vol = r["sub_vol"]

    if fr in coords_lead and to in coords_sub:
        lat_lead = coords_lead[fr]["Lat_lead"]
        lon_lead = coords_lead[fr]["Lon_lead"]
        lat_sub = coords_sub[to]["Lat_sub"]
        lon_sub = coords_sub[to]["Lon_sub"]

        vol_txt = f"{vol:,.0f}" if pd.notnull(vol) else "n/a"
        tooltip_html = f"{fr} → {to}<br>Volume: {vol_txt}"
        popup_html = (
            f"<b>Lead:</b> {fr} → <b>Sub:</b> {to}"
            f"<br><b>Volume:</b> {vol_txt}"
        )

        path = AntPath(
            locations=[[lat_lead, lon_lead], [lat_sub, lon_sub]],
            color="#ffdc43",
            weight=5,
            opacity=0.9,
            dash_array=[10, 20],
            delay=800,
            pulse_color="#0052a0",
            paused=False,
            reverse=False,
            hardware_accelerated=True
        )
        folium.Tooltip(tooltip_html, sticky=True).add_to(path)
        folium.Popup(popup_html, max_width=320).add_to(path)
        path.add_to(m)

        bounds.extend([[lat_lead, lon_lead], [lat_sub, lon_sub]])

# === 5b) Apply arrowheads ONCE after all paths are on the map ===
arrowheads_once_js = f"""
<script>
(function() {{
  var map = {m.get_name()};
  function applyArrowheads() {{
    try {{
      Object.values(map._layers || {{}}).forEach(function(layer) {{
        if (layer && typeof layer.arrowheads === 'function') {{
          layer.arrowheads({{
            size: '16px',
            frequency: 'endonly',
            yawn: 45,
            fill: true,
            color: '#e63946'
          }});
        }}
      }});
    }} catch (e) {{
      console.warn('Arrowheads plugin failed:', e);
    }}
  }}
  map.whenReady(function() {{
    applyArrowheads();
    map.on('layeradd', applyArrowheads);
  }});
}})();
</script>
"""
m.get_root().html.add_child(Element(arrowheads_once_js))

# === 6) Fit map to all aggregated bounds ===
if bounds:
    m.fit_bounds(bounds)

# Render in Streamlit
st.subheader("Lead Factory To Sub Factory")
st.components.v1.html(m._repr_html_(), height=400)



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












































































































































































































































































