# factory_dashboard_fixed.py

import io
from datetime import datetime

import numpy as np
import pandas as pd
import folium
from folium.plugins import AntPath
from folium import Element, JavascriptLink

import streamlit as st

# -------------------- Page settings --------------------
st.set_page_config(
    page_title="Bomag SDMs Factory Production Relocation Dashboard",
    layout="wide"
)

# -------------------- Data loader (single source of truth) --------------------
@st.cache_data(show_spinner=False)
def load_data(xlsx_file) -> pd.DataFrame:
    """
    Loads and merges the required sheets:
      - From (original location data)
      - To (lead factory coordinates)
      - Values (volume %)
    Performs basic validations and returns a single merged DataFrame.
    Accepts both file-like objects (Streamlit UploadedFile) and file paths/URLs.
    """
    # Read only the required sheets
    df_from = pd.read_excel(xlsx_file, sheet_name="From", engine="openpyxl")
    df_to   = pd.read_excel(xlsx_file, sheet_name="To", engine="openpyxl")
    df_val  = pd.read_excel(xlsx_file, sheet_name="Values", engine="openpyxl")

    # Normalize column names (trim spaces)
    for d in (df_from, df_to, df_val):
        d.columns = d.columns.str.strip()

    # Required columns check (helps debug early)
    required_from = {"FM", "Name", "Emission", "Engine", "Factory today", "Latitude", "Longitude"}
    required_to   = {"FM", "Plan Lead Factory", "Latitude", "Longitude"}
    required_val  = {"FM", "Volume Lead Plant (%)"}

    missing = [
        ("From",   required_from - set(df_from.columns)),
        ("To",     required_to   - set(df_to.columns)),
        ("Values", required_val  - set(df_val.columns)),
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


# -------------------- Helper: find 'Sales Region' column --------------------
def find_sales_region_col(columns) -> str | None:
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
        "mainsalesregion", "salesregion", "main_sales_region",
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


# -------------------- Small utility: format coordinates --------------------
def format_coords(lat, lon, decimals: int = 5) -> str:
    """Return a friendly 'lat, lon' string or 'n/a' if missing."""
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
    # Fallback to your GitHub file (make sure the URL is correct and public)
    default_url = "https://raw.githubusercontent.com/ManjunathBiradar-01/Factory-relocation-map/main/Footprint_SDR.xlsx"
    try:
        df = load_data(default_url)
        st.sidebar.info("Using default dataset from GitHub (upload a file to override).")
    except Exception as e:
        st.error(f"Failed to load default file from GitHub: {e}")
        st.stop()


# -------------------- Tabs --------------------
tab1, tab2 = st.tabs(["Dashboard", "Edit Dataset"])

with tab1:
    st.title("Factory Production Relocation Dashboard")

    sales_region_col = find_sales_region_col(df.columns)

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
    if sales_region_col and sales_region_filter:
        filtered_df = filtered_df[filtered_df[sales_region_col].astype(str).isin(sales_region_filter)]

    # -------------------- Map centering --------------------
    coords = []
    if not filtered_df.empty:
        coords.extend(filtered_df[["Lat_today", "Lon_today"]].dropna().values.tolist())
        coords.extend(filtered_df[["Lat_lead",  "Lon_lead"]].dropna().values.tolist())

    if coords:
        center_lat = float(np.mean([c[0] for c in coords]))
        center_lon = float(np.mean([c[1] for c in coords]))
    else:
        center_lat, center_lon = 20.0, 0.0  # global fallback

    m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles="OpenStreetMap")

    # Load Leaflet arrowheads plugin (once per map)
    m.get_root().header.add_child(JavascriptLink(
        "https://unpkg.com/leaflet-arrowheads@1.2.2/src/leaflet-arrowheads.js"
    ))

    # Tidy tooltip/popup CSS
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

    # -------------------- Plot markers & flows --------------------
    bounds = []  # collect endpoints for fit_bounds

    for _, row in filtered_df.iterrows():
        lat_today, lon_today = row["Lat_today"], row["Lon_today"]
        lat_lead,  lon_lead  = row["Lat_lead"],  row["Lon_lead"]

        # Optional Sales Region line for popup
        sales_region_line = ""
        if sales_region_col and pd.notnull(row.get(sales_region_col, None)):
            sales_region_line = f"<br><b>Sales Region:</b> {row.get(sales_region_col, '')}"

        # Markers
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

        # Flow with animated path
        if (
            pd.notnull(lat_today) and pd.notnull(lon_today)
            and pd.notnull(lat_lead) and pd.notnull(lon_lead)
        ):
            # Volume formatting
            vol_raw = row.get("Volume Lead Plant (%)")
            try:
                vol_num = float(vol_raw) if pd.notnull(vol_raw) else None
            except Exception:
                vol_num = None

            vol_txt = f"{vol_num:.0f}%" if vol_num is not None else ("n/a" if pd.isna(vol_raw) else str(vol_raw))
            from_name = (row.get("Factory today", "") or "").strip() or "n/a"
            to_name   = (row.get("Plan Lead Factory", "") or "").strip() or "n/a"

            tooltip_html = f"From: {from_name} → To: {to_name}<br>Volume Lead Plant: {vol_txt}"
            popup_html   = (
                f"<b>From:</b> {from_name} → <b>To:</b> {to_name}<br>"
                f"<b>Volume Lead Plant:</b> {vol_txt}"
            )

            path = AntPath(
                locations=[[lat_today, lon_today], [lat_lead, lon_lead]],
                color="#e63946",      # red
                weight=5,
                opacity=0.9,
                dash_array=[10, 20],  # pattern of dash/space
                delay=800,            # smaller is faster
                pulse_color="#ffd166",
                paused=False,
                reverse=False,
                hardware_accelerated=True
            )
            folium.Tooltip(tooltip_html, sticky=True).add_to(path)
            folium.Popup(popup_html, max_width=320).add_to(path)
            path.add_to(m)

            # Add an arrowhead at the END of the path via plugin
            # (path.get_name() is the JavaScript variable name emitted by folium)
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

            # Keep for auto-zoom
            bounds.extend([[lat_today, lon_today], [lat_lead, lon_lead]])

    # Auto-zoom to all drawn flows
    if bounds:
        m.fit_bounds(bounds)

    # -------------------- Render map --------------------
    st.subheader("Production Relocation Map")
    st.components.v1.html(m._repr_html_(), height=600)

    # -------------------- Optional: show filtered data with readable coords --------------------
    if not filtered_df.empty:
        filtered_df = filtered_df.copy()
        filtered_df["Factory Today Location"] = filtered_df.apply(
            lambda r: format_coords(r["Lat_today"], r["Lon_today"]), axis=1
        )
        filtered_df["Lead Factory Location"] = filtered_df.apply(
            lambda r: format_coords(r["Lat_lead"], r["Lon_lead"]), axis=1
        )

        with st.expander("Show filtered data"):
            cols_to_show = ["FM", "Name"]
            if sales_region_col:
                cols_to_show.append(sales_region_col)
            cols_to_show += [
                "Emission", "Engine", "Factory today",
                "Factory Today Location",
                "Plan Lead Factory",
                "Lead Factory Location",
                "Volume Lead Plant (%)",
                "Lat_today", "Lon_today", "Lat_lead", "Lon_lead",
            ]
            cols_to_show = [c for c in cols_to_show if c in filtered_df.columns]
            st.dataframe(filtered_df[cols_to_show].reset_index(drop=True))


with tab2:
    st.subheader("Edit Full Dataset")
    edited_df = st.data_editor(df, num_rows="dynamic")
    bcol1, bcol2 = st.columns([1, 3])
    with bcol1:
        if st.button("Prepare Download"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                edited_df.to_excel(writer, index=False, sheet_name="UpdatedData")
            st.session_state["_edited_bytes"] = output.getvalue()

    with bcol2:
        if "_edited_bytes" in st.session_state:
            st.download_button(
                label="Click to Download updated_factory_data.xlsx",
                data=st.session_state["_edited_bytes"],
                file_name="updated_factory_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )



































































