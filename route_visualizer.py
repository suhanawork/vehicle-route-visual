import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, box
import folium
from matplotlib import colors as mcolors
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from shapely.ops import transform
from pyproj import Transformer
import hashlib

def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["password_correct"]:
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password")
        st.stop()

check_password()

st.set_page_config(layout="wide")
st.title("LCV Route Heatmap Visualizer")

GPKG_PATH = r"https://digitsthra-my.sharepoint.com/:u:/p/suhana/EdBvm6WyDgRBnUVTctGHfQgBSgyrFsxpSwK4J0qmtXT0dQ?e=JpAdaq"
LAYER_NAME = "indialatestosm__lines"

@st.cache_resource
def get_gpkg_crs():
    road_layer = gpd.read_file(GPKG_PATH, layer=LAYER_NAME, rows=1)

    road_layer = road_layer[
        (road_layer["highway"].notnull()) & 
        (~road_layer["highway"].isin(["rail", "light_rail", "railway", "power", "proposed", "construction"]))
    ]

    road_layer = road_layer[
        road_layer["name"].notnull() & 
        (road_layer["name"].str.len() > 3)
    ]

    return road_layer.crs

@st.cache_data
def load_relevant_roads(bounds):
    roads_crs = get_gpkg_crs()
    bounds_geom = gpd.GeoSeries([box(*bounds)], crs="EPSG:4326").to_crs(roads_crs)
    bbox_tuple = tuple(bounds_geom.total_bounds)
    return gpd.read_file(GPKG_PATH, layer=LAYER_NAME, bbox=bbox_tuple).to_crs("EPSG:4326")

def interpolate_line(_line, interval_km=5.0):
    if _line.length == 0:
        return []
    project_to_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    project_to_deg = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    line_m = transform(project_to_m, _line)
    total_length_m = line_m.length
    interval_m = interval_km * 1000
    num_points = int(total_length_m // interval_m) + 1
    points_m = [line_m.interpolate(i * interval_m) for i in range(num_points + 1)]
    points_deg = [transform(project_to_deg, pt) for pt in points_m]
    return [(pt.y, pt.x) for pt in points_deg]

def hash_linestring(line):
    return hashlib.md5(line.wkb).hexdigest()

@st.cache_data
def process_trips(_gdf_points):
    _gdf_points = _gdf_points.sort_values("time_Stamp")
    trip_lines = _gdf_points.groupby("trip_Id")["geometry"].apply(
        lambda pts: LineString(pts.drop_duplicates().tolist()) if len(pts.drop_duplicates()) > 1 else None
    ).dropna()
    trip_lines_gdf = gpd.GeoDataFrame(trip_lines, geometry="geometry", crs="EPSG:4326").reset_index()

    interp_points = []
    route_point_map = {}
    for _, row in trip_lines_gdf.iterrows():
        trip_id = row["trip_Id"]
        coords = interpolate_line(row.geometry)
        interp_points.extend(coords)
        route_point_map[trip_id] = coords

    return trip_lines_gdf, interp_points, route_point_map

@st.cache_data
def get_top10_routes(_trip_lines_gdf, _roads_gdf):
    valid_road_types = ["highway", "expressway", "street", "road", "avenue", "boulevard", "lane", "drive"]

    valid_roads = _roads_gdf[_roads_gdf["name"].str.contains("|".join(valid_road_types), case=False, na=False)]
    
    intersected = gpd.sjoin(_trip_lines_gdf, valid_roads, how="inner", predicate="intersects")
    
    # Ensure there's a valid "name" field for the road
    road_column = "name" if "name" in valid_roads.columns else valid_roads.columns[0]
    
    # Remove rows with empty road names (important!)
    intersected = intersected[intersected[road_column] != ""]
    
    # Remove duplicate trips
    intersected = intersected.drop_duplicates(subset=["trip_Id", "geometry"])
    
    # Group by valid road names and calculate the trip count
    road_counts = intersected.groupby(road_column).agg(
        count=("trip_Id", lambda x: len(set(x))),
        trip_ids=("trip_Id", lambda x: list(set(x)))
    ).reset_index()

    # Sort by trip count
    road_counts = road_counts.sort_values("count", ascending=False)

    # Return top 10 roads based on trip count
    return road_counts.head(10), road_column

def create_route_map(gdf_points, route_point_map, top10, selected_road, road_column):
    from collections import defaultdict
    import numpy as np
    import folium

    # Initialize usage & road name maps for ALL trips
    trip_usage = defaultdict(int)
    trip_to_road = {}

    # Count all trips and map to roads from top10
    for _, row in top10.iterrows():
        road_name = row[road_column]
        trip_ids = row["trip_ids"]
        for trip_id in trip_ids:
            trip_usage[trip_id] += 1
            trip_to_road[trip_id] = road_name

    # Add remaining trips (not in top10) as 0-count with "Other"
    for trip_id in route_point_map:
        trip_usage.setdefault(trip_id, 0)
        trip_to_road.setdefault(trip_id, "Other")

    # Center map on selected route if present
    selected_coords = []
    if selected_road in top10[road_column].values:
        selected_trip_ids = top10[top10[road_column] == selected_road]["trip_ids"].iloc[0]
        for trip_id in selected_trip_ids:
            selected_coords.extend(route_point_map.get(trip_id, []))

    # If there are selected coordinates, center map on them
    if selected_coords:
        avg_lat = sum(lat for lat, _ in selected_coords) / len(selected_coords)
        avg_lon = sum(lon for _, lon in selected_coords) / len(selected_coords)
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11)
    else:
        # Calculate the bounding box for all trip points to get a better zoom level
        bounds = gdf_points.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2

        # Define a zoom level based on the bounding box size
        map_width = bounds[2] - bounds[0]  # Longitude range
        map_height = bounds[3] - bounds[1]  # Latitude range
        zoom_level = max(11, int(10 - (max(map_width, map_height) * 5)))

        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level)

    # Find max usage for color scale
    max_usage = max(trip_usage.values()) if trip_usage else 1

    def interpolate_color(count):
        if count == 0:
            return "#555555"  # dark grey
        norm = np.log(count + 1) / np.log(max_usage + 1)
        norm = max(0.1, norm)
        r = int(255 * norm)
        g = int(255 * (1 - norm))
        return f"#{r:02x}{g:02x}00"

    # Draw routes
    for trip_id, coords in route_point_map.items():
        if not coords:
            continue

        count = trip_usage.get(trip_id, 0)
        road_name = trip_to_road.get(trip_id, "Other")
        is_selected = selected_road and road_name == selected_road

        color = "#00FF00" if is_selected else interpolate_color(count)
        weight = 6 if is_selected else (2 + int((count / max_usage) * 3))

        folium.PolyLine(coords, color=color, weight=weight, opacity=0.85).add_to(m)

    return m

def create_heatmap(interp_points, center_latlon):
    center_lat, center_lon = center_latlon
    bounds = np.array([point for point in interp_points])
    min_lat, min_lon = bounds[:, 0].min(), bounds[:, 1].min()
    max_lat, max_lon = bounds[:, 0].max(), bounds[:, 1].max()

    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon
    zoom_level = 11  # Default zoom level for a wide view
    if lat_diff > lon_diff:
        zoom_level = 8  # Zoom out if the latitudes are more spread out

    m = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=zoom_level)
    
    sampled_points = interp_points[::3]
    HeatMap(sampled_points, radius=15, blur=8, min_opacity=0.3, max_zoom=10).add_to(m)
    
    return m
# =================== Streamlit App ===================
uploaded_csv = st.file_uploader("Upload Vehicle Trip CSV", type="csv")

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    required_columns = {"Latitude", "Longitude", "trip_Id", "time_Stamp", "vehicle_Id"}
    if not required_columns.issubset(df.columns):
        st.error(f"CSV is missing required columns: {required_columns}")
        st.stop()

    gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs="EPSG:4326")
    bounds = gdf_points.total_bounds

    try:
        roads_gdf = load_relevant_roads(bounds)
        st.success("Road layer loaded for selected area!")
    except Exception as e:
        st.error(f"Failed to load road data: {e}")
        st.stop()

    trip_lines_gdf, interp_points, route_point_map = process_trips(gdf_points)

    top10, road_column = get_top10_routes(trip_lines_gdf, roads_gdf)

    view_mode = st.radio("Map View", ["Route Map", "Heatmap"], horizontal=True)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Top 10 Most Traveled Roads")

        selected_road = st.selectbox("Select a road to highlight", top10[road_column])
        st.dataframe(top10[[road_column, "count"]], use_container_width=True)

        # CSV Export Button - place here beside route selection
        csv_data = []
        for _, row in top10.iterrows():
            for trip_id in row["trip_ids"]:
                coords = route_point_map.get(trip_id, [])
                for lat, lon in coords:
                    csv_data.append({
                        "road": row[road_column],
                        "trip_id": trip_id,
                        "lat": lat,
                        "lon": lon
                    })
        export_df = pd.DataFrame(csv_data)
        csv = export_df.to_csv(index=False).encode("utf-8")

        with st.spinner("Preparing download..."):
            st.download_button(
                "Download Top Routes as CSV",
                csv,
                "top_routes_with_coords.csv",
                "text/csv",
                key="download_button_1"
            )

    with col2:
        if view_mode == "Route Map":
            map_obj = create_route_map(gdf_points, route_point_map, top10, selected_road, road_column)
        else:
            map_obj = create_heatmap(interp_points, [gdf_points.geometry.y.mean(), gdf_points.geometry.x.mean()])
        
        st_folium(map_obj, width=900, height=550)