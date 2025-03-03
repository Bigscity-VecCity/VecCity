import geopandas as gpd
import pandas as pd
import numpy as np
import os
from shapely.geometry import Point, LineString, Polygon

# Create output directory
output_dir = "test_data"
os.makedirs(output_dir, exist_ok=True)

# 1. Generate Map Entity Data (Shapefile)
geo_data = []

# Generate POI Data
for i in range(10):
    geo_data.append({
        "geo_uid": f"poi_{i}",
        "type": "point",
        "entity_type": "POI",
        "poi_type": f"type_{np.random.randint(1, 5)}",
        "geometry": Point(np.random.uniform(-180, 180), np.random.uniform(-90, 90))
    })

# Generate Road Data
for i in range(5):
    geo_data.append({
        "geo_uid": f"road_{i}",
        "type": "polyline",
        "entity_type": "Road",
        "road_highway": np.random.choice([True, False]),
        "road_lanes": np.random.randint(1, 5),
        "road_max_speed": np.random.randint(30, 120),
        "road_bridge": np.random.choice([True, False]),
        "road_roundabout": np.random.choice([True, False]),
        "road_oneway": np.random.choice([True, False]),
        "road_tunnel": np.random.choice([True, False]),
        "road_type": f"road_type_{np.random.randint(1, 5)}",
        "geometry": LineString([(np.random.uniform(-180, 180), np.random.uniform(-90, 90)) for _ in range(3)])
    })

# Generate Parcel Data
for i in range(3):
    geo_data.append({
        "geo_uid": f"parcel_{i}",
        "type": "polygon",
        "entity_type": "Parcel",
        "parcel_type": f"parcel_type_{np.random.randint(1, 5)}",
        "geometry": Polygon([(np.random.uniform(-180, 180), np.random.uniform(-90, 90)) for _ in range(4)])
    })

# Convert to GeoDataFrame and save as SHP file
geo_gdf = gpd.GeoDataFrame(geo_data)
geo_gdf.to_file(os.path.join(output_dir, "geo_data.gpkg"), driver="GPKG")

# 2. Generate Relationship Data (CSV)
rel_data = []
geo_ids = [entry["geo_uid"] for entry in geo_data]

for _ in range(10):
    orig, dest = np.random.choice(geo_ids, 2, replace=False)
    rel_data.append({
        "orig_geo_id": orig,
        "dest_geo_id": dest,
        "rel_type": np.random.choice(["geospatial relationship", "social relationship"]),
        "weight": np.random.choice([1, np.random.uniform(0.1, 5)])
    })

rel_df = pd.DataFrame(rel_data)
rel_df.to_csv(os.path.join(output_dir, "rel_data.csv"), index=False)

# 3. Generate Trajectory Data (CSV)
traj_data = []

for i in range(20):
    traj_data.append({
        "traj_id": f"traj_{i}",
        "traj_type": np.random.choice(["Check-in", "Coordinates"]),
        "geo_id": np.random.choice(geo_ids),
        "usr_id": np.random.randint(1, 10),
        "timestamp": np.random.randint(1609459200, 1640995200)  # Random timestamps in 2021
    })

traj_df = pd.DataFrame(traj_data)
traj_df.to_csv(os.path.join(output_dir, "traj_data.csv"), index=False)

print(f"Test dataset generated in {output_dir}/")
