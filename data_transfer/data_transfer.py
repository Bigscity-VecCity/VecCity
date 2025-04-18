import geopandas as gpd
import pandas as pd
import numpy as np
import os


# 从命令行读取参数
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--output_name', type=str, default='test_data', help='output directory')
parser.add_argument('--geo_path', type=str, default='test_data', help='path to geo data')
parser.add_argument('--rel_path', type=str, default='test_data', help='path to rel data')
parser.add_argument('--traj_path', type=str, default='test_data', help='path to traj data')
args = parser.parse_args()


# Create output directory
output_dir = args.output_name
os.makedirs(output_dir, exist_ok=True)

# Load GeoPackage (GPKG) File
gpkg_file = args.geo_path
geo_gdf = gpd.read_file(gpkg_file)

# Convert to .geo format
geo_data = []
poi_id_counter = 0
road_id_counter = 0
region_id_counter = 0
for _, row in geo_gdf.iterrows():
    entity = {
        "geo_id": row.get("geo_uid", ""),
        "type": row.get("type", ""),
        "coordinates": row.geometry.wkt,
        "poi_id": poi_id_counter if row.get("type") == "point" else "",
        "poi_type": row.get("entity_type", "") if row.get("type") == "point" else "",
        "road_id": road_id_counter if row.get("type") == "polyline" else "",
        "road_type": row.get("entity_type", "") if row.get("type") == "polyline" else "",
        "region_id": region_id_counter if row.get("type") == "polygon" else "",
        "region_type": row.get("entity_type", "") if row.get("type") == "polygon" else "",
    }
    # 其他的属性直接复制
    for key in row.keys():
        if key not in entity:
            entity[key] = row.get(key, "")
    if row.get("type") == "point":
        poi_id_counter += 1
    elif row.get("type") == "polyline":
        road_id_counter += 1
    elif row.get("type") == "polygon":
        region_id_counter += 1
    geo_data.append(entity)

# Convert to DataFrame and save as CSV file
geo_df = pd.DataFrame(geo_data)
geo_df.to_csv(os.path.join(output_dir, "{}.geo").format(output_dir), index=False)

print(f"Converted GeoPackage data to .geo format in {output_dir}/{output_dir}.geo")

# Load Relationship Data (CSV)
rel_csv_path = args.rel_path
rel_df = pd.read_csv(rel_csv_path)

# Determine Relationship Type Based on Entity Type
def get_entity_type(geo_id):
    row = geo_df[geo_df["geo_id"] == geo_id]
    if row.empty:
        return None
    entity_type = row.iloc[0]["type"]
    if entity_type == "point":
        return "POI"
    elif entity_type == "polyline":
        return "Road"
    elif entity_type == "polygon":
        return "Region"
    return None

# Generate .grel file (Geospatial Relationships)
grel_data = []
srel_data = []
grel_id_counter = 0
srel_id_counter = 0
for _, rel in rel_df.iterrows():
    orig_type = get_entity_type(rel["orig_geo_id"])
    dest_type = get_entity_type(rel["dest_geo_id"])
    if orig_type and dest_type:
        rel_type = f"{orig_type}2{dest_type}"
        if rel["rel_type"] == "geospatial relationship":
            grel_data.append({
                "rel_id": grel_id_counter,
                "type": "GR",
                "orig_geo_id": rel["orig_geo_id"],
                "dest_geo_id": rel["dest_geo_id"],
                "rel_type": rel_type,
                "weight": rel["weight"]
            })
            # 其他的属性直接复制
            for key in rel.keys():
                if key not in grel_data[-1]:
                    grel_data[-1][key] = rel.get(key, "")
            grel_id_counter += 1
        elif rel["rel_type"] == "social relationship":
            srel_data.append({
                "rel_id": srel_id_counter,
                "type": "SR",
                "start_time": rel["start_time"],  
                "end_time": rel["end_time"],  
                "orig_geo_id": rel["orig_geo_id"],
                "dest_geo_id": rel["dest_geo_id"],
                "flow": rel["weight"] # Random weight value
            })
            # 其他的属性直接复制
            for key in rel.keys():
                if key not in srel_data[-1]:
                    srel_data[-1][key] = rel.get(key, "")
            srel_id_counter += 1


# Convert to DataFrame and save as .grel file
grel_df = pd.DataFrame(grel_data)
grel_df.to_csv(os.path.join(output_dir, f"{output_dir}.grel"), index=False)

# Convert to DataFrame and save as .srel file
srel_df = pd.DataFrame(srel_data)
srel_df.to_csv(os.path.join(output_dir, f"{output_dir}.srel"), index=False)

print(f"Generated geospatial relationship file in {output_dir}/{output_dir}.grel")
print(f"Generated social relationship file in {output_dir}/{output_dir}.srel")

# Load Trajectory Data (CSV)
traj_csv_path = args.traj_path
traj_df = pd.read_csv(traj_csv_path)

# Generate .cdtraj file (Coordinates Trajectory)
cdtraj_data = []
traj_id_counter = 0
for _, traj in traj_df.iterrows():
    if traj["traj_type"] == "Coordinates":
        cdtraj_data.append({
            "dyna_id": f"{traj_id_counter}",
            "type": "cdt",
            "time": traj["timestamp"],
            "geo_id": traj["geo_id"],
            "traj_id": traj["traj_id"],
            "usr_id": traj["usr_id"]
        })
        #其他属性直接复制
        for key in traj.keys():
            if key not in cdtraj_data[-1]:
                cdtraj_data[-1][key] = traj.get(key, "")
        traj_id_counter += 1

# Convert to DataFrame and save as .cdtraj file
cdtraj_df = pd.DataFrame(cdtraj_data)
cdtraj_data = sorted(cdtraj_data, key=lambda x: (x["traj_id"], x["time"]))
cdtraj_df.to_csv(os.path.join(output_dir, f"{output_dir}.cdtraj"), index=False)

print(f"Generated coordinates trajectory file in {output_dir}/geo_data.cdtraj")

# # Load Trajectory Data (CSV)
# traj_csv_path = os.path.join(output_dir, "traj_data.csv")
# traj_df = pd.read_csv(traj_csv_path)

# Generate .cdtraj file (Coordinates Trajectory)
citraj_data = []
traj_id_counter = 0
for _, traj in traj_df.iterrows():
    if traj["traj_type"] == "Check-in":
        citraj_data.append({
            "dyna_id": f"{traj_id_counter}",
            "type": "cit",
            "time": traj["timestamp"],
            "geo_id": traj["geo_id"],
            "traj_id": traj["traj_id"],
            "usr_id": traj["usr_id"]
        })
        #其他属性直接复制
        for key in traj.keys():
            if key not in citraj_data[-1]:
                citraj_data[-1][key] = traj.get(key, "")
        traj_id_counter += 1

# traj_id 相同的按照time进行排序
citraj_data = sorted(citraj_data, key=lambda x: (x["traj_id"], x["time"]))
# Convert to DataFrame and save as .cdtraj file
citraj_df = pd.DataFrame(citraj_data)
citraj_df.to_csv(os.path.join(output_dir, f"{output_dir}.citraj"), index=False)

print(f"Generated check-in trajectory file in {output_dir}/{output_dir}.citraj")
