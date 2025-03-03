# VecCity Standard Input Support

VecCity now supports **standard input formats**, enabling seamless integration with structured geospatial and trajectory data. Running the provided script will automatically convert standard format files into atomic files for further processing.

To ensure smooth execution, please follow the instructions below to properly format your original files before running the script.

## 1. Formatting Map Entity Files

Map entity data should be stored in **GeoPackage (GPKG) format** and must include the following required attributes:

- `geo_uid`: Unique ID for each geographical entity
- `type`: Geographical category (`point`, `polyline`, `polygon`)
- `entity_type`: Specifies the category of map entity
- `geometry`: Stores the coordinate information of the entity

All attributes listed above are **mandatory** and must be included in the SHP file.

## 2. Formatting Relationship Files

Relationships between geographical entities should be stored in a **CSV file**, where each row represents a relationship. The file must include the following required fields:

- `orig_geo_id`: ID of the originating geographical entity (must match `geo_uid` in the entity file)
- `dest_geo_id`: ID of the destination geographical entity (must match `geo_uid` in the entity file)
- `rel_type`: Specifies the relationship type (`geospatial relationship` or `social relationship`)
- `weight`: Defines the weight of the relationship (use `1` as the default for unweighted relationships)
- `start_time`: Specific for social relationship, indicating the start time of this record. (standard Unix timestamp since 1970)
- `end_time`: Specific for social relationship, indicating the end time of this record (standard Unix timestamp since 1970)

## 3. Formatting Trajectory Files

Trajectory data should be stored in a **CSV file**, where each row represents a trajectory point. The file must include the following required fields:

- `traj_id`: Unique ID of the trajectory (type: `int`)
- `traj_type`: Type of trajectory point (`Check-in` or `Coordinates`)
- `geo_id`: ID of the geographical entity associated with the trajectory point
- `usr_id`: User ID associated with the trajectory point (default value: `0`)
- `timestamp`: Timestamp of the trajectory record (standard Unix timestamp since 1970)

By ensuring your data conforms to these specifications, VecCity can efficiently process and analyze the information. If you encounter any issues, please refer to the documentation or reach out to the development team for support.

## 4. Running the File Conversion

Once your data is formatted correctly, you can convert the files using the following command:

```bash
python data_transfer.py --output_name [xxx]\
--geo_path [path_to_geo_shp]\
--rel_path [path_to_rel_csv]\
--traj_path [path_to_traj_csv]
```

Replace `[xxx]` with the appropriate file name to execute the conversion process. 

After running the command, a new folder named `[xxx]` will be created, containing the atomic files. Move this folder to `../raw_data` using the following command:

```bash 
mv -r [xxx] ../raw_data
```
You can then modify the dataset parameter in your program to [xxx] and execute the analysis on this dataset.

By ensuring your data conforms to these specifications, VecCity can efficiently process and analyze the information. If you encounter any issues, please refer to the documentation or reach out to the development team for support.