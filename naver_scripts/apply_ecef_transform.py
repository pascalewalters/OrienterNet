import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import json
from pyproj import Transformer
import math
import geojson


def get_data(data_dir, ecef_transform_file, map_id):

    # Load the data from the specified directory
    # This function should be implemented to load your specific data format
    trajectory_file = os.path.join(data_dir, "trajectories.txt")

    trajectories_df = pd.read_csv(
        trajectory_file, sep=", ", header=1, skiprows=0, engine="python"
    )

    records_camera_file = os.path.join(data_dir, "records_camera.txt")
    records_df = pd.read_csv(
        records_camera_file, sep=", ", header=1, skiprows=0, engine="python"
    )

    with open(ecef_transform_file, "r") as f:
        ecef_transform_dict = json.load(f)
    # Get the transform parameterized by the quaternion from the ECEF tool
    transform_rotation_ecef = Rotation.from_quat(
        [
            ecef_transform_dict["qx"],
            ecef_transform_dict["qy"],
            ecef_transform_dict["qz"],
            ecef_transform_dict["qw"],
        ]
    )
    preprocess_rotation = Rotation.from_matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    transform_rotation = transform_rotation_ecef * preprocess_rotation
    transform_rotation_matrix = transform_rotation.as_matrix()

    ecef_transform = np.identity(4)
    ecef_transform[:3, :3] = transform_rotation_matrix
    ecef_transform[0, 3] = ecef_transform_dict["tx"]
    ecef_transform[1, 3] = ecef_transform_dict["ty"]
    ecef_transform[2, 3] = ecef_transform_dict["tz"]

    features = []
    split_image_names = []
    for _, record_row in records_df.iterrows():
        # Get the corresponding row in the trajectories DataFrame
        trajectory_row = trajectories_df[
            (trajectories_df["# timestamp"] == record_row["# timestamp"])
            & (trajectories_df["device_id"] == record_row["device_id"])
        ].iloc[0]
        rotation = Rotation.from_quat(
            [
                trajectory_row["qx"],
                trajectory_row["qy"],
                trajectory_row["qz"],
                trajectory_row["qw"],
            ]
        )
        rotation_matrix = rotation.as_matrix()
        translation = np.array(
            [[trajectory_row["tx"], trajectory_row["ty"], trajectory_row["tz"]]]
        ).T
        camera_position = -np.dot(rotation_matrix.T, translation)

        sensor_point = np.identity(4)
        sensor_point[:3, :3] = rotation_matrix
        sensor_point[0, 3] = camera_position[0][0]
        sensor_point[1, 3] = camera_position[1][0]
        sensor_point[2, 3] = camera_position[2][0]

        # Apply the ECEF transform to the sensor point
        ecef_pose = ecef_transform @ sensor_point
        # Transform the ECEF point to a lat, long, altitude
        ecef_position = transformer.transform(
            ecef_pose[0, 3], ecef_pose[1, 3], ecef_pose[2, 3]
        )

        lat_radians = math.radians(ecef_position[1])
        lon_radians = math.radians(ecef_position[0])

        east = [-math.sin(lon_radians), math.cos(lon_radians), 0]
        north = [
            -math.sin(lat_radians) * math.cos(lon_radians),
            -math.sin(lat_radians) * math.sin(lon_radians),
            math.cos(lat_radians),
        ]
        up = [
            math.cos(lat_radians) * math.cos(lon_radians),
            math.cos(lat_radians) * math.sin(lon_radians),
            math.sin(lat_radians),
        ]
        enu = np.array([east, north, up])  # 3x3

        # Apply the transform to the rotation of the sensor point
        r = enu.T @ ecef_pose[:3, :3]
        r_rotation = Rotation.from_matrix(r)
        # Get the Euler angles of the rotation
        yaw, pitch, roll = r_rotation.as_euler("yzx", degrees=True)

        feature = geojson.Feature(
            geometry=geojson.Point(ecef_position, precision=16),
            properties={
                "map": map_id,
                "sourceFile": record_row["image_path"],
                "imageUrl": os.path.basename(record_row["image_path"]),
                # Bearing is degrees from north, clockwise
                "bearing": (360 - yaw) % 360,
                "pitch": pitch,
                "roll": roll,
            },
        )
        features.append(feature)
        image_name = os.path.basename(record_row["image_path"]).split(".")[0]
        split_image_names.append(image_name)

    return features, split_image_names


# MAP_ID = "m_37272a41df19b0eb" # Coex 1F
MAP_ID = "m_ec13525d241ff4da"  # Gangnam B2

# Define a Pyproj Transformer object to go from geocentric (ECEF) to lat, long
transformer = Transformer.from_crs(
    {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
    {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
)

features = []
splits = {"train": {}, "val": {}, "test": {}}

train_data_dir = "/home/pascale/Documents/VPS/WayIL/Naver/gangnam/GangnamStation/B2/release/mapping/sensors"
train_features, train_image_names = get_data(
    train_data_dir,
    "/home/pascale/Documents/VPS/WayIL/Naver/gangnam/alignments/B2_mapping_alignment.json",
    MAP_ID,
)
features.extend(train_features)
splits["train"][MAP_ID] = train_image_names

val_data_dir = "/home/pascale/Documents/VPS/WayIL/Naver/gangnam/GangnamStation/B2/release/validation/sensors"
val_features, val_image_names = get_data(
    val_data_dir,
    "/home/pascale/Documents/VPS/WayIL/Naver/gangnam/alignments/B2_validation_alignment.json",
    MAP_ID,
)
features.extend(val_features)
splits["val"][MAP_ID] = val_image_names

# test_data_dir = "/home/pascale/Documents/VPS/WayIL/kapture-localization/pipeline/examples/coex_1F/coex_1F_release_test/1F/release/test/sensors/"
# test_features, test_image_names = get_data(
#     test_data_dir, "test_calibration.json", MAP_ID
# )
# features.extend(test_features)
# splits["test"][MAP_ID] = test_image_names
splits["test"][MAP_ID] = []

# Create a GeoJSON FeatureCollection
feature_collection = geojson.FeatureCollection(features)

# Write the FeatureCollection to a GeoJSON file
with open("gangnam-B2-combined-output.geojson", "w") as f:
    geojson.dump(feature_collection, f)

# # Write the splits to a JSON file
# with open("hyundai_1F_image_splits.json", "w") as f:
#     json.dump(splits, f, indent=4)
