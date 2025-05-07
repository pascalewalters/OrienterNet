import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement


def save_camera_poses_to_ply(xs, ys, zs, filename="camera_trajectory.ply"):
    """
    Save camera positions to PLY file
    Args:
        xs, ys, zs: Lists of camera positions
        filename: Output PLY filename
    """
    # Combine coordinates into vertex array
    vertices = np.array(
        list(zip(xs, ys, zs)), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )

    # Create edges between consecutive camera positions
    edges = []
    for i in range(len(vertices) - 1):
        edges.append((i, i + 1))
    edges = np.array(edges, dtype=[("vertex1", "i4"), ("vertex2", "i4")])

    # Create PLY elements
    vertex_element = PlyElement.describe(vertices, "vertex")
    edge_element = PlyElement.describe(edges, "edge")

    # Create and save PLY file
    PlyData([vertex_element, edge_element], text=True).write(filename)


DATA_DIR = "/home/pascale/Documents/VPS/WayIL/Naver/gangnam/GangnamStation/B2/release/mapping/sensors/"
OUTPUT_PLY = "/home/pascale/Documents/VPS/WayIL/Naver/gangnam/gangnam_B2_camera_trajectory_mapping.ply"
trajectory_file = os.path.join(DATA_DIR, "trajectories.txt")

trajectories_df = pd.read_csv(
    trajectory_file, sep=", ", header=1, skiprows=0, engine="python"
)

records_camera_file = os.path.join(DATA_DIR, "records_camera.txt")
records_df = pd.read_csv(
    records_camera_file, sep=", ", header=1, skiprows=0, engine="python"
)

# # Get unique device IDs
# device_ids = trajectories_df['device_id'].unique()
# print(f"Available device IDs: {device_ids}")

# # Select a device ID to plot
# device_id = device_ids[1]  # Change index to plot different device

# # Filter data for selected device
# device_data = trajectories_df[trajectories_df['device_id'] == device_id]

# # Sort by timestamp
# device_data = device_data.sort_values('# timestamp')

xs = []
ys = []
zs = []

for idx, row in trajectories_df.iterrows():
    rotation = Rotation.from_quat([row["qx"], row["qy"], row["qz"], row["qw"]])
    rotation_matrix = rotation.as_matrix()
    translation = np.array([[row["tx"], row["ty"], row["tz"]]]).T
    camera_position = -np.dot(rotation_matrix.T, translation)
    xs.append(-camera_position[0][0])
    ys.append(camera_position[1][0])
    zs.append(camera_position[2][0])

plt.plot(xs, ys, "k--", alpha=0.5)
plt.show()

# After collecting camera positions
save_camera_poses_to_ply(xs, zs, ys, OUTPUT_PLY)
