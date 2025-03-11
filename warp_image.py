import matplotlib.pyplot as plt
import json

from maploc.demo import Demo
from maploc.osm.viz import GeoPlotter
from maploc.osm.tiling import TileManager
from maploc.osm.viz import Colormap, plot_nodes
from maploc.utils.viz_2d import plot_images
import numpy as np
from maploc.utils.viz_localization import (
    likelihood_overlay,
    plot_dense_rotations,
    add_circle_inset,
)
from maploc.utils.viz_2d import features_to_RGB
from pathlib import Path
import os

image_path = "/home/kevinmeng/workspace/mappedin/VPS/OrienterNet/datasets/YYC/f_64ff43e17107d168de589c1a/images/b3150ca4570c10ecba3f89dfb672c35e79eb845e.jpg"
prior_address = "2000 Airport Rd NE, Calgary, AB T2E 6W5"
osm_path = Path("/home/kevinmeng/workspace/mappedin/VPS/OrienterNet/yyc_osms/f_64ff43e17107d168de589c1a_osm.json")

# Get ground truth from geojson file
geojson_path = Path("/home/kevinmeng/workspace/mappedin/VPS/OrienterNet/datasets/YYC/combined-output.geojson")
image_filename = os.path.basename(image_path)

# Load and parse the geojson file to find the ground truth
ground_truth = None
with open(geojson_path, 'r') as f:
    data = json.load(f)
    for feature in data['features']:
        if feature['properties']['imageUrl'] == image_filename:
            coords = feature['geometry']['coordinates']
            ground_truth = {
                'lon': coords[0],
                'lat': coords[1],
                'bearing': float(feature['properties']['bearing'])
            }
            break

if ground_truth is None:
    print(f"Warning: Could not find ground truth for image {image_filename}")

# Continue with the rest of the script
demo = Demo(num_rotations=256, device="cpu")

image, camera, gravity, proj, bbox = demo.read_input_image(
    image_path,
    prior_address= [ground_truth['lat'], ground_truth['lon']],# prior_address,
    tile_size_meters=128,  # try 64, 256, etc.
)

tiler = TileManager.from_bbox(proj, bbox + 10, demo.config.data.pixel_per_meter,
                              path=osm_path)

canvas = tiler.query(bbox)

# Run the inference
uv, yaw, prob, neural_map, image_rectified = demo.localize(
    image, camera, canvas, gravity=gravity
)

map_viz = Colormap.apply(canvas.raster)
plot_images([image, map_viz], titles=["input image", "OpenStreetMap raster"])
plot_nodes(1, canvas.raster[1], fontsize=6, size=10) # canvas.raster[2] does not exist after areas were removed

# Visualize the predictions
overlay = likelihood_overlay(prob.numpy().max(-1), map_viz.mean(-1, keepdims=True))
(neural_map_rgb,) = features_to_RGB(neural_map.numpy())
plot_images([overlay, neural_map_rgb], titles=["prediction", "neural map"])
ax = plt.gcf().axes[0]
ax.scatter(*canvas.to_uv(bbox.center), s=5, c="red")
plot_dense_rotations(ax, prob, w=0.005, s=1 / 25)
add_circle_inset(ax, uv)
plt.show()

# Plot as interactive figure
bbox_latlon = proj.unproject(canvas.bbox)
plot = GeoPlotter(zoom=16.5)
plot.raster(map_viz, bbox_latlon, opacity=0.5)
plot.raster(likelihood_overlay(prob.numpy().max(-1)), proj.unproject(bbox))
# plot.points(proj.latlonalt[:2], "red", name="location prior", size=10)
print(f"location prior: {proj.latlonalt[:2]}")
plot.points(proj.unproject(canvas.to_xy(uv)), "black", name="argmax", size=10)

if ground_truth:
    plot.points([ground_truth['lat'], ground_truth['lon']], "blue", name="ground truth", size=10)
    print(f"Ground truth: lat={ground_truth['lat']}, lon={ground_truth['lon']}, bearing={ground_truth['bearing']}")

plot.bbox(bbox_latlon, "blue", name="map tile")
plot.fig.show()

