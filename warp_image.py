import matplotlib.pyplot as plt

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

image_path = "IMG_0937.jpg"
prior_address = "14 Erb St. W, Waterloo, ON N2L 1S7"

demo = Demo(num_rotations=256, device="cpu")

image, camera, gravity, proj, bbox = demo.read_input_image(
    image_path,
    prior_address=prior_address,
    tile_size_meters=128,  # try 64, 256, etc.
)

tiler = TileManager.from_bbox(proj, bbox + 10, demo.config.data.pixel_per_meter)
canvas = tiler.query(bbox)

# Run the inference
uv, yaw, prob, neural_map, image_rectified = demo.localize(
    image, camera, canvas, gravity=gravity
)

map_viz = Colormap.apply(canvas.raster)
plot_images([image, map_viz], titles=["input image", "OpenStreetMap raster"])
plot_nodes(1, canvas.raster[2], fontsize=6, size=10)

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
plot.points(proj.latlonalt[:2], "red", name="location prior", size=10)
plot.points(proj.unproject(canvas.to_xy(uv)), "black", name="argmax", size=10)
plot.bbox(bbox_latlon, "blue", name="map tile")
plot.fig.show()

