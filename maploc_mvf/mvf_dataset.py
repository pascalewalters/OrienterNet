import json
import torch
from torch.utils.data import Dataset
from typing import Any, Dict
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import pickle
from pyproj import Proj, Transformer
import math
import cv2
import matplotlib.pyplot as plt

from maploc.utils.io import read_image
from maploc.utils.wrappers import Camera
from maploc.data.utils import random_flip, random_rot90


def process_image(image, resize_image=None):
    """Process image for training"""
    # Convert to tensor and normalize
    image = (
        torch.from_numpy(np.ascontiguousarray(image))
        .permute(2, 0, 1)  # CHW format
        .float()
        .div_(255)
    )

    # Create valid mask (all pixels are valid)
    valid = torch.ones_like(image[0], dtype=torch.bool)

    # Resize if needed
    if resize_image is not None:
        # First resize maintaining aspect ratio
        h, w = image.shape[-2:]
        scale = resize_image / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize keeping aspect ratio
        image = F.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        valid = (
            F.interpolate(
                valid.unsqueeze(0).unsqueeze(0).float(),
                size=(new_h, new_w),
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
            .bool()
        )

        # Create padded tensors
        padded_image = torch.zeros((3, resize_image, resize_image), dtype=image.dtype)
        padded_valid = torch.zeros((resize_image, resize_image), dtype=valid.dtype)

        pad_h = (resize_image - new_h) // 2
        pad_w = (resize_image - new_w) // 2

        padded_image[:, pad_h : pad_h + new_h, pad_w : pad_w + new_w] = image
        padded_valid[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = valid

        image = padded_image
        valid = padded_valid

    return image, valid


def create_local_mercator_transform(ref_lat, ref_lon):
    """Create a transformer for local Mercator projection"""
    # WGS84 projection
    wgs84 = Proj("epsg:4326")

    # Custom local Mercator centered at reference point
    local_mercator = Proj(
        f"+proj=tmerc +lat_0={ref_lat} +lon_0={ref_lon} "
        "+k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    )

    # Create transformer with always_xy=True to ensure consistent coordinate order
    return Transformer.from_proj(wgs84, local_mercator, always_xy=True)


class YYCDatasetMVF(Dataset):
    """
    Dataset class for the YYC dataset
    """

    def __init__(self, cfg: Dict[str, Any], stage: str):
        super().__init__()

        self.cfg = cfg
        self.stage = stage

        combined_geojson_path = Path(self.cfg.data.paths.combined_geojson_path)
        with open(combined_geojson_path, "r") as f:
            geojson_data = json.load(f)

        splits_path = Path(self.cfg.data.paths.split_file)
        with open(splits_path, "r") as f:
            splits_data = json.load(f)

        if self.stage in ["train", "val"]:
            split = splits_data[self.stage]
            self.map_ids = list(split.keys())

            self.image_names = []
            for map_id in self.map_ids:
                self.image_names.extend(split[map_id])

            # Build the ground truth data dictionary to be accessed at train time
            self.gt_data_dict = {}
            for feature in geojson_data["features"]:
                if feature["properties"]["map"] in self.map_ids:
                    image_name = "".join(
                        feature["properties"]["imageUrl"].split(".")[:-1]
                    )
                    self.gt_data_dict[image_name] = {
                        "image_url": feature["properties"]["imageUrl"],
                        "map": feature["properties"]["map"],
                        "bearing": feature["properties"]["bearing"],
                        "latitude": feature["geometry"]["coordinates"][1],
                        "longitude": feature["geometry"]["coordinates"][0],
                        "altitude": feature["geometry"]["coordinates"][2],
                    }
        elif self.stage == "test":
            dont_use_map_ids = []
            dont_use_image_names = []
            for s in ["train", "val"]:
                split = splits_data[s]
                dont_use_map_ids.extend(list(split.keys()))
                for map_id in split.keys():
                    dont_use_image_names.extend(split[map_id])

            self.gt_data_dict = {}
            self.image_names = []
            self.map_ids = []
            for feature in geojson_data["features"]:
                image_name = "".join(feature["properties"]["imageUrl"].split(".")[:-1])
                if image_name not in dont_use_image_names:
                    self.gt_data_dict[image_name] = {
                        "image_url": feature["properties"]["imageUrl"],
                        "map": feature["properties"]["map"],
                        "bearing": feature["properties"]["bearing"],
                        "latitude": feature["geometry"]["coordinates"][1],
                        "longitude": feature["geometry"]["coordinates"][0],
                        "altitude": feature["geometry"]["coordinates"][2],
                    }
                    self.image_names.append(image_name)
                    self.map_ids.append(feature["properties"]["map"])

        else:
            raise NotImplementedError(f"Split is not recognized: {self.stage}")

        self.raster_maps = {}
        for map_id in self.map_ids:
            self.raster_maps[map_id] = Path(
                self.cfg.data.paths.raster_map_path, f"raster_map_{map_id}.npy"
            )

        with open(self.cfg.data.paths.map_data_path, "rb") as f:
            self.map_dict = pickle.load(f)

        # The reference point for the ENU transform is the centroid of
        # all bounding shapes across all maps
        centroids = []
        for map_id, map_data in self.map_dict.items():
            for shape in map_data["bounding_shapes"]:
                centroids.append([shape.centroid.x, shape.centroid.y])

        # Debug the reference point calculation
        centroids = np.array(centroids)
        ref_lon, ref_lat = centroids.mean(axis=0)

        # Create transformer for local Mercator projection
        self.transformer = create_local_mercator_transform(ref_lat, ref_lon)

        # Load transforms
        tfs = []
        if stage == "train" and cfg.data.augmentation.image.apply:
            args = OmegaConf.masked_copy(
                cfg.data.augmentation.image,
                ["brightness", "contrast", "saturation", "hue"],
            )
            tfs.append(transforms.ColorJitter(**args))
        self.tfs = transforms.Compose(tfs)

    def visualize_sample(self, data):
        """
        Visualize the dataset sample including raster, image, UV point and map mask
        """

        # Convert image from tensor [C,H,W] back to numpy [H,W,C]
        image = data["image"].permute(1, 2, 0).numpy()

        # Convert raster from tensor [C,H,W] to numpy [H,W,C]
        raster = data["map"].permute(1, 2, 0).numpy()

        # Create colored visualization where each channel gets its own color
        colors = [
            [1, 0, 0],  # Red for channel 0
            [0, 1, 0],  # Green for channel 1
            [0, 0, 1],  # Blue for channel 2
        ]

        # Create RGB visualization
        colored_raster = np.zeros((raster.shape[0], raster.shape[1], 3))
        for i in range(raster.shape[-1]):  # For each channel
            mask = raster[:, :, i] > 0
            colored_raster[mask] = colors[i]

        # Clip to [0,1] range
        colored_raster = np.clip(colored_raster, 0, 1)

        # Get UV coordinates
        uv = data["uv"].numpy()

        # Convert map mask from tensor to numpy if it exists
        map_mask = data["map_mask"].numpy() if "map_mask" in data else None

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Plot image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Input Image")

        # Plot colored raster map
        axes[0, 1].imshow(colored_raster)
        axes[0, 1].scatter(uv[0], uv[1], c="white", marker="x", s=100)
        axes[0, 1].set_title("Raster Map with UV Point")

        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=colors[i]) for i in range(len(colors))
        ]
        axes[0, 1].legend(
            legend_elements,
            [f"Channel {i}" for i in range(len(colors))],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )

        if map_mask is not None:
            # Plot map mask
            axes[1, 0].imshow(map_mask, cmap="gray")
            axes[1, 0].set_title("Map Mask")

            # Plot combined visualization
            combined = colored_raster.copy()
            combined[map_mask] = combined[map_mask] * 0.5  # Darken masked areas
            axes[1, 1].imshow(combined)
            axes[1, 1].scatter(uv[0], uv[1], c="white", marker="x", s=100)
            axes[1, 1].set_title("Combined View")

        plt.tight_layout()
        plt.show()

    def transform_coordinates(self, lat, lon):
        """Transform WGS84 coordinates to local Mercator"""
        try:
            # Note: transformer expects (lon, lat) order when always_xy=True
            east, north = self.transformer.transform(lon, lat)

            # Validate output is finite
            if not (math.isfinite(east) and math.isfinite(north)):
                raise ValueError(
                    f"Transform produced non-finite values: east={east}, north={north}"
                )

            return east, north

        except Exception as e:
            print(f"Transform failed for lat={lat}, lon={lon}")
            raise e

    def to_uv(self, xy, xy_min, xy_max, canvas_scaling):
        """
        Convert xy (meters) to uv (canvas coordinates)
        """
        min_ = xy_min
        max_ = xy_max
        if isinstance(xy, torch.Tensor):
            min_ = torch.from_numpy(min_).to(xy)
            max_ = torch.from_numpy(max_).to(xy)

        canvas_coords_0 = (xy[0] - min_[0]) * self.cfg.data.pixel_per_meter
        canvas_coords_1 = (max_[1] - xy[1]) * self.cfg.data.pixel_per_meter

        # return [
        #     int(canvas_coords_0 * self.cfg.data.map_resize_dim / canvas_scaling[0]),
        #     int(canvas_coords_1 * self.cfg.data.map_resize_dim / canvas_scaling[1]),
        # ]
        return [int(canvas_coords_0), int(canvas_coords_1)]

    def create_map_mask(self, bounding_shapes, canvas_scaling, xy_min, xy_max, raster):
        """
        Create a map mask by setting all pixels outside of the bounding shapes to False
        """
        c, h, w = raster.shape
        # map_mask = np.zeros(
        #     (self.cfg.data.map_resize_dim, self.cfg.data.map_resize_dim),
        #     dtype=np.uint8,
        # )
        map_mask = np.zeros(
            (h, w),
            dtype=np.uint8,
        )
        for shape in bounding_shapes:
            canvas_coords = []
            for point in shape.exterior.coords:
                xy_w = np.array(self.transform_coordinates(point[1], point[0]))
                vu = self.to_uv(xy_w, xy_min, xy_max, canvas_scaling)
                canvas_coords.append(vu)

            canvas_coords = np.array(canvas_coords).astype(np.int32)
            map_mask = cv2.fillPoly(map_mask, [canvas_coords], 255)

        # map_mask = np.flipud(map_mask)

        return cv2.threshold(map_mask, 127, 1, cv2.THRESH_BINARY)[1].astype(bool)

    def random_crop_around_point(self, raster, uv_point, map_mask, crop_size):
        """
        Randomly crop raster centered around UV point
        Args:
            raster: [C,H,W] raster map
            uv_point: [2] UV coordinates
            map_mask: [H,W] boolean mask
            crop_size: Integer size of square crop
        Returns:
            cropped_raster: [C,crop_size,crop_size]
            cropped_mask: [crop_size,crop_size]
            new_uv: [2] Updated UV coordinates in cropped space
        """
        c, h, w = raster.shape

        # Ensure UV point is within crop bounds
        half_size = crop_size // 2

        # Ensure center point stays within valid crop bounds
        center_x = np.clip(int(uv_point[0]), half_size, w - half_size)
        center_y = np.clip(int(uv_point[1]), half_size, h - half_size)

        # Calculate valid range for random offset
        max_offset = half_size
        offset_x = np.random.randint(-max_offset, max_offset)
        offset_y = np.random.randint(-max_offset, max_offset)

        # Apply offset while ensuring crop stays within image bounds
        center_x = np.clip(center_x + offset_x, half_size, w - half_size)
        center_y = np.clip(center_y + offset_y, half_size, h - half_size)

        # Calculate crop coordinates
        x1 = center_x - half_size
        x2 = center_x + half_size
        y1 = center_y - half_size
        y2 = center_y + half_size

        # Debug checks
        assert x1 >= 0 and y1 >= 0, f"Negative coordinates: x1={x1}, y1={y1}"
        assert x2 <= w and y2 <= h, f"Coordinates exceed bounds: x2={x2}/{w}, y2={y2}/{h}"

        # Crop raster and mask
        cropped_raster = raster[:, y1:y2, x1:x2]
        cropped_mask = map_mask[y1:y2, x1:x2] if map_mask is not None else None

        # Update UV coordinates relative to crop
        new_uv = np.array([uv_point[0] - x1, uv_point[1] - y1])

        return cropped_raster, cropped_mask, new_uv

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.stage == "train" and self.cfg.data.random:
            seed = None
        else:
            seed = [self.cfg.data.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)

        image_name = self.image_names[idx]
        image_data = self.gt_data_dict[image_name]
        image_path = Path(self.cfg.data.paths.photos_dir, image_data["image_url"])
        # Load and process image
        image = read_image(image_path)

        # Simple orientation from bearing
        roll, pitch = 0.0, 0.0  # Assuming flat ground
        yaw = float(image_data["bearing"])  # Use bearing as yaw

        # Create camera parameters for the image
        h, w = image.shape[:2]  # Get image dimensions
        cam_dict = {
            "model": "SIMPLE_RADIAL",
            "width": w,
            "height": h,
            "params": np.array(
                [
                    max(w, h) * 0.93,  # focal length estimate (based on ~66 degree FOV)
                    w / 2,  # cx (principal point x)
                    h / 2,  # cy (principal point y)
                    0.1,  # k1 (radial distortion)
                ]
            ),
        }
        cam = Camera.from_dict(cam_dict).float()
        image, valid = process_image(image, self.cfg.data.resize_image)

        # Get raster map data
        raster = np.load(self.raster_maps[image_data["map"]])
        # raster = cv2.resize(
        #     raster,
        #     (self.cfg.data.map_resize_dim, self.cfg.data.map_resize_dim),
        #     interpolation=cv2.INTER_NEAREST,
        # )
        raster = np.transpose(raster, (2, 0, 1))

        bounds = [
            bounding_shape.bounds
            for bounding_shape in self.map_dict[image_data["map"]]["bounding_shapes"]
        ]
        bounds = np.array(bounds)
        lon_min, lat_min = min(bounds[:, 0]), min(bounds[:, 1])
        lon_max, lat_max = max(bounds[:, 2]), max(bounds[:, 3])

        east_min, north_min = self.transform_coordinates(lat_min, lon_min)
        east_max, north_max = self.transform_coordinates(lat_max, lon_max)

        bounding_box_size = np.array([east_max - east_min, north_max - north_min])
        canvas_scaling = np.ceil(
            bounding_box_size * self.cfg.data.pixel_per_meter
        ).astype(int)

        # Get ground truth position from lat/long
        latlon_gt = torch.tensor(
            [image_data["latitude"], image_data["longitude"]]
        ).numpy()

        # Coordinates in meters from the centroid of the whole airport
        xy_w_gt = np.array(self.transform_coordinates(*latlon_gt))

        # Get coordinates in the canvas
        uv_gt = np.array(
            self.to_uv(
                xy_w_gt,
                np.array([east_min, north_min]),
                np.array([east_max, north_max]),
                canvas_scaling,
            )
        )

        map_center_xy = [
            east_min + (east_max - east_min) / 2,
            north_min + (north_max - north_min) / 2,
        ]
        uv_init = np.array(
            self.to_uv(
                np.array(map_center_xy),
                np.array([east_min, north_min]),
                np.array([east_max, north_max]),
                canvas_scaling,
            )
        )

        # Map augmentations for training
        heading = np.deg2rad(90 - yaw)

        # Optional: create mask for search area
        if self.cfg.data.add_map_mask:
            map_mask = self.create_map_mask(
                self.map_dict[image_data["map"]]["bounding_shapes"],
                canvas_scaling,
                np.array([east_min, north_min]),
                np.array([east_max, north_max]),
                raster
            )
        else:
            c, h, w = raster.shape
            # map_mask = np.ones(
            #     (self.cfg.data.map_resize_dim, self.cfg.data.map_resize_dim),
            #     dtype=torch.bool,
            # )
            map_mask = np.ones(
                (h, w),
                dtype=bool,
            )

        # Crop the map randomly such that the UV point is within the cropped area
        raster, map_mask, uv_gt = self.random_crop_around_point(
            raster, uv_gt, map_mask, self.cfg.data.map_resize_dim
        )

        # Apply augmentations
        if self.stage == "train":
            if self.cfg.data.augmentation.rot90:
                raster, uv_gt, heading, map_mask = random_rot90(
                    raster, uv_gt, heading, map_mask, seed
                )
            if self.cfg.data.augmentation.flip:
                image, raster, uv_gt, heading, map_mask = random_flip(
                    image, raster, uv_gt, heading, map_mask, seed
                )
        # This is taken from the orienternet code. Not sure if this is true for the YYC dataset
        yaw = 90 - np.rad2deg(heading)

        # The dictionary to be returned
        data = {
            "index": idx,
            "name": image_name,
            "scene": image_data["map"],
        }

        if self.cfg.data.add_map_mask:
            data["map_mask"] = torch.from_numpy(map_mask.copy())

        map_copy = np.ascontiguousarray(raster).copy()

        data = {
            **data,
            "image": image,
            "valid": valid,
            "camera": cam,
            "map": torch.from_numpy(map_copy).long(),
            "uv": torch.from_numpy(uv_gt.copy()).float(),
            "uv_init": torch.from_numpy(uv_init.copy()).float(),
            "roll_pitch_yaw": torch.tensor((roll, pitch, yaw)).float(),
            "pixels_per_meter": torch.tensor(self.cfg.data.pixel_per_meter).float(),
        }

        # Visualize the sample (remove this after debugging)
        # self.visualize_sample(data)

        return data
