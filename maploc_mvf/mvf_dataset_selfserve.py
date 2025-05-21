import json
import torch
from torch.utils.data import Dataset
from typing import Any, Dict
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import dill as pickle
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


def process_valid_mask(valid_img, resize_image=None):
    """Process valid mask for training"""
    # Convert to grayscale if needed and threshold to binary
    if len(valid_img.shape) == 3:
        valid_img = cv2.cvtColor(valid_img, cv2.COLOR_BGR2GRAY)

    # Convert to boolean mask
    valid = (valid_img > 127).astype(np.uint8)

    # Convert to tensor
    valid = torch.from_numpy(valid).bool()

    # Resize if needed
    if resize_image is not None:
        # Calculate new dimensions maintaining aspect ratio
        h, w = valid.shape
        scale = resize_image / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize using nearest neighbor interpolation
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

        # Create padded tensor
        padded_valid = torch.zeros((resize_image, resize_image), dtype=torch.bool)

        # Calculate padding
        pad_h = (resize_image - new_h) // 2
        pad_w = (resize_image - new_w) // 2

        # Apply padding
        padded_valid[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = valid
        valid = padded_valid

    return valid


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


class NaverDatasetMVF(Dataset):
    """
    Dataset class for the Naver dataset
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
            val_image_names = []
            test_image_names = []
            for map_id in self.map_ids:
                self.image_names.extend(split[map_id])
                val_image_names.extend(splits_data["val"][map_id])
                test_image_names.extend(splits_data["test"][map_id])

            # Build the ground truth data dictionary to be accessed at train time
            self.gt_data_dict = {}
            for feature in geojson_data["features"]:
                # if feature["properties"]["map"] in self.map_ids:
                image_name = "".join(feature["properties"]["imageUrl"].split(".")[:-1])
                if image_name in self.image_names:
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
            self.map_ids = list(set(self.map_ids))

        else:
            raise NotImplementedError(f"Split is not recognized: {self.stage}")

        self.raster_maps = {}
        for map_id in self.map_ids:
            self.raster_maps[map_id] = Path(
                self.cfg.data.paths.raster_map_path, f"raster_map_{map_id}.npy"
            )

        self.mvf_data = {}
        for map_id in self.map_ids:
            mvf_data_path = Path(
                self.cfg.data.paths.mvf_data_path, f"{map_id}_mvf_data.pkl"
            )
            with open(mvf_data_path, "rb") as f:
                mvf_data = pickle.load(f)

            self.mvf_data[map_id] = {
                "spaces": mvf_data[map_id]["spaces"],
                "transformer": create_local_mercator_transform(
                    mvf_data["center_point_lat"], mvf_data["center_point_lon"]
                ),
            }

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
        canvas_scaling = [raster.shape[1], raster.shape[0]]

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
        u = (uv[0] * canvas_scaling[0] / 2) + canvas_scaling[0] / 2
        v = (uv[1] * canvas_scaling[1] / 2) + canvas_scaling[1] / 2

        # Convert map mask from tensor to numpy if it exists
        map_mask = data["map_mask"].numpy() if "map_mask" in data else None

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Plot image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Input Image")

        # Plot colored raster map
        axes[0, 1].imshow(colored_raster)
        axes[0, 1].scatter(u, v, c="white", marker="x", s=100)
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
            axes[1, 1].scatter(u, v, c="white", marker="x", s=100)
            axes[1, 1].set_title("Combined View")

        plt.tight_layout()
        plt.show()

    def transform_coordinates(self, lat, lon, map_id):
        """Transform WGS84 coordinates to local Mercator"""
        try:
            # Note: transformer expects (lon, lat) order when always_xy=True
            east, north = self.mvf_data[map_id]["transformer"].transform(lon, lat)

            # Validate output is finite
            if not (math.isfinite(east) and math.isfinite(north)):
                raise ValueError(
                    f"Transform produced non-finite values: east={east}, north={north}"
                )

            return np.array([east, north], dtype=np.float64)

        except Exception as e:
            print(f"Transform failed for lat={lat}, lon={lon}")
            raise e

    def to_uv(self, xy, xy_min, xy_max, canvas_scaling):
        """
        Convert xy (meters) to uv (canvas coordinates)
        Args:
            xy: position in meters [x, y]
            xy_min: minimum bounds in meters
            xy_max: maximum bounds in meters
            canvas_scaling: image dimensions [width, height]
        Returns:
            uv coordinates centered and normalized [-1, 1]
        """
        min_ = xy_min
        max_ = xy_max
        if isinstance(xy, torch.Tensor):
            min_ = torch.from_numpy(min_).to(xy)
            max_ = torch.from_numpy(max_).to(xy)

        # Convert to pixel coordinates first
        u = (xy[0] - min_[0]) * self.cfg.data.pixel_per_meter
        v = (xy[1] - min_[1]) * self.cfg.data.pixel_per_meter

        u = np.clip(u, 1, canvas_scaling[0] - 1)
        v = np.clip(v, 1, canvas_scaling[1] - 1)

        u, v = self.pixel_to_normalized_uv([u, v], canvas_scaling)

        return [u, v]

    def create_map_mask(self, spaces, canvas_scaling, xy_min, xy_max, raster, map_id):
        """
        Create a map mask by setting all pixels outside of the bounding shapes to False
        """
        _, h, w = raster.shape
        map_mask = np.zeros(
            (h, w),
            dtype=np.uint8,
        )
        for space in spaces:
            canvas_coords = []
            for point in space.shape.polygon.exterior.coords:
                # Convert lat/lon to local meters
                xy_w = np.array(self.transform_coordinates(point[1], point[0], map_id))
                # Convert meters to normalized UV coordinates
                uv = self.to_uv(xy_w, xy_min, xy_max, canvas_scaling)
                # Convert normalized UV to pixel coordinates
                pixel_u, pixel_v = self.normalized_to_pixel_uv(uv, [w, h])
                canvas_coords.append([pixel_u, pixel_v])

            # Convert to numpy array
            canvas_coords = np.array(canvas_coords).astype(np.int32)

            # Fill polygon with ones
            cv2.fillPoly(map_mask, [canvas_coords], 1)

        return map_mask.astype(bool)

    def random_crop_around_point(self, raster, uv_point, map_mask, crop_size):
        """
        Randomly crop raster centered around UV point
        Args:
            raster: [C,H,W] raster map
            uv_point: [2] Normalized UV coordinates in range [-1,1]
            map_mask: [H,W] boolean mask
            crop_size: Integer size of square crop
        Returns:
            cropped_raster: [C,crop_size,crop_size]
            cropped_mask: [crop_size,crop_size]
            new_uv: [2] Updated UV coordinates in cropped space
        """
        _, h, w = raster.shape
        half_size = crop_size // 2
        pixel_u, pixel_v = self.normalized_to_pixel_uv(uv_point, [w, h])

        if w < crop_size:
            new_u = uv_point[0]
            x1 = 0
            x2 = w
        else:
            pixel_u = np.clip(pixel_u, 1, w - 1)
            max_offset_x = min(pixel_u, w - pixel_u, half_size)
            offset_x = np.random.randint(-max_offset_x, max_offset_x)
            center_x = pixel_u + offset_x
            x1 = np.clip(center_x - half_size, 0, w - crop_size)
            x2 = x1 + crop_size

            # Convert pixel coordinates back to normalized UV in cropped space
            new_pixel_u = pixel_u - x1
            # Convert back to normalized coordinates [-1, 1]
            new_u = (new_pixel_u / (crop_size / 2)) - 1

        if h < crop_size:
            new_v = uv_point[1]
            y1 = 0
            y2 = h
        else:
            pixel_v = np.clip(pixel_v, 1, h - 1)
            max_offset_y = min(pixel_v, h - pixel_v, half_size)
            offset_y = np.random.randint(-max_offset_y, max_offset_y)
            center_y = pixel_v + offset_y
            y1 = np.clip(center_y - half_size, 0, h - crop_size)
            y2 = y1 + crop_size

            # Convert pixel coordinates back to normalized UV in cropped space
            new_pixel_v = pixel_v - y1
            # Convert back to normalized coordinates [-1, 1]
            new_v = (new_pixel_v / (crop_size / 2)) - 1

        # Crop raster and mask
        cropped_raster = raster[:, y1:y2, x1:x2]
        cropped_mask = map_mask[y1:y2, x1:x2] if map_mask is not None else None

        return cropped_raster, cropped_mask, np.array([new_u, new_v])

    def normalized_to_pixel_uv(self, uv_normalized, canvas_size):
        """Convert normalized UV coordinates [-1,1] to pixel coordinates"""
        pixel_u = int((uv_normalized[0] + 1) * (canvas_size[0] / 2))
        pixel_v = int((uv_normalized[1] + 1) * (canvas_size[1] / 2))
        return np.array([pixel_u, pixel_v])

    def pixel_to_normalized_uv(self, uv_pixels, canvas_size):
        """Convert pixel coordinates back to normalized UV coordinates [-1,1]"""
        u = (uv_pixels[0] / (canvas_size[0] / 2)) - 1
        v = (uv_pixels[1] / (canvas_size[1] / 2)) - 1
        return np.array([u, v])

    def __len__(self):
        return len(self.image_names)

    def validate_uv_point(self, uv_point, canvas_size, image_name):
        """
        Validate if UV point falls within map bounds
        Args:
            uv_point: [2] Normalized UV coordinates in range [-1,1]
            canvas_size: [width, height] of map
            image_name: Name of image for logging
        Returns:
            bool: True if valid, False if invalid
        """
        pixel_u, pixel_v = self.normalized_to_pixel_uv(uv_point, canvas_size)

        # Check if point is within bounds
        if (
            pixel_u < 0
            or pixel_u >= canvas_size[0]
            or pixel_v < 0
            or pixel_v >= canvas_size[1]
        ):
            print(
                f"Warning: Point {[pixel_u, pixel_v]} outside map bounds {canvas_size} "
                f"for image {image_name}"
            )
            return False
        return True

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
        # Not using the valid image for the Naver dataset. It seems to be a mask for the image
        # valid_path = Path(self.cfg.data.paths.valid_dir, image_data["image_url"])
        # valid = read_image(valid_path)
        valid = np.ones_like(image)

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
        image, _ = process_image(image, self.cfg.data.resize_image)
        # Apply transforms to image
        if self.tfs is not None:
            image = self.tfs(image)
        valid = process_valid_mask(valid, self.cfg.data.resize_image)

        # Get raster map data
        raster = np.load(self.raster_maps[image_data["map"]])

        lat_lon_bounds = [
            space.shape.polygon.exterior.bounds
            for space in self.mvf_data[image_data["map"]]["spaces"]
        ]
        lat_lon_bounds = np.array(lat_lon_bounds, dtype=np.float64)

        # Get ground truth position from lat/long
        latlon_gt = np.array(
            [image_data["latitude"], image_data["longitude"]], dtype=np.float64
        )

        lon_min = min(lat_lon_bounds[:, 0])
        lat_min = min(lat_lon_bounds[:, 1])
        lon_max = max(lat_lon_bounds[:, 2])
        lat_max = max(lat_lon_bounds[:, 3])

        # Transform to local coordinates
        east_min, north_min = self.transform_coordinates(
            lat_min, lon_min, image_data["map"]
        )
        east_max, north_max = self.transform_coordinates(
            lat_max, lon_max, image_data["map"]
        )

        bounding_box_size = np.array([east_max - east_min, north_max - north_min])
        canvas_scaling = np.ceil(
            bounding_box_size * self.cfg.data.pixel_per_meter
        ).astype(int)

        # Coordinates in meters from the centroid of the whole airport
        xy_w_gt = np.array(self.transform_coordinates(*latlon_gt, image_data["map"]))

        # Get coordinates in the canvas
        uv_gt = np.array(
            self.to_uv(
                xy_w_gt,
                np.array([east_min, north_min]),
                np.array([east_max, north_max]),
                canvas_scaling,
            )
        )

        # Validate UV coordinates
        if not self.validate_uv_point(uv_gt, canvas_scaling, image_name):
            # Skip to next valid sample
            return self.__getitem__((idx + 1) % len(self))

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

        # Optional: create mask for search area
        if self.cfg.data.add_map_mask:
            map_mask = self.create_map_mask(
                self.mvf_data[image_data["map"]]["spaces"],
                canvas_scaling,
                np.array([east_min, north_min]),
                np.array([east_max, north_max]),
                raster,
                image_data["map"],
            )
        else:
            map_mask = np.ones(
                (raster.shape[1], raster.shape[2]),
                dtype=bool,
            )

        if self.stage != "test":
            # Crop the map randomly such that the UV point is within the cropped area
            raster, map_mask, uv_gt = self.random_crop_around_point(
                raster, uv_gt, map_mask, self.cfg.data.map_resize_dim
            )

        # Simple orientation from bearing
        roll, pitch = 0.0, 0.0  # Assuming flat ground
        yaw = float(image_data["bearing"])  # Use bearing as yaw

        # Apply augmentations
        if self.stage == "train":
            if self.cfg.data.augmentation.rot90:
                # Convert UV to pixel coordinates
                h, w = raster.shape[-2:]
                uv_pixels = self.normalized_to_pixel_uv(uv_gt, [w, h])
                heading = np.deg2rad(90 - yaw)
                raster, uv_pixels, heading, map_mask = random_rot90(
                    raster, uv_pixels, heading, map_mask, seed
                )
                # Convert back to normalized coordinates
                uv_gt = self.pixel_to_normalized_uv(
                    uv_pixels, [raster.shape[-1], raster.shape[-2]]
                )
            if self.cfg.data.augmentation.flip:
                h, w = raster.shape[-2:]
                uv_pixels = self.normalized_to_pixel_uv(uv_gt, [w, h])
                image, raster, uv_pixels, heading, map_mask = random_flip(
                    image, raster, uv_pixels, heading, map_mask, seed
                )
                # Convert back to normalized coordinates
                uv_gt = self.pixel_to_normalized_uv(
                    uv_pixels, [raster.shape[-1], raster.shape[-2]]
                )

        # yaw = 90 - np.rad2deg(heading)

        # The dictionary to be returned
        data = {
            "index": idx,
            "name": image_name,
            "scene": image_data["map"],
        }

        if self.cfg.data.add_map_mask:
            map_mask_tensor = torch.from_numpy(
                np.ascontiguousarray(map_mask.copy())
            ).float()
            map_mask_tensor = (
                F.interpolate(
                    map_mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=(self.cfg.data.map_resize_dim, self.cfg.data.map_resize_dim),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .bool()
            )
            data["map_mask"] = map_mask_tensor

        raster_tensor = torch.from_numpy(np.ascontiguousarray(raster).copy())
        raster_tensor = (
            F.interpolate(
                raster_tensor.unsqueeze(0),
                size=(self.cfg.data.map_resize_dim, self.cfg.data.map_resize_dim),
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
            .long()
        )

        uv_gt = self.normalized_to_pixel_uv(
            uv_gt, [self.cfg.data.map_resize_dim, self.cfg.data.map_resize_dim]
        )
        uv_init = self.normalized_to_pixel_uv(
            uv_init, [self.cfg.data.map_resize_dim, self.cfg.data.map_resize_dim]
        )

        data = {
            **data,
            "image": image,
            "valid": valid,
            "camera": cam,
            "map": raster_tensor,
            "uv": torch.from_numpy(uv_gt.copy()).float(),
            "uv_init": torch.from_numpy(uv_init.copy()).float(),
            "roll_pitch_yaw": torch.tensor((roll, pitch, yaw)).float(),
            "pixels_per_meter": torch.tensor(self.cfg.data.pixel_per_meter).float(),
        }

        # Visualize the sample (remove this after debugging)
        # self.visualize_sample(data)

        return data
