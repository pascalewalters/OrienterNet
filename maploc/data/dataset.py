# Copyright (c) Meta Platforms, Inc. and affiliates.

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.transforms as tvf
from omegaconf import DictConfig, OmegaConf

from ..models.utils import deg2rad, rotmat2d
from ..osm.tiling import TileManager
from ..utils.geo import BoundaryBox
from ..utils.io import read_image
from ..utils.wrappers import Camera
from .image import pad_image, rectify_image, resize_image
from .utils import decompose_rotmat, random_flip, random_rot90
import torch.nn.functional as F


class MapLocDataset(torchdata.Dataset):
    default_cfg = {
        "seed": 0,
        "accuracy_gps": 15,
        "random": True,
        "num_threads": None,
        # map
        "num_classes": None,
        "pixel_per_meter": "???",
        "crop_size_meters": "???",
        "max_init_error": "???",
        "max_init_error_rotation": None,
        "init_from_gps": True,
        "return_gps": False,
        "force_camera_height": None,
        # pose priors
        "add_map_mask": False,
        "mask_radius": None,
        "mask_pad": 1,
        "prior_range_rotation": None,
        # image preprocessing
        "target_focal_length": None,
        "reduce_fov": None,
        "resize_image": None,
        "pad_to_square": False,  # legacy
        "pad_to_multiple": 32,
        "rectify_pitch": True,
        "augmentation": {
            "rot90": False,
            "flip": False,
            "image": {
                "apply": False,
                "brightness": 0.5,
                "contrast": 0.4,
                "saturation": 0.4,
                "hue": 0.5 / 3.14,
            },
        },
    }

    def __init__(
        self,
        stage: str,
        cfg: DictConfig,
        names: List[str],
        data: Dict[str, Any],
        image_dirs: Dict[str, Path],
        tile_managers: Dict[str, TileManager],
        image_ext: str = "",
    ):
        self.stage = stage
        self.cfg = deepcopy(cfg)
        self.data = data
        self.image_dirs = image_dirs
        self.tile_managers = tile_managers
        self.names = names
        self.image_ext = image_ext

        tfs = []
        if stage == "train" and cfg.augmentation.image.apply:
            args = OmegaConf.masked_copy(
                cfg.augmentation.image, ["brightness", "contrast", "saturation", "hue"]
            )
            tfs.append(tvf.ColorJitter(**args))
        self.tfs = tvf.Compose(tfs)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.stage == "train" and self.cfg.random:
            seed = None
        else:
            seed = [self.cfg.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)

        scene, image_id = self.names[idx]
        
        if self.cfg.init_from_gps:            
            latlon_gps = [
                torch.tensor([
                    self.data['latitude'][idx],
                    self.data['longitude'][idx]
                ]).numpy()
            ]
            # print(f"latlong coords: {latlon_gps}")
            xy_w_init = self.tile_managers[scene].projection.project(latlon_gps)
            


        # error = np.random.RandomState(seed).uniform(-1, 1, size=2)
        # xy_w_init += error * self.cfg.max_init_error

        bbox_tile = BoundaryBox(
            xy_w_init - self.cfg.crop_size_meters,
            xy_w_init + self.cfg.crop_size_meters,
        )
        # print(f"BBOX TILE min: {bbox_tile.min_}, max: {bbox_tile.max_}")
        return self.get_view(idx, scene, image_id, seed, bbox_tile)

    def get_view(self, idx, scene, image_id, seed, bbox_tile):
        data = {
            "index": idx,
            "name": image_id,
            "scene": scene,
            #"sequence": None,
        } 

        # Simple orientation from bearing
        roll, pitch = 0.0, 0.0  # Assuming flat ground
        yaw = self.data['bearing'][idx]  # Use bearing as yaw
        
        # Load and process image
        image = read_image(self.image_dirs[scene] / (image_id))
        
        # Create camera parameters for the image
        h, w = image.shape[:2]  # Get image dimensions
        cam_dict = {
            "model": "SIMPLE_RADIAL",
            "width": w,
            "height": h,
            "params": np.array([
                max(w, h) * 0.93,  # focal length estimate (based on ~66 degree FOV)
                w/2,               # cx (principal point x)
                h/2,               # cy (principal point y)
                0.1                # k1 (radial distortion)
            ])
        }
        cam = Camera.from_dict(cam_dict).float()
        image, valid = self.process_image(image, seed)

        # raster extraction
        canvas = self.tile_managers[scene].query(bbox_tile)
        
        # Get ground truth position from lat/long
        latlon_gt = torch.tensor([
            self.data['latitude'][idx],
            self.data['longitude'][idx]
        ]).numpy()
        
        # world coordinates
        xy_w_gt = self.tile_managers[scene].projection.project(latlon_gt)
        
        uv_gt = canvas.to_uv(xy_w_gt)
        uv_init = canvas.to_uv(bbox_tile.center)
        raster = canvas.raster
        
        if uv_gt.ndim > 1:
            uv_gt = uv_gt.squeeze() 

        # Map augmentations for training
        heading = np.deg2rad(90 - yaw)
        if self.stage == "train":
            if self.cfg.augmentation.rot90:
                raster, uv_gt, heading = random_rot90(raster, uv_gt, heading, seed)
            if self.cfg.augmentation.flip:
                image, raster, uv_gt, heading = random_flip(
                    image, raster, uv_gt, heading, seed
                )
        yaw = 90 - np.rad2deg(heading)

        # Optional: create mask for search area
        if self.cfg.add_map_mask:
            data["map_mask"] = torch.from_numpy(self.create_map_mask(canvas))

        return {
            **data,
            "image": image.cpu(),
            "valid": valid.cpu(),
            "camera": cam.cpu(),
            "canvas": canvas,
            "map": torch.from_numpy(np.ascontiguousarray(raster)).long().cpu(),
            "uv": torch.from_numpy(uv_gt).float().cpu(),
            "uv_init": torch.from_numpy(uv_init).float().cpu(),
            "roll_pitch_yaw": torch.tensor((roll, pitch, yaw)).float().cpu(),
            "pixels_per_meter": torch.tensor(canvas.ppm).float().cpu(),
        }

    def process_image(self, image, seed):
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
        if self.cfg.resize_image is not None:
            target_size = self.cfg.resize_image
            
            # First resize maintaining aspect ratio
            h, w = image.shape[-2:]
            scale = target_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # Resize keeping aspect ratio
            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            valid = F.interpolate(
                valid.unsqueeze(0).unsqueeze(0).float(),
                size=(new_h, new_w),
                mode='nearest'
            ).squeeze(0).squeeze(0).bool()
            
            # Create padded tensors
            padded_image = torch.zeros((3, target_size, target_size), dtype=image.dtype)
            padded_valid = torch.zeros((target_size, target_size), dtype=valid.dtype)
            
            pad_h = (target_size - new_h) // 2
            pad_w = (target_size - new_w) // 2
            
            padded_image[:, pad_h:pad_h+new_h, pad_w:pad_w+new_w] = image
            padded_valid[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = valid
            
            image = padded_image
            valid = padded_valid

        return image, valid

    def create_map_mask(self, canvas):
        map_mask = np.zeros(canvas.raster.shape[-2:], bool)
        radius = self.cfg.mask_radius or self.cfg.max_init_error
        mask_min, mask_max = np.round(
            canvas.to_uv(canvas.bbox.center)
            + np.array([[-1], [1]]) * (radius + self.cfg.mask_pad) * canvas.ppm
        ).astype(int)
        map_mask[mask_min[1] : mask_max[1], mask_min[0] : mask_max[0]] = True
        return map_mask
